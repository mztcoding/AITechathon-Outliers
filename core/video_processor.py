"""
core/video_processor.py
Main orchestration pipeline for FreightGuard AI.

This is the "brain" that connects all subsystems:
  Frame Sampler → Detector → Tracker → Deduplicator → Gemini → Decision Engine

Key optimization decisions made here:
1. Frame sampling (only process 1 in N frames)
2. Deduplication via pHash similarity check against Redis cache
3. Tracker-based skip (don't re-analyze confirmed trucks)
4. All decisions logged with reasoning for transparency
"""

import cv2
import numpy as np
import time
from typing import Optional, Generator, Tuple, List, Dict, Callable
from dataclasses import dataclass, field
from loguru import logger

from config.settings import config
from core.detector import TruckDetector, Detection
from core.tracker import IoUTracker, Track, TrackStatus
from core.gemini_client import gemini_client, GeminiResult
from core.decision_engine import decision_engine, RiskAssessment
from core.redis_client import redis_client
from core.utils import CostTracker, draw_bbox_on_frame, phash_distance


@dataclass
class FrameResult:
    """Results for a single processed frame."""
    frame_idx: int
    frame: np.ndarray                           # Annotated frame
    was_sampled: bool = False                   # Was this frame analyzed?
    detections: List[Detection] = field(default_factory=list)
    tracks: List[Track] = field(default_factory=list)
    assessments: List[RiskAssessment] = field(default_factory=list)
    gemini_calls_this_frame: int = 0
    gemini_skips_this_frame: int = 0
    processing_time_ms: float = 0.0


class VideoProcessor:
    """
    Main pipeline orchestrator.

    Processing flow for each sampled frame:
    ┌────────────────────────────────────────────────────────┐
    │ 1. Frame sampling check (skip if not sampled frame)    │
    │ 2. YOLO detection on sampled frame                     │
    │ 3. Quality gate (blur + area + confidence)             │
    │ 4. IoU tracker update → assign track IDs              │
    │ 5. For each CONFIRMED track:                           │
    │    a. Check if truck_id already in Redis cache         │
    │    b. Compute pHash → check against processed_hashes  │
    │    c. If similar hash found → SKIP (deduplication)    │
    │    d. Otherwise → send to Gemini                       │
    │ 6. Decision engine → compute risk scores               │
    │ 7. Annotate frame with bounding boxes + risk info      │
    │ 8. Update cost tracker                                 │
    └────────────────────────────────────────────────────────┘
    """

    def __init__(self):
        self.detector = TruckDetector()
        self.tracker = IoUTracker()
        self.cost_tracker = CostTracker()
        self._frame_count = 0
        self._processed_count = 0
        self._phash_threshold = config.deduplication.phash_similarity_threshold

    def process_video(
        self,
        video_path: str,
        on_frame: Optional[Callable[[FrameResult], None]] = None,
        max_frames: Optional[int] = None,
    ) -> Generator[FrameResult, None, None]:
        """
        Process a video file frame by frame.

        Args:
            video_path: Path to video file
            on_frame: Optional callback for each frame result
            max_frames: Stop after N frames (for testing)

        Yields:
            FrameResult for each frame (sampled or not)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(
            f"📹 Processing video: {fps:.1f}fps, {total_frames} frames, "
            f"sampling 1/{config.processing.frame_sample_rate}"
        )

        # Reset state for fresh video
        self.tracker = IoUTracker()
        self.cost_tracker.reset()
        decision_engine.reset()
        redis_client.flush_session()

        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if max_frames and frame_idx >= max_frames:
                    break

                t_start = time.perf_counter()

                result = self._process_frame(frame, frame_idx)

                result.processing_time_ms = (time.perf_counter() - t_start) * 1000

                if on_frame:
                    on_frame(result)

                yield result
                frame_idx += 1

        finally:
            cap.release()
            logger.info(
                f"✅ Video processing complete. "
                f"Frames: {frame_idx}, "
                f"Gemini calls: {self.cost_tracker.gemini_calls_made}, "
                f"Cost reduction: {self.cost_tracker.cost_reduction_pct:.1f}%"
            )

    def process_frame_bytes(self, frame_bytes: bytes, frame_idx: int) -> FrameResult:
        """Process a single frame from bytes (for streaming use cases)."""
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return self._process_frame(frame, frame_idx)

    def _process_frame(self, frame: np.ndarray, frame_idx: int) -> FrameResult:
        """Core frame processing logic."""
        self._frame_count += 1
        redis_client.increment_stat("frames_total")
        self.cost_tracker.record_frame(sampled=False)

        result = FrameResult(frame_idx=frame_idx, frame=frame.copy())

        # ── Step 1: Frame Sampling ─────────────────────────────────────────
        if frame_idx % config.processing.frame_sample_rate != 0:
            # Non-sampled frame: just return clean frame (no annotation)
            result.was_sampled = False
            return result

        result.was_sampled = True
        self._processed_count += 1
        redis_client.increment_stat("frames_sampled")
        self.cost_tracker.record_frame(sampled=True)

        # ── Step 2: YOLO Detection ─────────────────────────────────────────
        all_detections = self.detector.detect(frame, frame_idx)
        redis_client.increment_stat("detections_total", len(all_detections))

        # Filter to quality-passed detections only
        good_detections = [d for d in all_detections if d.quality_passed]
        redis_client.increment_stat("quality_passed", len(good_detections))

        result.detections = good_detections
        self.cost_tracker.detections_total += len(all_detections)
        self.cost_tracker.detections_quality_passed += len(good_detections)

        if not good_detections:
            return result

        # ── Step 3: Tracker Update ─────────────────────────────────────────
        tracker_inputs = [(d.bbox, d.confidence) for d in good_detections]
        updated_tracks = self.tracker.update(tracker_inputs, frame_idx)
        result.tracks = self.tracker.active_tracks

        # ── Step 4: Gemini Analysis for eligible tracks ────────────────────
        # Match detections to tracks (by bbox proximity)
        detection_map = self._match_detections_to_tracks(good_detections, updated_tracks)

        for track in self.tracker.active_tracks:
            logger.warning(f"🔍 TRACK {track.track_id}: status={track.status} hits={track.hits} ready={track.is_ready_for_gemini} analyzed={track.gemini_analyzed}")
            if track.gemini_analyzed:

                continue

            # Get corresponding detection
            detection = detection_map.get(track.track_id)
            if detection is None or detection.crop_resized is None:
                continue

            truck_id = f"truck_{track.track_id}"

            # ── Deduplication Check ────────────────────────────────────────

            # Check 1: Redis truck cache (exact track ID)
            if redis_client.has_truck_result(truck_id):
                logger.debug(f"⚡ Cache hit: {truck_id} — skipping Gemini")
                redis_client.increment_stat("tracker_deduped")
                self.cost_tracker.record_gemini_call(made=False)
                self.cost_tracker.detections_deduped += 1

                # Still apply cached decision to frame
                cached = redis_client.get_truck_result(truck_id)
                if cached:
                    risk_score = self._risk_level_to_score(cached["risk_level"])
                    track.risk_score = risk_score
                    track.action = self._score_to_action(risk_score)
                result.gemini_skips_this_frame += 1
                continue

            # Check 2: Perceptual hash similarity
            if detection.phash:
                if self._is_hash_duplicate(detection.phash):
                    logger.debug(f"🔍 pHash dedup: {truck_id}")
                    redis_client.increment_stat("hash_deduped")
                    self.cost_tracker.record_gemini_call(made=False)
                    self.cost_tracker.detections_deduped += 1
                    result.gemini_skips_this_frame += 1
                    continue

                # Register this hash as processed
                redis_client.add_processed_hash(detection.phash)
                # Store locally for this session
                track.phash = detection.phash

            # ── Gemini Analysis ────────────────────────────────────────────
            self.cost_tracker.record_gemini_call(made=True)
            result.gemini_calls_this_frame += 1

            gemini_result = gemini_client.analyze_truck(
                truck_id=truck_id,
                crop_image=detection.crop_resized,
            )

            # ── Decision Engine ────────────────────────────────────────────
            assessment = decision_engine.assess(
                track=track,
                gemini_result=gemini_result,
                truck_id=truck_id,
            )

            result.assessments.append(assessment)

            # Update track with final decision
            self.tracker.mark_analyzed(
                track.track_id,
                risk_score=assessment.risk_score,
                action=assessment.action,
            )

        # ── Step 5: Annotate Frame ─────────────────────────────────────────
        annotated = self._annotate_frame(frame.copy(), result)
        result.frame = annotated

        return result

    def _match_detections_to_tracks(
        self,
        detections: List[Detection],
        tracks: List[Track],
    ) -> Dict[int, Detection]:
        """
        Simple nearest-bbox matching of detections to track IDs.
        Returns {track_id: detection}.
        """
        from core.utils import compute_iou
        mapping = {}

        for track in tracks:
            best_iou = 0.0
            best_det = None
            for det in detections:
                iou = compute_iou(track.bbox, det.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_det = det
            if best_det and best_iou > 0.3:
                mapping[track.track_id] = best_det

        return mapping

    def _is_hash_duplicate(self, phash: str) -> bool:
        """
        Check if this image is perceptually similar to any processed hash.
        Uses Hamming distance for fuzzy matching.
        """
        processed = redis_client.get_all_processed_hashes()
        if not processed:
            return False

        for existing_hash in processed:
            try:
                dist = phash_distance(phash, existing_hash)
                if dist <= self._phash_threshold:
                    return True
            except Exception:
                continue

        return False

    def _annotate_frame(self, frame: np.ndarray, result: FrameResult) -> np.ndarray:
        """Draw detection boxes and risk indicators on frame."""
        if not result.was_sampled:
            return frame

        # Get latest assessments from decision engine
        assessment_map = {a.track_id: a for a in decision_engine.all_assessments}

        for track in result.tracks:
            if track.bbox:
                # Look up latest assessment for this track
                assessment = assessment_map.get(track.track_id)
                risk_score = assessment.risk_score if assessment else track.risk_score
                action = assessment.action if assessment else track.action

                if risk_score < 0:
                    risk_score = 0  # Unscored track

                frame = draw_bbox_on_frame(
                    frame=frame,
                    bbox=track.bbox,
                    track_id=track.track_id,
                    risk_score=risk_score,
                    action=action if action != "PENDING" else "NO_ACTION",
                    confidence=track.confidence,
                )

        # Add overlay stats
        frame = self._add_stats_overlay(frame, result)

        return frame

    def _add_stats_overlay(self, frame: np.ndarray, result: FrameResult) -> np.ndarray:
        """Add a compact stats overlay to the frame."""
        h, w = frame.shape[:2]
        stats = self.cost_tracker.summary()

        overlay_lines = [
            f"FreightGuard AI | Frame {result.frame_idx}",
            f"Tracks: {len(result.tracks)} | Gemini calls: {stats['gemini_calls_made']}",
            f"Cost saved: ${stats['cost_saved_usd']:.4f} ({stats['cost_reduction_pct']:.1f}%)",
        ]

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (400, 70), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        for i, line in enumerate(overlay_lines):
            cv2.putText(
                frame, line, (10, 25 + i * 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 255, 140), 1, cv2.LINE_AA
            )

        return frame

    def _risk_level_to_score(self, level: str) -> float:
        return {"low": 20.0, "medium": 60.0, "high": 88.0}.get(level, 50.0)

    def _score_to_action(self, score: float) -> str:
        if score >= config.risk.inspect_immediately_threshold:
            return "INSPECT_IMMEDIATELY"
        elif score >= config.risk.inspect_next_toll_threshold:
            return "INSPECT_AT_NEXT_TOLL"
        return "NO_ACTION"

    @property
    def stats(self) -> dict:
        return {
            "cost_tracker": self.cost_tracker.summary(),
            "tracker": self.tracker.get_stats(),
            "decision": {
                "high_risk": decision_engine.high_risk_count,
                "medium_risk": decision_engine.medium_risk_count,
                "priority_queue_size": len(decision_engine.get_priority_queue()),
            },
        }
