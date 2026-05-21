"""
core/tracker.py
Lightweight IoU-based Multi-Object Tracker for FreightGuard AI.

Design philosophy:
- ZERO ML dependencies (no DeepSORT overhead for hackathon edge deployment)
- Simple IoU matching sufficient for highway cameras (trucks don't teleport)
- Assigns persistent track IDs to avoid re-processing the same truck
- Flags tracks as: NEW | ACTIVE | LOST | CONFIRMED
- CONFIRMED tracks = eligible for Gemini analysis

Why not SORT/DeepSORT?
- SORT requires scipy linear assignment (added complexity)
- DeepSORT requires a Re-ID model (heavy)
- For fixed camera + slow-moving trucks, IoU matching achieves ~95% of SORT accuracy
- We can add SORT later as a drop-in upgrade
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum
import time

from core.utils import compute_iou
from config.settings import config


class TrackStatus(str, Enum):
    NEW = "new"            # First detection
    ACTIVE = "active"      # Matched in recent frames
    LOST = "lost"          # Unmatched for some frames
    CONFIRMED = "confirmed"  # Stable enough for Gemini analysis
    ANALYZED = "analyzed"  # Gemini has processed this truck


@dataclass
class Track:
    """Represents a persistent truck track across frames."""
    track_id: int
    bbox: Tuple[int, int, int, int]   # (x1, y1, x2, y2)
    confidence: float
    status: TrackStatus = TrackStatus.NEW
    hits: int = 1                      # Total matched frames
    consecutive_hits: int = 1          # Current streak
    consecutive_misses: int = 0        # Current miss streak
    last_seen_frame: int = 0
    first_seen_frame: int = 0
    phash: Optional[str] = None        # Best perceptual hash
    gemini_analyzed: bool = False      # Has Gemini processed this?
    risk_score: float = -1.0           # -1 = not yet scored
    action: str = "PENDING"
    detection_timestamps: List[float] = field(default_factory=list)

    def update(self, bbox: Tuple, confidence: float, frame_idx: int):
        """Update track with new matching detection."""
        self.bbox = bbox
        self.confidence = max(self.confidence, confidence)  # Keep best confidence
        self.hits += 1
        self.consecutive_hits += 1
        self.consecutive_misses = 0
        self.last_seen_frame = frame_idx
        self.detection_timestamps.append(time.time())
        # Keep only last 30 timestamps
        if len(self.detection_timestamps) > 30:
            self.detection_timestamps = self.detection_timestamps[-30:]

        # Promote to CONFIRMED after min_hits
        if self.hits >= config.tracker.min_track_hits and self.status == TrackStatus.NEW:
            self.status = TrackStatus.CONFIRMED
        elif self.status != TrackStatus.ANALYZED:
            self.status = TrackStatus.ACTIVE

    def mark_missed(self):
        """Track not matched in this frame."""
        self.consecutive_hits = 0
        self.consecutive_misses += 1
        if self.consecutive_misses > 2:
            self.status = TrackStatus.LOST

    @property
    def is_ready_for_gemini(self) -> bool:
        """
        True if track is stable enough and hasn't been analyzed yet.
        Prevents sending brand-new or fleeting detections to Gemini.
        """
        return (
            self.status in [TrackStatus.CONFIRMED, TrackStatus.ACTIVE]
            and self.hits >= config.tracker.min_track_hits
            and not self.gemini_analyzed
        )

    @property
    def detection_frequency(self) -> float:
        """
        How frequently this truck has been detected (detections/second).
        Higher frequency → truck is slow-moving or stopped → higher inspection priority.
        """
        if len(self.detection_timestamps) < 2:
            return 0.0
        duration = self.detection_timestamps[-1] - self.detection_timestamps[0]
        if duration <= 0:
            return 0.0
        return len(self.detection_timestamps) / duration


class IoUTracker:
    """
    IoU-based multi-object tracker.

    Algorithm:
    1. For each new detection, compute IoU with all active tracks
    2. Greedy match: highest IoU pair first (if > threshold)
    3. Unmatched detections → new tracks
    4. Unmatched tracks → increment miss counter → drop if too old
    """

    def __init__(self):
        self._next_id = 1
        self._tracks: Dict[int, Track] = {}
        self._iou_threshold = config.tracker.iou_threshold
        self._max_age = config.tracker.max_track_age

    @property
    def active_tracks(self) -> List[Track]:
        """All non-LOST tracks."""
        return [t for t in self._tracks.values() if t.status != TrackStatus.LOST]

    @property
    def confirmed_tracks(self) -> List[Track]:
        return [t for t in self._tracks.values() if t.status == TrackStatus.CONFIRMED]

    def update(
        self,
        detections: List[Tuple[Tuple[int, int, int, int], float]],
        frame_idx: int,
    ) -> List[Track]:
        """
        Update tracker with new detections.

        Args:
            detections: List of (bbox, confidence) tuples
            frame_idx: Current frame index

        Returns:
            List of updated/new tracks for this frame
        """
        if not detections:
            # Mark all active tracks as missed
            for track in list(self._tracks.values()):
                if track.status != TrackStatus.LOST:
                    track.mark_missed()
            self._prune_lost_tracks()
            return []

        active = self.active_tracks

        if not active:
            # No existing tracks — create new ones for all detections
            return self._create_new_tracks(detections, frame_idx)

        # Compute IoU matrix: active_tracks × detections
        iou_matrix = self._compute_iou_matrix(active, detections)

        # Greedy matching
        matched_tracks, matched_dets = self._greedy_match(iou_matrix)

        updated_tracks = []

        # Update matched tracks
        for track_idx, det_idx in zip(matched_tracks, matched_dets):
            bbox, conf = detections[det_idx]
            active[track_idx].update(bbox, conf, frame_idx)
            updated_tracks.append(active[track_idx])

        # Mark unmatched tracks as missed
        unmatched_track_indices = set(range(len(active))) - set(matched_tracks)
        for idx in unmatched_track_indices:
            active[idx].mark_missed()

        # Create new tracks for unmatched detections
        unmatched_det_indices = set(range(len(detections))) - set(matched_dets)
        new_det_list = [detections[i] for i in sorted(unmatched_det_indices)]
        new_tracks = self._create_new_tracks(new_det_list, frame_idx)
        updated_tracks.extend(new_tracks)

        # Prune old lost tracks
        self._prune_lost_tracks()

        return updated_tracks

    def get_track(self, track_id: int) -> Optional[Track]:
        return self._tracks.get(track_id)

    def mark_analyzed(self, track_id: int, risk_score: float, action: str):
        """Mark a track as Gemini-analyzed with its result."""
        if track_id in self._tracks:
            self._tracks[track_id].gemini_analyzed = True
            self._tracks[track_id].status = TrackStatus.ANALYZED
            self._tracks[track_id].risk_score = risk_score
            self._tracks[track_id].action = action

    def _compute_iou_matrix(
        self,
        tracks: List[Track],
        detections: List[Tuple],
    ) -> List[List[float]]:
        """Compute IoU for all track-detection pairs."""
        matrix = []
        for track in tracks:
            row = []
            for bbox, _ in detections:
                iou = compute_iou(track.bbox, bbox)
                row.append(iou)
            matrix.append(row)
        return matrix

    def _greedy_match(
        self,
        iou_matrix: List[List[float]],
    ) -> Tuple[List[int], List[int]]:
        """
        Greedy IoU matching (simpler but effective for sparse scenes).
        For dense scenes, Hungarian algorithm (scipy.linear_sum_assignment) is better.
        """
        matched_tracks = []
        matched_dets = []
        used_tracks = set()
        used_dets = set()

        # Flatten and sort all (iou, track_idx, det_idx) by IoU descending
        pairs = []
        for t_idx, row in enumerate(iou_matrix):
            for d_idx, iou in enumerate(row):
                if iou >= self._iou_threshold:
                    pairs.append((iou, t_idx, d_idx))

        pairs.sort(key=lambda x: x[0], reverse=True)

        for iou, t_idx, d_idx in pairs:
            if t_idx in used_tracks or d_idx in used_dets:
                continue
            matched_tracks.append(t_idx)
            matched_dets.append(d_idx)
            used_tracks.add(t_idx)
            used_dets.add(d_idx)

        return matched_tracks, matched_dets

    def _create_new_tracks(
        self,
        detections: List[Tuple],
        frame_idx: int,
    ) -> List[Track]:
        """Create new Track objects for unmatched detections."""
        new_tracks = []
        for bbox, confidence in detections:
            track = Track(
                track_id=self._next_id,
                bbox=bbox,
                confidence=confidence,
                first_seen_frame=frame_idx,
                last_seen_frame=frame_idx,
                detection_timestamps=[time.time()],
            )
            self._tracks[self._next_id] = track
            self._next_id += 1
            new_tracks.append(track)
        return new_tracks

    def _prune_lost_tracks(self):
        """Remove tracks that have been lost too long."""
        to_delete = [
            tid for tid, t in self._tracks.items()
            if t.status == TrackStatus.LOST and t.consecutive_misses > self._max_age
        ]
        for tid in to_delete:
            del self._tracks[tid]

    def get_stats(self) -> dict:
        return {
            "total_tracks_created": self._next_id - 1,
            "active_tracks": len(self.active_tracks),
            "confirmed_tracks": len(self.confirmed_tracks),
            "analyzed_tracks": sum(1 for t in self._tracks.values() if t.gemini_analyzed),
        }
