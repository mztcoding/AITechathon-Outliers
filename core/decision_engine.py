"""
core/decision_engine.py
Multi-signal risk scoring and action recommendation engine.

Combines:
1. Gemini vision analysis (weighted highest — 60%)
2. YOLO detection confidence (20%)
3. Tracking frequency heuristic (20%)

Outputs:
- Risk score: 0–100
- Action: INSPECT_IMMEDIATELY | INSPECT_AT_NEXT_TOLL | NO_ACTION
- Alert payload for toll plaza notification

Design note:
The multi-signal approach means even if Gemini is unavailable,
we still produce a score (lower quality but functional).
This is the "graceful degradation" principle.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
import time
from loguru import logger

from config.settings import config
from core.gemini_client import GeminiResult
from core.tracker import Track
from core.redis_client import redis_client


ACTION_INSPECT_IMMEDIATELY = "INSPECT_IMMEDIATELY"
ACTION_INSPECT_NEXT_TOLL = "INSPECT_AT_NEXT_TOLL"
ACTION_NO_ACTION = "NO_ACTION"


@dataclass
class RiskAssessment:
    """Complete risk assessment for a single truck."""
    truck_id: str
    track_id: int

    # Component scores (0-100)
    gemini_score: float = 0.0
    yolo_score: float = 0.0
    frequency_score: float = 0.0

    # Final output
    risk_score: float = 0.0
    risk_level: str = "low"
    action: str = ACTION_NO_ACTION
    confidence: float = 0.0

    # Evidence
    signals: List[str] = field(default_factory=list)
    explanation: str = ""

    # Metadata
    timestamp: float = field(default_factory=time.time)
    bbox: tuple = field(default_factory=tuple)
    gemini_was_cached: bool = False
    gemini_available: bool = True

    def to_alert_payload(self) -> dict:
        """Generate alert payload for toll plaza notification."""
        return {
            "alert_type": "TRUCK_INSPECTION",
            "priority": self._priority_label(),
            "truck_id": self.truck_id,
            "track_id": self.track_id,
            "risk_score": round(self.risk_score, 1),
            "risk_level": self.risk_level,
            "action": self.action,
            "signals": self.signals,
            "explanation": self.explanation,
            "confidence": round(self.confidence, 2),
            "timestamp": self.timestamp,
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.timestamp)),
        }

    def _priority_label(self) -> str:
        if self.action == ACTION_INSPECT_IMMEDIATELY:
            return "P1_CRITICAL"
        elif self.action == ACTION_INSPECT_NEXT_TOLL:
            return "P2_STANDARD"
        return "P3_CLEAR"

    def to_dict(self) -> dict:
        return {
            "truck_id": self.truck_id,
            "track_id": self.track_id,
            "risk_score": round(self.risk_score, 1),
            "risk_level": self.risk_level,
            "action": self.action,
            "confidence": round(self.confidence, 2),
            "signals": self.signals,
            "explanation": self.explanation,
            "gemini_score": round(self.gemini_score, 1),
            "yolo_score": round(self.yolo_score, 1),
            "frequency_score": round(self.frequency_score, 1),
            "gemini_was_cached": self.gemini_was_cached,
            "timestamp": self.timestamp,
        }


class DecisionEngine:
    """
    Fuses multi-modal signals into an actionable risk score.

    Score composition:
    ┌─────────────────────────────────────────────┐
    │  Gemini vision analysis     → 60% weight    │
    │  YOLO detection confidence  → 20% weight    │
    │  Detection frequency        → 20% weight    │
    └─────────────────────────────────────────────┘

    Thresholds (configurable):
    - ≥75 → INSPECT_IMMEDIATELY
    - 45-74 → INSPECT_AT_NEXT_TOLL
    - <45 → NO_ACTION
    """

    def __init__(self):
        self.w_gemini = config.risk.weight_gemini
        self.w_yolo = config.risk.weight_yolo_conf
        self.w_freq = config.risk.weight_track_freq
        self._assessments: Dict[str, RiskAssessment] = {}

    def assess(
        self,
        track: Track,
        gemini_result: Optional[GeminiResult],
        truck_id: str,
    ) -> RiskAssessment:
        """
        Compute final risk assessment for a truck.

        Args:
            track: Current track state (contains detection history)
            gemini_result: Gemini analysis result (may be None if skipped)
            truck_id: Unique truck identifier
        """
        # Component 1: Gemini score (0-100)
        gemini_score, gemini_available = self._compute_gemini_score(gemini_result)

        # Component 2: YOLO detection confidence score (0-100)
        yolo_score = self._compute_yolo_score(track.confidence)

        # Component 3: Detection frequency score (0-100)
        freq_score = self._compute_frequency_score(track.detection_frequency)

        # Weighted combination
        if gemini_available:
            risk_score = (
                self.w_gemini * gemini_score
                + self.w_yolo * yolo_score
                + self.w_freq * freq_score
            )
        else:
            # Gemini unavailable — redistribute weight across local signals.
            # Boost frequency and hits-based signal for reliable local detection.
            hits_score = self._compute_hits_score(track.hits)
            risk_score = (
                0.40 * yolo_score
                + 0.35 * freq_score
                + 0.25 * hits_score
            )
            logger.debug(
                f"Gemini unavailable for {truck_id} — local heuristics: "
                f"yolo={yolo_score:.1f} freq={freq_score:.1f} hits={hits_score:.1f} → {risk_score:.1f}"
            )

        risk_score = max(0.0, min(100.0, risk_score))

        # Determine risk level from score
        risk_level = self._score_to_level(risk_score)

        # Determine action
        action = self._score_to_action(risk_score)

        # Aggregate signals from all sources
        signals = []
        if gemini_result and gemini_result.signals:
            signals.extend(gemini_result.signals)
        if track.detection_frequency > 1.0:
            signals.append(f"High detection frequency ({track.detection_frequency:.1f}/s — slow/stopped truck)")
        if track.confidence > 0.85:
            signals.append(f"High YOLO confidence ({track.confidence:.0%})")
        if track.hits >= 8:
            signals.append(f"Extended camera presence ({track.hits} frames — lingering vehicle)")
        if not gemini_available:
            signals.append("⚠ Gemini unavailable — score from local signals only")

        # Explanation
        explanation = self._build_explanation(
            gemini_result, risk_score, action, track
        )

        assessment = RiskAssessment(
            truck_id=truck_id,
            track_id=track.track_id,
            gemini_score=gemini_score,
            yolo_score=yolo_score,
            frequency_score=freq_score,
            risk_score=risk_score,
            risk_level=risk_level,
            action=action,
            confidence=gemini_result.confidence if gemini_result else 0.3,
            signals=signals,
            explanation=explanation,
            bbox=track.bbox,
            gemini_was_cached=gemini_result.was_cached if gemini_result else False,
            gemini_available=gemini_available,
        )

        # Store assessment
        self._assessments[truck_id] = assessment

        # Log high-risk detections
        if action == ACTION_INSPECT_IMMEDIATELY:
            logger.warning(
                f"🚨 HIGH RISK: Truck {truck_id} | Score: {risk_score:.1f} | "
                f"Action: {action}"
            )
        elif action == ACTION_INSPECT_NEXT_TOLL:
            logger.info(f"⚠️  MEDIUM: Truck {truck_id} | Score: {risk_score:.1f}")

        # Push alert to Redis queue for toll plaza
        if action != ACTION_NO_ACTION:
            alert = assessment.to_alert_payload()
            redis_client.push_alert(alert)

        return assessment

    def _compute_gemini_score(self, result: Optional[GeminiResult]) -> tuple:
        """Convert Gemini risk_level + confidence to 0-100 score."""
        if result is None or result.error:
            return 50.0, False  # Default to medium uncertainty

        level_base = {"low": 15.0, "medium": 55.0, "high": 88.0}
        base = level_base.get(result.risk_level, 50.0)

        # Adjust by confidence (confidence modulates how far from base we go)
        adjustment = (result.confidence - 0.5) * 20.0
        score = base + adjustment

        return max(0.0, min(100.0, score)), True

    def _compute_yolo_score(self, confidence: float) -> float:
        """
        Map YOLO detection confidence to a risk factor.
        High YOLO confidence doesn't mean high risk, but low confidence
        means we should be less certain of our assessment.
        We use this as a reliability multiplier.
        """
        # Normalize confidence: 0.45 (threshold) to 1.0 → 0 to 100
        normalized = (confidence - config.processing.yolo_confidence_threshold) / \
                     (1.0 - config.processing.yolo_confidence_threshold)
        return max(0.0, min(100.0, normalized * 100))

    def _compute_frequency_score(self, frequency: float) -> float:
        """
        Higher detection frequency → truck is slow / stopped → more suspicious.
        6+ detections/sec (stationary truck in frame) = max risk signal.
        Curve: 0/s→0, 2/s→50, 4/s→80, 6/s→100
        """
        # Use log-ish curve so moderate frequency still scores meaningfully
        import math
        if frequency <= 0:
            return 0.0
        score = min(100.0, (1 - math.exp(-frequency / 3.0)) * 110.0)
        return max(0.0, score)

    def _compute_hits_score(self, hits: int) -> float:
        """
        Number of frames the truck has been tracked is a signal of how
        long it has lingered in camera view — heavier/slower trucks stay longer.
        hits: 0→0, 5→50, 10→80, 20→100
        """
        import math
        if hits <= 0:
            return 0.0
        return max(0.0, min(100.0, (1 - math.exp(-hits / 8.0)) * 110.0))

    def _score_to_level(self, score: float) -> str:
        if score >= config.risk.inspect_immediately_threshold:
            return "high"
        elif score >= config.risk.inspect_next_toll_threshold:
            return "medium"
        return "low"

    def _score_to_action(self, score: float) -> str:
        if score >= config.risk.inspect_immediately_threshold:
            return ACTION_INSPECT_IMMEDIATELY
        elif score >= config.risk.inspect_next_toll_threshold:
            return ACTION_INSPECT_NEXT_TOLL
        return ACTION_NO_ACTION

    def _build_explanation(
        self,
        gemini_result: Optional[GeminiResult],
        risk_score: float,
        action: str,
        track: Track,
    ) -> str:
        """Build human-readable explanation for inspector."""
        parts = []

        if gemini_result and not gemini_result.error:
            parts.append(gemini_result.explanation)

        if action == ACTION_INSPECT_IMMEDIATELY:
            parts.append(f"Risk score {risk_score:.0f}/100 — immediate inspection required.")
        elif action == ACTION_INSPECT_NEXT_TOLL:
            parts.append(f"Risk score {risk_score:.0f}/100 — flag for next toll plaza check.")
        else:
            parts.append(f"Risk score {risk_score:.0f}/100 — within acceptable parameters.")

        if track.hits > 5:
            parts.append(f"Detected {track.hits} times over {track.detection_frequency:.1f} det/s.")

        return " ".join(parts)

    @property
    def all_assessments(self) -> List[RiskAssessment]:
        return list(self._assessments.values())

    @property
    def high_risk_count(self) -> int:
        return sum(1 for a in self._assessments.values() if a.action == ACTION_INSPECT_IMMEDIATELY)

    @property
    def medium_risk_count(self) -> int:
        return sum(1 for a in self._assessments.values() if a.action == ACTION_INSPECT_NEXT_TOLL)

    def get_priority_queue(self) -> List[RiskAssessment]:
        """Return assessments sorted by risk score (highest first)."""
        return sorted(
            [a for a in self._assessments.values() if a.action != ACTION_NO_ACTION],
            key=lambda x: x.risk_score,
            reverse=True,
        )

    def reset(self):
        self._assessments.clear()


# Singleton
decision_engine = DecisionEngine()
