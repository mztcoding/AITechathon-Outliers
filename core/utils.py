"""
core/utils.py
Shared utility functions for FreightGuard AI.
Covers: image quality, perceptual hashing, logging setup, cost tracking.
"""

import cv2
import numpy as np
import imagehash
from PIL import Image
from loguru import logger
from typing import Optional, Tuple
import base64
import io
import time
import sys

from config.settings import config


def setup_logger():
    """Configure structured logging."""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
        level=config.log_level,
        colorize=True,
    )
    return logger


def compute_blur_score(image: np.ndarray) -> float:
    """
    Compute Laplacian variance as blur measure.
    Higher = sharper. Threshold typically ~50-100.
    Cost: negligible (pure numpy)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_brightness(image: np.ndarray) -> float:
    """Return mean pixel brightness (0-255)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return float(np.mean(gray))


def compute_phash(image: np.ndarray, hash_size: int = 8) -> str:
    """
    Compute perceptual hash (pHash) of an image crop.
    Returns 64-bit hex string.
    Two visually similar images → Hamming distance < threshold.
    Cost: ~1ms, zero network.
    """
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return str(imagehash.phash(pil_img, hash_size=hash_size))


def phash_distance(hash1: str, hash2: str) -> int:
    """Compute Hamming distance between two pHash hex strings."""
    h1 = imagehash.hex_to_hash(hash1)
    h2 = imagehash.hex_to_hash(hash2)
    return h1 - h2


def is_image_quality_acceptable(
    image: np.ndarray,
    min_sharpness: float = None,
    min_brightness: float = 30.0,
    max_brightness: float = 220.0,
) -> Tuple[bool, str]:
    """
    Quality gate before sending to Gemini.
    Returns (passes, rejection_reason).
    """
    min_sharpness = min_sharpness or config.processing.min_sharpness_score

    blur = compute_blur_score(image)
    if blur < min_sharpness:
        return False, f"Too blurry (score={blur:.1f} < {min_sharpness})"

    brightness = compute_brightness(image)
    if brightness < min_brightness:
        return False, f"Too dark (brightness={brightness:.1f})"
    if brightness > max_brightness:
        return False, f"Overexposed (brightness={brightness:.1f})"

    return True, "OK"


def crop_truck_region(frame: np.ndarray, bbox: Tuple[int, int, int, int], padding: int = 10) -> np.ndarray:
    """
    Safely crop truck region from frame with optional padding.
    bbox format: (x1, y1, x2, y2)
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    return frame[y1:y2, x1:x2].copy()


def encode_image_base64(image: np.ndarray, format: str = "JPEG", quality: int = 85) -> str:
    """
    Encode numpy image to base64 string for Gemini API.
    Uses JPEG compression to minimize token costs.
    """
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    buffer = io.BytesIO()
    pil_img.save(buffer, format=format, quality=quality)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def resize_for_api(image: np.ndarray, max_dim: int = 512) -> np.ndarray:
    """
    Resize image to max dimension while preserving aspect ratio.
    Smaller images = fewer tokens = lower cost.
    512px is sufficient for Gemini to assess truck loading.
    """
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def compute_bbox_area(bbox: Tuple[int, int, int, int]) -> int:
    """Compute bounding box area in pixels."""
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def compute_iou(box1: Tuple, box2: Tuple) -> float:
    """
    Compute Intersection over Union between two bboxes.
    Format: (x1, y1, x2, y2)
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def draw_bbox_on_frame(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    track_id: int,
    risk_score: float,
    action: str,
    confidence: float,
) -> np.ndarray:
    """
    Draw styled bounding box with risk info on frame.
    Color-coded by risk level.
    """
    x1, y1, x2, y2 = bbox

    # Color by risk
    if risk_score >= 75:
        color = (0, 0, 255)      # Red — HIGH
        label_bg = (0, 0, 200)
    elif risk_score >= 45:
        color = (0, 165, 255)    # Orange — MEDIUM
        label_bg = (0, 130, 200)
    else:
        color = (0, 200, 80)     # Green — LOW
        label_bg = (0, 150, 60)

    thickness = 3 if risk_score >= 75 else 2

    # Draw bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Prepare label text
    action_short = {
        "INSPECT_IMMEDIATELY": "⚠ INSPECT NOW",
        "INSPECT_AT_NEXT_TOLL": "→ NEXT TOLL",
        "NO_ACTION": "✓ CLEAR",
    }.get(action, action)

    label = f"T#{track_id} | Risk:{risk_score:.0f}% | {action_short}"

    # Draw label background
    (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_y = max(y1 - 10, text_h + 5)
    cv2.rectangle(frame, (x1, label_y - text_h - 5), (x1 + text_w + 4, label_y + baseline), label_bg, -1)

    # Draw label text
    cv2.putText(frame, label, (x1 + 2, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


class CostTracker:
    """Track API usage and compute savings estimates."""

    # Gemini Flash pricing (approximate, per image)
    GEMINI_COST_PER_CALL = 0.000315  # USD

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_frames_processed = 0
        self.frames_sampled = 0
        self.detections_total = 0
        self.detections_quality_passed = 0
        self.detections_deduped = 0
        self.gemini_calls_made = 0
        self.gemini_calls_saved = 0
        self.start_time = time.time()

    def record_frame(self, sampled: bool):
        self.total_frames_processed += 1
        if sampled:
            self.frames_sampled += 1

    def record_detection(self, quality_passed: bool, was_deduped: bool):
        self.detections_total += 1
        if quality_passed:
            self.detections_quality_passed += 1
        if was_deduped:
            self.detections_deduped += 1

    def record_gemini_call(self, made: bool):
        if made:
            self.gemini_calls_made += 1
        else:
            self.gemini_calls_saved += 1

    @property
    def naive_gemini_calls(self) -> int:
        """How many calls would naive pipeline have made."""
        return self.detections_total

    @property
    def actual_cost_usd(self) -> float:
        return self.gemini_calls_made * self.GEMINI_COST_PER_CALL

    @property
    def naive_cost_usd(self) -> float:
        return self.naive_gemini_calls * self.GEMINI_COST_PER_CALL

    @property
    def cost_saved_usd(self) -> float:
        return self.naive_cost_usd - self.actual_cost_usd

    @property
    def cost_reduction_pct(self) -> float:
        if self.naive_cost_usd == 0:
            return 0.0
        return (self.cost_saved_usd / self.naive_cost_usd) * 100

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    def summary(self) -> dict:
        return {
            "total_frames": self.total_frames_processed,
            "frames_sampled": self.frames_sampled,
            "detections_total": self.detections_total,
            "detections_quality_passed": self.detections_quality_passed,
            "detections_deduped": self.detections_deduped,
            "gemini_calls_made": self.gemini_calls_made,
            "gemini_calls_saved": self.gemini_calls_saved,
            "naive_cost_usd": round(self.naive_cost_usd, 4),
            "actual_cost_usd": round(self.actual_cost_usd, 6),
            "cost_saved_usd": round(self.cost_saved_usd, 4),
            "cost_reduction_pct": round(self.cost_reduction_pct, 1),
            "elapsed_seconds": round(self.elapsed_seconds, 1),
        }
