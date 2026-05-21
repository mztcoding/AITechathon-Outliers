"""
core/detector.py
YOLOv8n-based truck detector with multi-layer quality filtering.

Design:
- Uses YOLOv8n (nano) — fastest variant, ~30ms on CPU per frame
- Only detects relevant vehicle classes (trucks, buses, heavy vehicles)
- Applies confidence + area + quality gates BEFORE tracking or API calls
- Returns standardized Detection objects
- Handles corrupted/partial model files gracefully by deleting and re-downloading

YOLO COCO class IDs for vehicles:
  2 = car, 3 = motorcycle, 5 = bus, 7 = truck
  We focus on: 5 (bus) and 7 (truck) — large vehicles likely to be overloaded
"""

import cv2
import numpy as np
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple
from loguru import logger

from config.settings import config
from core.utils import (
    compute_blur_score,
    compute_bbox_area,
    crop_truck_region,
    compute_phash,
    resize_for_api,
)

# YOLO class IDs we care about
TARGET_CLASSES = {5: "bus", 7: "truck"}


@dataclass
class Detection:
    """Normalized detection output from YOLO."""
    bbox: Tuple[int, int, int, int]    # (x1, y1, x2, y2) absolute pixels
    confidence: float
    class_id: int
    class_name: str
    bbox_area: int
    crop: Optional[np.ndarray] = None
    crop_resized: Optional[np.ndarray] = None
    phash: Optional[str] = None
    blur_score: float = 0.0
    quality_passed: bool = True
    rejection_reason: str = ""


class TruckDetector:
    """
    Wraps YOLOv8n for truck/bus detection with quality gating.
    Handles corrupted model files by deleting and re-downloading cleanly.
    """

    def __init__(self):
        self._model = None
        self._model_loaded = False
        self._load_model()

    def _is_model_valid(self, model_path: str) -> bool:
        """
        Check if the model file exists and is a valid zip/PyTorch file.
        A partial download produces a file that is not a valid zip archive.
        """
        path = Path(model_path)
        if not path.exists():
            return False
        # PyTorch .pt files are zip archives — must be > 1MB to be complete
        # yolov8n.pt is ~6.2MB; anything under 1MB is definitely corrupted
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb < 1.0:
            logger.warning(
                f"Model file '{model_path}' is only {size_mb:.2f}MB — "
                f"likely a corrupted/partial download. Deleting."
            )
            path.unlink()
            return False
        # Also try opening as a zip to validate the central directory
        import zipfile
        try:
            with zipfile.ZipFile(path, 'r') as zf:
                zf.namelist()  # triggers central directory read
            return True
        except zipfile.BadZipFile:
            logger.warning(
                f"Model file '{model_path}' failed zip validation "
                f"(corrupted download). Deleting and will re-download."
            )
            path.unlink()
            return False
        except Exception:
            # If we can't validate, let YOLO try anyway
            return True

    def _load_model(self):
        """
        Load YOLOv8n model with corruption detection and clean re-download.

        If the .pt file exists but is corrupted (partial download),
        it is deleted so ultralytics can re-download it cleanly.
        """
        try:
            from ultralytics import YOLO
            model_path = config.processing.yolo_model

            # Validate before loading — delete if corrupted
            self._is_model_valid(model_path)

            logger.info(f"Loading YOLO model: {model_path}")
            self._model = YOLO(model_path)

            # Warm up with dummy inference to catch any runtime issues early
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self._model(dummy, verbose=False)

            self._model_loaded = True
            logger.info("✅ YOLOv8n loaded and warmed up")

        except ImportError:
            logger.error("ultralytics not installed. Run: pip install ultralytics")
            raise

        except Exception as e:
            err = str(e)
            # Detect the specific corrupted zip error from PyTorch
            if "central directory" in err or "zip archive" in err or "miniz" in err:
                model_path = config.processing.yolo_model
                p = Path(model_path)
                if p.exists():
                    logger.warning(
                        f"Detected corrupted model file: {model_path}\n"
                        f"Deleting and re-downloading on next load attempt."
                    )
                    p.unlink()
                raise RuntimeError(
                    f"YOLO model file was corrupted and has been deleted.\n"
                    f"Please restart the app — the model will re-download cleanly.\n"
                    f"Original error: {err}"
                ) from e
            else:
                logger.error(f"YOLO load failed: {e}")
                raise

    def detect(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
    ) -> List[Detection]:
        """
        Run truck detection on a single frame.
        Returns list of Detection objects that passed all quality gates.
        """
        if not self._model_loaded:
            return []

        try:
            results = self._model(
                frame,
                classes=list(TARGET_CLASSES.keys()),
                conf=config.processing.yolo_confidence_threshold,
                verbose=False,
                imgsz=640,
            )
        except Exception as e:
            logger.error(f"YOLO inference failed: {e}")
            return []

        detections = []
        raw_result = results[0]

        if raw_result.boxes is None or len(raw_result.boxes) == 0:
            return []

        for box in raw_result.boxes:
            try:
                detection = self._process_box(frame, box)
                if detection:
                    detections.append(detection)
            except Exception as e:
                logger.debug(f"Box processing error: {e}")
                continue

        logger.debug(f"Frame {frame_idx}: {len(detections)} detections passed quality gates")
        return detections

    def _process_box(self, frame: np.ndarray, box) -> Optional[Detection]:
        """Process a single YOLO detection box through quality gates."""
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = TARGET_CLASSES.get(class_id, "unknown")
        bbox = (x1, y1, x2, y2)

        # Gate 1: Area threshold
        area = compute_bbox_area(bbox)
        if area < config.processing.min_bbox_area:
            return Detection(
                bbox=bbox, confidence=confidence, class_id=class_id,
                class_name=class_name, bbox_area=area,
                quality_passed=False,
                rejection_reason=f"Too small: {area}px² < {config.processing.min_bbox_area}px²",
            )

        crop = crop_truck_region(frame, bbox, padding=8)
        if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
            return None

        # Gate 2: Blur/sharpness check
        blur_score = compute_blur_score(crop)
        if blur_score < config.processing.min_sharpness_score:
            return Detection(
                bbox=bbox, confidence=confidence, class_id=class_id,
                class_name=class_name, bbox_area=area, crop=crop,
                blur_score=blur_score, quality_passed=False,
                rejection_reason=f"Too blurry: {blur_score:.1f}",
            )

        phash = compute_phash(crop)
        crop_resized = resize_for_api(crop, max_dim=512)

        return Detection(
            bbox=bbox,
            confidence=confidence,
            class_id=class_id,
            class_name=class_name,
            bbox_area=area,
            crop=crop,
            crop_resized=crop_resized,
            phash=phash,
            blur_score=blur_score,
            quality_passed=True,
        )

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """Batch inference for efficiency."""
        if not self._model_loaded or not frames:
            return [[] for _ in frames]
        try:
            results = self._model(
                frames,
                classes=list(TARGET_CLASSES.keys()),
                conf=config.processing.yolo_confidence_threshold,
                verbose=False,
                imgsz=640,
            )
            return [
                [d for d in [self._process_box(frames[i], box)
                              for box in r.boxes or []] if d]
                for i, r in enumerate(results)
            ]
        except Exception as e:
            logger.error(f"Batch detection failed: {e}")
            return [[] for _ in frames]

    @property
    def is_loaded(self) -> bool:
        return self._model_loaded