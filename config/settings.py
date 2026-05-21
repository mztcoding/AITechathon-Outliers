"""
config/settings.py
Centralized configuration management for FreightGuard AI.
All parameters loaded from environment with sensible defaults.

FIX (2024): Moved all os.getenv() calls into __post_init__ to avoid
the dataclass field-default evaluation trap. Python evaluates field
defaults at CLASS DEFINITION time, not instantiation time — so any
os.getenv() in a field default captures the env *before* load_dotenv()
has run if another module imported settings first. Using __post_init__
guarantees env vars are read after load_dotenv() has populated os.environ.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Always reload .env — safe to call multiple times, later calls win
load_dotenv(override=True)


@dataclass
class GeminiConfig:
    api_key: str = ""
    model: str = "gemini-2.0-flash"
    rate_limit_per_minute: int = 15
    max_retries: int = 3
    timeout_seconds: int = 30

    def __post_init__(self):
        # Read from env at instantiation time (after load_dotenv has run)
        self.api_key = os.getenv("GEMINI_API_KEY", self.api_key)
        self.model = os.getenv("GEMINI_MODEL", self.model)
        self.rate_limit_per_minute = int(os.getenv("GEMINI_RATE_LIMIT", str(self.rate_limit_per_minute)))


@dataclass
class RedisConfig:
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str = None
    url: str = ""
    truck_cache_ttl: int = 300

    def __post_init__(self):
        self.host = os.getenv("REDIS_HOST", self.host)
        self.port = int(os.getenv("REDIS_PORT", str(self.port)))
        self.db = int(os.getenv("REDIS_DB", str(self.db)))
        self.password = os.getenv("REDIS_PASSWORD", "") or None
        self.url = os.getenv("REDIS_URL", self.url)
        self.truck_cache_ttl = int(os.getenv("TRUCK_CACHE_TTL", str(self.truck_cache_ttl)))


@dataclass
class ProcessingConfig:
    frame_sample_rate: int = 15
    yolo_confidence_threshold: float = 0.45
    min_bbox_area: int = 3000
    min_sharpness_score: float = 50.0
    yolo_model: str = "yolov8n.pt"

    def __post_init__(self):
        self.frame_sample_rate = int(os.getenv("FRAME_SAMPLE_RATE", str(self.frame_sample_rate)))
        self.yolo_confidence_threshold = float(os.getenv("YOLO_CONFIDENCE_THRESHOLD", str(self.yolo_confidence_threshold)))
        self.min_bbox_area = int(os.getenv("MIN_BBOX_AREA", str(self.min_bbox_area)))
        self.min_sharpness_score = float(os.getenv("MIN_SHARPNESS_SCORE", str(self.min_sharpness_score)))


@dataclass
class TrackerConfig:
    iou_threshold: float = 0.35
    max_track_age: int = 10
    min_track_hits: int = 2

    def __post_init__(self):
        self.iou_threshold = float(os.getenv("IOU_THRESHOLD", str(self.iou_threshold)))
        self.max_track_age = int(os.getenv("MAX_TRACK_AGE", str(self.max_track_age)))
        self.min_track_hits = int(os.getenv("MIN_TRACK_HITS", str(self.min_track_hits)))


@dataclass
class DeduplicationConfig:
    phash_similarity_threshold: int = 8

    def __post_init__(self):
        self.phash_similarity_threshold = int(
            os.getenv("PHASH_SIMILARITY_THRESHOLD", str(self.phash_similarity_threshold))
        )


@dataclass
class RiskConfig:
    weight_gemini: float = 0.60
    weight_yolo_conf: float = 0.20
    weight_track_freq: float = 0.20
    inspect_immediately_threshold: int = 75
    inspect_next_toll_threshold: int = 45

    def __post_init__(self):
        self.weight_gemini = float(os.getenv("WEIGHT_GEMINI", str(self.weight_gemini)))
        self.weight_yolo_conf = float(os.getenv("WEIGHT_YOLO_CONF", str(self.weight_yolo_conf)))
        self.weight_track_freq = float(os.getenv("WEIGHT_TRACK_FREQ", str(self.weight_track_freq)))


@dataclass
class AppConfig:
    gemini: GeminiConfig = None
    redis: RedisConfig = None
    processing: ProcessingConfig = None
    tracker: TrackerConfig = None
    deduplication: DeduplicationConfig = None
    risk: RiskConfig = None
    log_level: str = "INFO"
    debug: bool = False

    def __post_init__(self):
        # Instantiate sub-configs here so their __post_init__ runs NOW,
        # after load_dotenv() has already populated os.environ
        if self.gemini is None:
            self.gemini = GeminiConfig()
        if self.redis is None:
            self.redis = RedisConfig()
        if self.processing is None:
            self.processing = ProcessingConfig()
        if self.tracker is None:
            self.tracker = TrackerConfig()
        if self.deduplication is None:
            self.deduplication = DeduplicationConfig()
        if self.risk is None:
            self.risk = RiskConfig()
        self.log_level = os.getenv("LOG_LEVEL", self.log_level)
        self.debug = os.getenv("DEBUG", "false").lower() == "true"

    def reload_from_env(self):
        """Re-read .env and reinitialize all sub-configs. Call this after
        injecting a new API key at runtime (e.g. from the Streamlit sidebar)."""
        load_dotenv(override=True)
        self.gemini = GeminiConfig()
        self.redis = RedisConfig()
        self.processing = ProcessingConfig()
        self.tracker = TrackerConfig()
        self.deduplication = DeduplicationConfig()
        self.risk = RiskConfig()


# Singleton — sub-configs read env vars in their __post_init__
config = AppConfig()


class RedisKeys:
    FRAMES_QUEUE = "freightguard:frames_queue"
    TRUCK_CACHE = "freightguard:truck_cache:{truck_id}"
    GEMINI_RESULT = "freightguard:gemini_result:{truck_id}"
    PROCESSED_HASHES = "freightguard:processed_hashes"
    RATE_LIMITER = "freightguard:rate_limiter:gemini"
    SYSTEM_STATS = "freightguard:system_stats"
    ALERT_QUEUE = "freightguard:alert_queue"

    @staticmethod
    def truck_cache(truck_id: str) -> str:
        return f"freightguard:truck_cache:{truck_id}"

    @staticmethod
    def gemini_result(truck_id: str) -> str:
        return f"freightguard:gemini_result:{truck_id}"