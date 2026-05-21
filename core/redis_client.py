"""
core/redis_client.py
Redis abstraction layer for FreightGuard AI.

Responsibilities:
- Frame queue management
- Truck result caching with TTL
- Processed hash set (deduplication)
- Gemini rate limiting (token bucket)
- System stats aggregation
- Alert queue for toll plaza notifications
"""

import json
import time
from typing import Optional, Any, List, Dict
from loguru import logger

import redis

from config.settings import config, RedisKeys


class FreightRedisClient:
    """
    Thread-safe Redis client wrapper for FreightGuard AI.
    All operations use namespaced keys to prevent collisions.
    """

    def __init__(self):
        self._client: Optional[redis.Redis] = None
        self._connected = False
        self._mock_mode = False  # Fallback if Redis unavailable
        self._mock_store: Dict[str, Any] = {}
        self._mock_sets: Dict[str, set] = {}
        self._connect()

    def _connect(self):
        """Establish Redis connection with fallback to mock mode."""
        try:
            cfg = config.redis
            if cfg.url:
                self._client = redis.from_url(cfg.url, decode_responses=True, socket_timeout=3)
            else:
                self._client = redis.Redis(
                    host=cfg.host,
                    port=cfg.port,
                    db=cfg.db,
                    password=cfg.password or None,
                    decode_responses=True,
                    socket_timeout=3,
                    socket_connect_timeout=3,
                )
            # Test connection
            self._client.ping()
            self._connected = True
            logger.info(f"✅ Redis connected: {cfg.host}:{cfg.port}")
        except Exception as e:
            logger.warning(f"⚠️  Redis unavailable ({e}). Running in MOCK mode (no persistence).")
            self._mock_mode = True
            self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected and not self._mock_mode

    # ─── Frame Queue ──────────────────────────────────────────────────────────

    def enqueue_frame(self, frame_data: dict) -> bool:
        """Push a frame analysis task to the queue."""
        payload = json.dumps(frame_data)
        try:
            if self._mock_mode:
                self._mock_store.setdefault(RedisKeys.FRAMES_QUEUE, []).append(payload)
                return True
            self._client.rpush(RedisKeys.FRAMES_QUEUE, payload)
            return True
        except Exception as e:
            logger.error(f"Queue push failed: {e}")
            return False

    def dequeue_frame(self) -> Optional[dict]:
        """Pop a frame from the queue (blocking not used for simplicity)."""
        try:
            if self._mock_mode:
                queue = self._mock_store.get(RedisKeys.FRAMES_QUEUE, [])
                if queue:
                    return json.loads(queue.pop(0))
                return None
            result = self._client.lpop(RedisKeys.FRAMES_QUEUE)
            return json.loads(result) if result else None
        except Exception as e:
            logger.error(f"Queue pop failed: {e}")
            return None

    def get_queue_length(self) -> int:
        """Return current frame queue depth."""
        try:
            if self._mock_mode:
                return len(self._mock_store.get(RedisKeys.FRAMES_QUEUE, []))
            return self._client.llen(RedisKeys.FRAMES_QUEUE)
        except:
            return 0

    # ─── Truck Result Cache ───────────────────────────────────────────────────

    def cache_truck_result(self, truck_id: str, result: dict, ttl: int = None) -> bool:
        """Cache a truck's analysis result with TTL."""
        ttl = ttl or config.redis.truck_cache_ttl
        key = RedisKeys.truck_cache(truck_id)
        payload = json.dumps(result)
        try:
            if self._mock_mode:
                self._mock_store[key] = payload
                return True
            self._client.setex(key, ttl, payload)
            return True
        except Exception as e:
            logger.error(f"Cache set failed: {e}")
            return False

    def get_truck_result(self, truck_id: str) -> Optional[dict]:
        """Retrieve cached truck result."""
        key = RedisKeys.truck_cache(truck_id)
        try:
            if self._mock_mode:
                raw = self._mock_store.get(key)
            else:
                raw = self._client.get(key)
            return json.loads(raw) if raw else None
        except Exception as e:
            logger.error(f"Cache get failed: {e}")
            return None

    def has_truck_result(self, truck_id: str) -> bool:
        """Check if truck has cached result (O(1) key check)."""
        key = RedisKeys.truck_cache(truck_id)
        try:
            if self._mock_mode:
                return key in self._mock_store
            return bool(self._client.exists(key))
        except:
            return False

    # ─── Perceptual Hash Deduplication ───────────────────────────────────────

    def add_processed_hash(self, phash: str) -> bool:
        """Mark a perceptual hash as processed."""
        try:
            if self._mock_mode:
                self._mock_sets.setdefault(RedisKeys.PROCESSED_HASHES, set()).add(phash)
                return True
            self._client.sadd(RedisKeys.PROCESSED_HASHES, phash)
            # Set TTL on the set if not already set
            if self._client.ttl(RedisKeys.PROCESSED_HASHES) < 0:
                self._client.expire(RedisKeys.PROCESSED_HASHES, 3600)  # 1hr rolling window
            return True
        except Exception as e:
            logger.error(f"Hash add failed: {e}")
            return False

    def get_all_processed_hashes(self) -> set:
        """Return all processed hashes for similarity check."""
        try:
            if self._mock_mode:
                return self._mock_sets.get(RedisKeys.PROCESSED_HASHES, set())
            members = self._client.smembers(RedisKeys.PROCESSED_HASHES)
            return set(members) if members else set()
        except:
            return set()

    def is_hash_processed(self, phash: str) -> bool:
        """Exact hash membership check (O(1))."""
        try:
            if self._mock_mode:
                return phash in self._mock_sets.get(RedisKeys.PROCESSED_HASHES, set())
            return bool(self._client.sismember(RedisKeys.PROCESSED_HASHES, phash))
        except:
            return False

    # ─── Gemini Rate Limiter (Token Bucket) ──────────────────────────────────

    def check_and_consume_rate_limit(self) -> bool:
        """
        Token bucket rate limiter for Gemini calls.
        Allows up to N calls per minute.
        Returns True if call is allowed, False if rate limited.
        """
        key = RedisKeys.RATE_LIMITER
        limit = config.gemini.rate_limit_per_minute
        window = 60  # seconds

        try:
            if self._mock_mode:
                # Simplified mock: just allow all calls
                return True

            pipe = self._client.pipeline()
            now = time.time()
            window_start = now - window

            # Remove old entries
            pipe.zremrangebyscore(key, 0, window_start)
            # Count current window
            pipe.zcard(key)
            # Add current timestamp
            pipe.zadd(key, {str(now): now})
            pipe.expire(key, window)

            results = pipe.execute()
            current_count = results[1]

            if current_count >= limit:
                logger.warning(f"🚦 Gemini rate limit hit ({current_count}/{limit}/min)")
                return False
            return True

        except Exception as e:
            logger.error(f"Rate limiter error: {e}")
            return True  # Fail open — don't block processing

    # ─── System Stats ─────────────────────────────────────────────────────────

    def increment_stat(self, stat_name: str, amount: int = 1) -> int:
        """Atomically increment a named statistic."""
        key = f"{RedisKeys.SYSTEM_STATS}:{stat_name}"
        try:
            if self._mock_mode:
                val = self._mock_store.get(key, 0) + amount
                self._mock_store[key] = val
                return val
            return int(self._client.incrby(key, amount))
        except:
            return 0

    def get_stat(self, stat_name: str) -> int:
        """Get a named statistic value."""
        key = f"{RedisKeys.SYSTEM_STATS}:{stat_name}"
        try:
            if self._mock_mode:
                return int(self._mock_store.get(key, 0))
            val = self._client.get(key)
            return int(val) if val else 0
        except:
            return 0

    def get_all_stats(self) -> dict:
        """Retrieve all system statistics."""
        stat_names = [
            "frames_total", "frames_sampled", "detections_total",
            "quality_passed", "hash_deduped", "tracker_deduped",
            "gemini_calls", "gemini_skipped", "high_risk_count",
            "medium_risk_count", "low_risk_count",
        ]
        return {name: self.get_stat(name) for name in stat_names}

    def reset_stats(self):
        """Reset all statistics (useful for new video sessions)."""
        try:
            if self._mock_mode:
                keys_to_del = [k for k in self._mock_store if k.startswith(RedisKeys.SYSTEM_STATS)]
                for k in keys_to_del:
                    del self._mock_store[k]
            else:
                pattern = f"{RedisKeys.SYSTEM_STATS}:*"
                keys = self._client.keys(pattern)
                if keys:
                    self._client.delete(*keys)
        except Exception as e:
            logger.error(f"Stats reset failed: {e}")

    # ─── Alert Queue ──────────────────────────────────────────────────────────

    def push_alert(self, alert: dict) -> bool:
        """Push an inspection alert to the toll plaza alert queue."""
        try:
            payload = json.dumps(alert)
            if self._mock_mode:
                self._mock_store.setdefault(RedisKeys.ALERT_QUEUE, []).append(payload)
                return True
            self._client.rpush(RedisKeys.ALERT_QUEUE, payload)
            self._client.ltrim(RedisKeys.ALERT_QUEUE, -100, -1)  # Keep last 100 alerts
            return True
        except Exception as e:
            logger.error(f"Alert push failed: {e}")
            return False

    def get_recent_alerts(self, count: int = 20) -> List[dict]:
        """Get most recent N alerts from queue."""
        try:
            if self._mock_mode:
                raw_list = self._mock_store.get(RedisKeys.ALERT_QUEUE, [])
                return [json.loads(x) for x in raw_list[-count:]]
            items = self._client.lrange(RedisKeys.ALERT_QUEUE, -count, -1)
            return [json.loads(x) for x in items]
        except Exception as e:
            logger.error(f"Alert fetch failed: {e}")
            return []

    def flush_session(self):
        """Clear all session data (hashes, queue) for a fresh run."""
        try:
            keys = [
                RedisKeys.FRAMES_QUEUE,
                RedisKeys.PROCESSED_HASHES,
                RedisKeys.ALERT_QUEUE,
            ]
            if self._mock_mode:
                for k in keys:
                    self._mock_store.pop(k, None)
                    self._mock_sets.pop(k, None)
            else:
                self._client.delete(*keys)
            self.reset_stats()
            logger.info("🔄 Session data flushed")
        except Exception as e:
            logger.error(f"Flush failed: {e}")


# Singleton
redis_client = FreightRedisClient()
