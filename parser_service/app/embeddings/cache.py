from typing import Optional, Dict, Any, List, Union
import redis
from redis.connection import ConnectionPool
from redis.retry import Retry
from redis.backoff import ExponentialBackoff
from redis.exceptions import RedisError
import zlib
import json
import time
import logging
from dataclasses import dataclass
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for Redis cache"""

    host: str = "redis"
    port: int = 6379
    db: int = 0
    pool_size: int = 10
    socket_timeout: int = 5
    retry_attempts: int = 3
    circuit_breaker_threshold: int = 5
    compression_threshold: int = 1024  # 1KB
    max_item_size: int = 1024 * 1024  # 1MB
    default_ttl: int = 3600  # 1 hour
    namespace: str = "emb"


class CircuitBreaker:
    """Circuit breaker pattern implementation"""

    def __init__(self, threshold: int = 5, reset_time: int = 60):
        self.threshold = threshold
        self.reset_time = reset_time
        self.failures = 0
        self.last_failure_time = 0
        self.is_open = False

    def record_failure(self) -> None:
        current_time = time.time()
        if current_time - self.last_failure_time > self.reset_time:
            self.failures = 0

        self.failures += 1
        self.last_failure_time = current_time

        if self.failures >= self.threshold:
            self.is_open = True

    def record_success(self) -> None:
        self.failures = 0
        self.is_open = False

    def can_proceed(self) -> bool:
        if not self.is_open:
            return True

        if time.time() - self.last_failure_time > self.reset_time:
            self.is_open = False
            return True

        return False


class CacheMetrics:
    """Track cache performance metrics"""

    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.errors = 0
        self.total_latency = 0
        self.total_operations = 0

    def record_hit(self, latency: float) -> None:
        self.hits += 1
        self._record_operation(latency)

    def record_miss(self, latency: float) -> None:
        self.misses += 1
        self._record_operation(latency)

    def record_error(self) -> None:
        self.errors += 1

    def _record_operation(self, latency: float) -> None:
        self.total_operations += 1
        self.total_latency += latency

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0

    @property
    def average_latency(self) -> float:
        return (
            self.total_latency / self.total_operations
            if self.total_operations > 0
            else 0
        )

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "errors": self.errors,
            "hit_rate": self.hit_rate,
            "average_latency": self.average_latency,
            "total_operations": self.total_operations,
        }


class EmbeddingCache:
    """Enhanced Redis cache implementation for embeddings"""

    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.metrics = CacheMetrics()
        self.circuit_breaker = CircuitBreaker(
            threshold=self.config.circuit_breaker_threshold
        )

        # Configure retry strategy
        retry = Retry(ExponentialBackoff(cap=10, base=1), self.config.retry_attempts)

        # Initialize connection pool
        self.pool = ConnectionPool(
            host=self.config.host,
            port=self.config.port,
            db=self.config.db,
            max_connections=self.config.pool_size,
            socket_timeout=self.config.socket_timeout,
            retry=retry,
            decode_responses=True,
        )

    def _get_key(self, key: str) -> str:
        """Generate namespaced cache key"""
        return f"{self.config.namespace}:{key}"

    def _compress(self, data: str) -> bytes:
        """Compress data if it exceeds threshold"""
        data_bytes = data.encode("utf-8")
        if len(data_bytes) > self.config.compression_threshold:
            return zlib.compress(data_bytes)
        return data_bytes

    def _decompress(self, data: bytes) -> str:
        """Decompress data if it's compressed"""
        try:
            return zlib.decompress(data).decode("utf-8")
        except zlib.error:
            return data.decode("utf-8")

    @contextmanager
    def _get_connection(self):
        """Get Redis connection from pool with automatic cleanup"""
        if not self.circuit_breaker.can_proceed():
            raise RedisError("Circuit breaker is open")

        connection = redis.Redis(connection_pool=self.pool)
        try:
            yield connection
        except RedisError as e:
            self.circuit_breaker.record_failure()
            self.metrics.record_error()
            logger.error(f"Redis operation failed: {str(e)}")
            raise
        finally:
            connection.close()

    def get(self, key: str) -> Optional[List[float]]:
        """Get embedding from cache"""
        start_time = time.time()
        try:
            with self._get_connection() as conn:
                cached_data = conn.get(self._get_key(key))

                if cached_data is None:
                    self.metrics.record_miss(time.time() - start_time)
                    return None

                # Handle different response types
                if isinstance(cached_data, bytes):
                    cached_data = self._decompress(cached_data)
                elif isinstance(cached_data, str):
                    pass  # String can be used directly
                else:
                    logger.error(f"Unexpected cache data type: {type(cached_data)}")
                    return None

                try:
                    self.circuit_breaker.record_success()
                    self.metrics.record_hit(time.time() - start_time)
                    return json.loads(cached_data)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.error(f"Failed to decode cached data: {str(e)}")
                    return None

        except RedisError as e:
            logger.error(f"Failed to get embedding from cache: {str(e)}")
            return None

    def set(self, key: str, embedding: List[float], ttl: Optional[int] = None) -> bool:
        """Set embedding in cache"""
        if not embedding:
            return False

        start_time = time.time()
        try:
            data = json.dumps(embedding)
            if len(data) > self.config.max_item_size:
                logger.warning(f"Embedding size exceeds maximum: {len(data)} bytes")
                return False

            compressed_data = self._compress(data)

            with self._get_connection() as conn:
                conn.setex(
                    self._get_key(key), ttl or self.config.default_ttl, compressed_data
                )

            self.circuit_breaker.record_success()
            self.metrics.record_hit(time.time() - start_time)
            return True

        except RedisError as e:
            logger.error(f"Failed to set embedding in cache: {str(e)}")
            return False

    def delete(self, key: str) -> bool:
        """Delete embedding from cache"""
        try:
            with self._get_connection() as conn:
                return bool(conn.delete(self._get_key(key)))
        except RedisError as e:
            logger.error(f"Failed to delete embedding from cache: {str(e)}")
            return False

    def get_metrics(self) -> Dict[str, Any]:
        """Get cache performance metrics"""
        return self.metrics.get_metrics()

    def clear(self) -> bool:
        """Clear all cached embeddings"""
        try:
            with self._get_connection() as conn:
                pattern = f"{self.config.namespace}:*"
                keys = conn.keys(pattern)

                # Handle empty results
                if not keys:
                    return True

                # Ensure keys is a list and handle response types
                if isinstance(keys, (list, tuple)):
                    key_list = []
                    for k in keys:
                        if isinstance(k, bytes):
                            key_list.append(k.decode("utf-8"))
                        elif isinstance(k, str):
                            key_list.append(k)
                        else:
                            logger.warning(f"Unexpected key type: {type(k)}")
                            continue

                    if key_list:
                        conn.delete(*key_list)
                return True
        except RedisError as e:
            logger.error(f"Failed to clear cache: {str(e)}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """Check cache health status"""
        try:
            with self._get_connection() as conn:
                conn.ping()
                return {
                    "status": "healthy",
                    "circuit_breaker": (
                        "closed" if self.circuit_breaker.can_proceed() else "open"
                    ),
                    "metrics": self.get_metrics(),
                }
        except RedisError as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "circuit_breaker": "open",
                "metrics": self.get_metrics(),
            }
