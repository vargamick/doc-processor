import unittest
from unittest.mock import MagicMock, patch
import redis
import json
import zlib
from .cache import EmbeddingCache, CacheConfig, RedisError


class TestEmbeddingCache(unittest.TestCase):
    def setUp(self):
        self.config = CacheConfig(
            host="localhost",
            port=6379,
            db=0,
            pool_size=2,
            compression_threshold=10,  # Small threshold for testing
            max_item_size=1024,
            default_ttl=60,
        )
        self.sample_embedding = [0.1, 0.2, 0.3]
        self.sample_key = "test_key"

    @patch("redis.Redis")
    def test_get_hit(self, mock_redis):
        # Setup mock
        mock_conn = MagicMock()
        mock_redis.return_value = mock_conn
        mock_conn.get.return_value = json.dumps(self.sample_embedding)

        cache = EmbeddingCache(self.config)
        result = cache.get(self.sample_key)

        self.assertEqual(result, self.sample_embedding)
        mock_conn.get.assert_called_once_with(f"emb:{self.sample_key}")

    @patch("redis.Redis")
    def test_get_miss(self, mock_redis):
        # Setup mock
        mock_conn = MagicMock()
        mock_redis.return_value = mock_conn
        mock_conn.get.return_value = None

        cache = EmbeddingCache(self.config)
        result = cache.get(self.sample_key)

        self.assertIsNone(result)
        mock_conn.get.assert_called_once_with(f"emb:{self.sample_key}")

    @patch("redis.Redis")
    def test_set_success(self, mock_redis):
        # Setup mock
        mock_conn = MagicMock()
        mock_redis.return_value = mock_conn
        mock_conn.setex.return_value = True

        cache = EmbeddingCache(self.config)
        result = cache.set(self.sample_key, self.sample_embedding)

        self.assertTrue(result)
        mock_conn.setex.assert_called_once()

    @patch("redis.Redis")
    def test_set_with_compression(self, mock_redis):
        # Setup mock
        mock_conn = MagicMock()
        mock_redis.return_value = mock_conn
        mock_conn.setex.return_value = True

        # Create large embedding to trigger compression
        large_embedding = [0.1] * 100

        cache = EmbeddingCache(self.config)
        result = cache.set(self.sample_key, large_embedding)

        self.assertTrue(result)
        mock_conn.setex.assert_called_once()

    @patch("redis.Redis")
    def test_delete(self, mock_redis):
        # Setup mock
        mock_conn = MagicMock()
        mock_redis.return_value = mock_conn
        mock_conn.delete.return_value = 1

        cache = EmbeddingCache(self.config)
        result = cache.delete(self.sample_key)

        self.assertTrue(result)
        mock_conn.delete.assert_called_once_with(f"emb:{self.sample_key}")

    @patch("redis.Redis")
    def test_clear(self, mock_redis):
        # Setup mock
        mock_conn = MagicMock()
        mock_redis.return_value = mock_conn
        mock_conn.keys.return_value = [b"emb:key1", b"emb:key2"]
        mock_conn.delete.return_value = 2

        cache = EmbeddingCache(self.config)
        result = cache.clear()

        self.assertTrue(result)
        mock_conn.keys.assert_called_once_with("emb:*")
        mock_conn.delete.assert_called_once()

    @patch("redis.Redis")
    def test_circuit_breaker(self, mock_redis):
        # Setup mock to raise error
        mock_conn = MagicMock()
        mock_redis.return_value = mock_conn
        mock_conn.get.side_effect = RedisError("Connection failed")

        cache = EmbeddingCache(self.config)

        # Trigger circuit breaker
        for _ in range(self.config.circuit_breaker_threshold):
            result = cache.get(self.sample_key)
            self.assertIsNone(result)

        # Verify circuit is open
        result = cache.get(self.sample_key)
        self.assertIsNone(result)
        self.assertTrue(cache.circuit_breaker.is_open)

    @patch("redis.Redis")
    def test_compression(self, mock_redis):
        # Setup mock
        mock_conn = MagicMock()
        mock_redis.return_value = mock_conn

        # Create data that will be compressed
        large_data = json.dumps([0.1] * 1000)
        compressed = zlib.compress(large_data.encode("utf-8"))
        mock_conn.get.return_value = compressed

        cache = EmbeddingCache(self.config)
        result = cache.get(self.sample_key)

        # Explicit type check and assertion
        if result is None:
            self.fail("Expected non-None result")

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1000)

    @patch("redis.Redis")
    def test_metrics(self, mock_redis):
        # Setup mock
        mock_conn = MagicMock()
        mock_redis.return_value = mock_conn
        mock_conn.get.side_effect = [
            json.dumps(self.sample_embedding),  # Hit
            None,  # Miss
            RedisError("Failed"),  # Error
        ]

        cache = EmbeddingCache(self.config)

        # Generate some metrics
        cache.get(self.sample_key)  # Hit
        cache.get(self.sample_key)  # Miss
        cache.get(self.sample_key)  # Error

        metrics = cache.get_metrics()
        self.assertEqual(metrics["hits"], 1)
        self.assertEqual(metrics["misses"], 1)
        self.assertEqual(metrics["errors"], 1)

    @patch("redis.Redis")
    def test_health_check(self, mock_redis):
        # Setup mock
        mock_conn = MagicMock()
        mock_redis.return_value = mock_conn
        mock_conn.ping.return_value = True

        cache = EmbeddingCache(self.config)
        health = cache.health_check()

        self.assertEqual(health["status"], "healthy")
        self.assertEqual(health["circuit_breaker"], "closed")
        self.assertIn("metrics", health)

    @patch("redis.Redis")
    def test_connection_pool_reuse(self, mock_redis):
        # Create multiple cache instances with same config
        cache1 = EmbeddingCache(self.config)
        cache2 = EmbeddingCache(self.config)

        # Verify connection pool is reused
        self.assertEqual(id(cache1.pool), id(cache2.pool))


if __name__ == "__main__":
    unittest.main()
