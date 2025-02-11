import unittest
import os
from unittest.mock import patch
import numpy as np
from .base import EmbeddingProcessor, EmbeddingConfig
from .cache import EmbeddingCache, CacheConfig


class TestEmbeddingIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up environment variables for testing
        os.environ["REDIS_HOST"] = "localhost"
        os.environ["REDIS_PORT"] = "6379"
        os.environ["REDIS_DB"] = "0"
        os.environ["REDIS_POOL_SIZE"] = "2"
        os.environ["REDIS_TIMEOUT"] = "5"
        os.environ["REDIS_COMPRESSION_THRESHOLD"] = "10"
        os.environ["REDIS_MAX_ITEM_SIZE"] = "1024"
        os.environ["REDIS_TTL"] = "60"

    def setUp(self):
        self.config = EmbeddingConfig(
            model_name="all-MiniLM-L6-v2",
            batch_size=2,
            cache_enabled=True,
            device="cpu",  # Use CPU for testing
        )
        self.processor = EmbeddingProcessor(self.config)
        self.test_texts = [
            "This is a test sentence.",
            "Another test sentence for embedding.",
            "A third sentence to test batch processing.",
        ]

    def test_end_to_end_single_embedding(self):
        """Test complete pipeline for single text embedding with caching"""
        # First embedding should generate and cache
        embedding1 = self.processor.generate_embedding(self.test_texts[0])

        self.assertIsInstance(embedding1, list)
        self.assertTrue(all(isinstance(x, float) for x in embedding1))
        self.assertEqual(
            len(embedding1), 384
        )  # Expected dimension for all-MiniLM-L6-v2

        # Second call should use cache
        embedding2 = self.processor.generate_embedding(self.test_texts[0])

        # Results should be identical when retrieved from cache
        self.assertEqual(embedding1, embedding2)

    def test_end_to_end_batch_embedding(self):
        """Test complete pipeline for batch embedding with caching"""
        # First batch should generate and cache all embeddings
        embeddings1 = self.processor.generate_embeddings_batch(self.test_texts)

        self.assertEqual(len(embeddings1), len(self.test_texts))
        for emb in embeddings1:
            self.assertIsInstance(emb, list)
            self.assertTrue(all(isinstance(x, float) for x in emb))
            self.assertEqual(len(emb), 384)  # Expected dimension

        # Second batch should use cache
        embeddings2 = self.processor.generate_embeddings_batch(self.test_texts)

        # Results should be identical when retrieved from cache
        for emb1, emb2 in zip(embeddings1, embeddings2):
            self.assertEqual(emb1, emb2)

    def test_cache_persistence(self):
        """Test that cache persists between processor instances"""
        # Generate and cache embeddings with first processor
        embeddings1 = self.processor.generate_embeddings_batch(self.test_texts)

        # Create new processor instance
        processor2 = EmbeddingProcessor(self.config)

        # Get embeddings with new processor (should use cache)
        embeddings2 = processor2.generate_embeddings_batch(self.test_texts)

        # Results should be identical
        for emb1, emb2 in zip(embeddings1, embeddings2):
            self.assertEqual(emb1, emb2)

    def test_mixed_cache_batch(self):
        """Test batch processing with mix of cached and uncached texts"""
        # Cache first text
        embedding1 = self.processor.generate_embedding(self.test_texts[0])

        # Process batch with mix of cached and uncached
        embeddings = self.processor.generate_embeddings_batch(self.test_texts)

        # First embedding should match cached version
        self.assertEqual(embeddings[0], embedding1)

        # All embeddings should have correct format
        for emb in embeddings:
            self.assertIsInstance(emb, list)
            self.assertTrue(all(isinstance(x, float) for x in emb))
            self.assertEqual(len(emb), 384)

    def test_cache_compression(self):
        """Test compression for large embeddings"""
        # Create large text to generate large embedding
        large_text = " ".join(["test"] * 1000)

        # Generate embedding
        embedding = self.processor.generate_embedding(large_text)

        # Verify embedding is cached and retrievable
        cached_embedding = self.processor.generate_embedding(large_text)
        self.assertEqual(embedding, cached_embedding)

    def test_error_handling(self):
        """Test error handling in the pipeline"""
        # Test with invalid text type
        with self.assertRaises(RuntimeError):
            self.processor.generate_embedding(None)  # type: ignore

        # Test with empty batch
        empty_result = self.processor.generate_embeddings_batch([])
        self.assertEqual(empty_result, [])

        # Test with invalid texts in batch
        with self.assertRaises(RuntimeError):
            self.processor.generate_embeddings_batch([None, "test"])  # type: ignore

    def test_model_info(self):
        """Test model information retrieval"""
        info = self.processor.get_model_info()

        self.assertEqual(info["model_name"], "all-MiniLM-L6-v2")
        self.assertEqual(info["device"], "cpu")
        self.assertTrue(info["cache_enabled"])
        self.assertEqual(info["embedding_dimension"], 384)
        self.assertEqual(info["batch_size"], 2)

    def test_cache_metrics(self):
        """Test cache metrics tracking"""
        # Get initial metrics
        initial_metrics = self.processor.cache.get_metrics()  # type: ignore

        # Generate new embedding
        self.processor.generate_embedding(self.test_texts[0])

        # Get metrics after first generation (should show miss)
        after_miss_metrics = self.processor.cache.get_metrics()  # type: ignore
        self.assertEqual(after_miss_metrics["misses"], initial_metrics["misses"] + 1)

        # Generate same embedding again
        self.processor.generate_embedding(self.test_texts[0])

        # Get metrics after cache hit
        after_hit_metrics = self.processor.cache.get_metrics()  # type: ignore
        self.assertEqual(after_hit_metrics["hits"], after_miss_metrics["hits"] + 1)

    def test_cache_health(self):
        """Test cache health check"""
        health = self.processor.cache.health_check()  # type: ignore

        self.assertEqual(health["status"], "healthy")
        self.assertEqual(health["circuit_breaker"], "closed")
        self.assertIn("metrics", health)


if __name__ == "__main__":
    unittest.main()
