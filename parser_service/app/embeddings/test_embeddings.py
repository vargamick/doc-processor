import unittest
from unittest.mock import MagicMock, patch
import torch
import numpy as np
import json
from .base import EmbeddingProcessor, EmbeddingConfig


class TestEmbeddingProcessor(unittest.TestCase):
    def setUp(self):
        self.config = EmbeddingConfig(
            model_name="all-MiniLM-L6-v2",
            batch_size=2,
            cache_enabled=True,
            device="cpu",  # Use CPU for testing
        )
        # Mock embeddings in different formats
        self.mock_tensor = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
        self.mock_array = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        self.mock_tensor_list = [
            torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
            torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32),
        ]
        self.mock_batch_tensor = torch.tensor(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float32
        )
        self.mock_batch_array = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32
        )

    @patch("sentence_transformers.SentenceTransformer")
    def test_single_embedding_tensor(self, mock_transformer):
        """Test embedding generation with tensor output"""
        mock_model = MagicMock()
        mock_model.encode.return_value = self.mock_tensor
        mock_transformer.return_value = mock_model

        processor = EmbeddingProcessor(self.config)
        result = processor.generate_embedding("Test text")

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[0], 0.1)
        self.assertAlmostEqual(result[1], 0.2)
        self.assertAlmostEqual(result[2], 0.3)

    @patch("sentence_transformers.SentenceTransformer")
    def test_single_embedding_array(self, mock_transformer):
        """Test embedding generation with numpy array output"""
        mock_model = MagicMock()
        mock_model.encode.return_value = self.mock_array
        mock_transformer.return_value = mock_model

        processor = EmbeddingProcessor(self.config)
        result = processor.generate_embedding("Test text")

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[0], 0.1)
        self.assertAlmostEqual(result[1], 0.2)
        self.assertAlmostEqual(result[2], 0.3)

    @patch("sentence_transformers.SentenceTransformer")
    def test_single_embedding_tensor_list(self, mock_transformer):
        """Test embedding generation with list of tensors output"""
        mock_model = MagicMock()
        mock_model.encode.return_value = self.mock_tensor_list
        mock_transformer.return_value = mock_model

        processor = EmbeddingProcessor(self.config)
        result = processor.generate_embedding("Test text")

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 6)  # Concatenated tensors
        self.assertAlmostEqual(result[0], 0.1)
        self.assertAlmostEqual(result[3], 0.4)

    @patch("sentence_transformers.SentenceTransformer")
    def test_batch_embedding_tensor(self, mock_transformer):
        """Test batch embedding generation with tensor output"""
        mock_model = MagicMock()
        mock_model.encode.return_value = self.mock_batch_tensor
        mock_transformer.return_value = mock_model

        processor = EmbeddingProcessor(self.config)
        results = processor.generate_embeddings_batch(["Text 1", "Text 2"])

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)
        self.assertEqual(len(results[0]), 3)
        self.assertEqual(len(results[1]), 3)
        self.assertAlmostEqual(results[0][0], 0.1)
        self.assertAlmostEqual(results[1][0], 0.4)

    @patch("sentence_transformers.SentenceTransformer")
    def test_device_compatibility(self, mock_transformer):
        """Test tensor device compatibility handling"""
        mock_model = MagicMock()
        # Simulate tensor on different device
        cuda_tensor = MagicMock(spec=torch.Tensor)
        cuda_tensor.device.type = "cuda"
        cuda_tensor.to.return_value = self.mock_tensor
        mock_model.encode.return_value = cuda_tensor
        mock_transformer.return_value = mock_model

        processor = EmbeddingProcessor(self.config)
        result = processor.generate_embedding("Test text")

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
        cuda_tensor.to.assert_called_once_with("cpu")

    @patch("sentence_transformers.SentenceTransformer")
    @patch("redis.Redis")
    def test_caching_with_tensor_output(self, mock_redis, mock_transformer):
        """Test caching behavior with tensor output"""
        # Setup mocks
        mock_model = MagicMock()
        mock_model.encode.return_value = self.mock_tensor
        mock_transformer.return_value = mock_model

        mock_redis_instance = MagicMock()
        mock_redis.return_value = mock_redis_instance
        mock_redis_instance.get.return_value = None  # Cache miss initially

        processor = EmbeddingProcessor(self.config)

        # First call - should generate embedding
        result1 = processor.generate_embedding("Test text")

        # Verify embedding was generated and cached
        mock_model.encode.assert_called_once()
        self.assertTrue(mock_redis_instance.setex.called)

        # Setup cache hit for second call
        expected_list = [0.1, 0.2, 0.3]
        mock_redis_instance.get.return_value = json.dumps(expected_list)

        # Second call - should use cache
        result2 = processor.generate_embedding("Test text")

        # Verify model was not called again
        mock_model.encode.assert_called_once()

        # Verify results match
        self.assertEqual(result1, result2)
        self.assertEqual(result2, expected_list)

    def test_invalid_output_handling(self):
        """Test handling of invalid model outputs"""
        mock_model = MagicMock()
        mock_model.encode.return_value = "invalid"  # Invalid output type

        processor = EmbeddingProcessor(self.config)
        processor.model = mock_model

        with self.assertRaises(TypeError):
            processor.generate_embedding("Test text")

    def test_model_info(self):
        """Test model information retrieval"""
        with patch("sentence_transformers.SentenceTransformer") as mock_transformer:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_transformer.return_value = mock_model

            processor = EmbeddingProcessor(self.config)
            info = processor.get_model_info()

            self.assertEqual(info["model_name"], "all-MiniLM-L6-v2")
            self.assertEqual(info["batch_size"], 2)
            self.assertTrue(info["cache_enabled"])
            self.assertEqual(info["device"], "cpu")
            self.assertEqual(info["embedding_dimension"], 384)


if __name__ == "__main__":
    unittest.main()
