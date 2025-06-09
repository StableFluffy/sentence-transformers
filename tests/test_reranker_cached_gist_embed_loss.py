from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader


class TestRerankerCachedGISTEmbedLoss(unittest.TestCase):
    def setUp(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.reranker_url = "http://localhost:8000"
        self.loss = losses.RerankerCachedGISTEmbedLoss(
            model=self.model,
            reranker_url=self.reranker_url,
            mini_batch_size=4,
            reranker_batch_size=8,
            margin_strategy="absolute",
            margin=0.1,
        )

    @patch("aiohttp.ClientSession.post")
    def test_reranker_call(self, mock_post):
        """Test that the reranker API is called correctly."""
        # Mock the batch API response
        mock_response = AsyncMock()
        mock_response.status = 200
        # Response format for batch API: list of responses
        mock_response.json = AsyncMock(return_value=[
            {"scores": [0.9, 0.1, 0.2, 0.8]}
        ])
        mock_post.return_value.__aenter__.return_value = mock_response

        # Test data
        queries = ["What is AI?", "How to cook?"]
        documents = ["Artificial Intelligence is...", "To cook pasta..."]

        # Call the reranker
        scores = self.loss.call_reranker_batch(queries, documents)

        # Check the results
        expected_shape = (2, 2)  # 2 queries x 2 documents
        self.assertEqual(scores.shape, expected_shape)
        self.assertTrue(np.allclose(scores.flatten(), [0.9, 0.1, 0.2, 0.8]))

    @patch.object(losses.RerankerCachedGISTEmbedLoss, "call_reranker_batch")
    def test_forward_pass(self, mock_reranker):
        """Test the forward pass of the loss function."""
        # Mock reranker scores
        mock_reranker.return_value = np.array([[0.9, 0.1], [0.1, 0.9]])

        # Create dummy data
        sentence_features = [
            {
                "input_ids": torch.randint(0, 1000, (2, 128)),
                "attention_mask": torch.ones(2, 128),
            },
            {
                "input_ids": torch.randint(0, 1000, (2, 128)),
                "attention_mask": torch.ones(2, 128),
            },
        ]
        labels = torch.tensor([0, 1])

        # Forward pass
        with torch.no_grad():
            loss = self.loss(sentence_features, labels)

        # Check that loss is computed
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # Scalar loss
        self.assertGreater(loss.item(), 0)  # Loss should be positive

    def test_margin_strategies(self):
        """Test different margin strategies for false negative filtering."""
        # Test absolute margin
        loss_abs = losses.RerankerCachedGISTEmbedLoss(
            model=self.model,
            reranker_url=self.reranker_url,
            margin_strategy="absolute",
            margin=0.1,
        )
        self.assertEqual(loss_abs.margin_strategy, "absolute")

        # Test relative margin
        loss_rel = losses.RerankerCachedGISTEmbedLoss(
            model=self.model,
            reranker_url=self.reranker_url,
            margin_strategy="relative",
            margin=0.05,
        )
        self.assertEqual(loss_rel.margin_strategy, "relative")

        # Test invalid margin strategy
        with self.assertRaises(ValueError):
            losses.RerankerCachedGISTEmbedLoss(
                model=self.model,
                reranker_url=self.reranker_url,
                margin_strategy="invalid",
            )

    def test_config_dict(self):
        """Test that get_config_dict returns correct configuration."""
        config = self.loss.get_config_dict()
        
        expected_keys = {
            "reranker_url",
            "temperature",
            "mini_batch_size",
            "margin_strategy",
            "margin",
            "reranker_batch_size",
            "instruction",
            "max_length",
            "timeout",
        }
        
        self.assertEqual(set(config.keys()), expected_keys)
        self.assertEqual(config["reranker_url"], self.reranker_url)
        self.assertEqual(config["margin_strategy"], "absolute")
        self.assertEqual(config["margin"], 0.1)

    @patch("aiohttp.ClientSession.post")
    def test_api_error_handling(self, mock_post):
        """Test error handling when API calls fail."""
        # Mock API error
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_post.return_value.__aenter__.return_value = mock_response

        # Test data
        queries = ["What is AI?"]
        documents = ["Artificial Intelligence is..."]

        # Should raise RuntimeError
        with self.assertRaises(RuntimeError) as context:
            self.loss.call_reranker_batch(queries, documents)
        
        self.assertIn("Reranker API error", str(context.exception))

    def test_batch_processing(self):
        """Test that large batches are processed correctly."""
        # Create a loss with small reranker batch size
        loss = losses.RerankerCachedGISTEmbedLoss(
            model=self.model,
            reranker_url=self.reranker_url,
            mini_batch_size=2,
            reranker_batch_size=4,  # Small batch for testing
        )

        with patch.object(loss, "_call_reranker_async") as mock_async:
            # Mock async calls to return sequential scores
            async def mock_return(session, pairs):
                return [0.5 + 0.1 * i for i in range(len(pairs))]
            
            mock_async.side_effect = mock_return

            # Test with more pairs than batch size
            queries = ["Q1", "Q2", "Q3"]
            documents = ["D1", "D2", "D3", "D4"]
            
            scores = loss.call_reranker_batch(queries, documents)
            
            # Check that multiple batches were called
            self.assertEqual(mock_async.call_count, 3)  # 12 pairs / 4 batch_size = 3 calls
            self.assertEqual(scores.shape, (3, 4))


if __name__ == "__main__":
    unittest.main()