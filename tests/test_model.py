import unittest
import os
import tempfile


class ModelTest(unittest.TestCase):
    def test_forward(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch not installed")
        from mahjong_agent.features import FEATURE_SIZE
        from mahjong_agent.models.hybrid_transformer import HybridTransformer
        model = HybridTransformer(d_model=48, layers=1, heads=6)
        output = model(
            torch.zeros(2, FEATURE_SIZE),
            torch.zeros(2, 5, 8),
            torch.ones(2, 5, dtype=torch.bool),
        )
        self.assertEqual(tuple(output["logits"].shape), (2, 5))
        self.assertEqual(tuple(output["value"].shape), (2,))

    def test_v2_forward_and_versioned_checkpoint(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch not installed")
        from mahjong_agent.models import create_model
        from mahjong_agent.training.checkpoint import (load_model_from_checkpoint,
                                                       save_checkpoint)
        model = create_model(2, d_model=48, layers=1, heads=6)
        output = model(
            torch.zeros(2, 16, 12), torch.zeros(2, 5, 4, 12),
            torch.ones(2, 5, dtype=torch.bool),
            torch.ones(2, 16, dtype=torch.bool),
            torch.ones(2, 5, 4, dtype=torch.bool))
        self.assertEqual(tuple(output["belief_logits"].shape), (2, 3, 34, 5))
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.pt")
            save_checkpoint(path, model, metadata={"belief_mode": "aux"})
            loaded, metadata = load_model_from_checkpoint(path)
            self.assertEqual(metadata["feature_version"], 2)
            self.assertEqual(loaded.feature_version, 2)


if __name__ == "__main__":
    unittest.main()
