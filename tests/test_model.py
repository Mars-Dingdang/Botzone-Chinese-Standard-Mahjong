import unittest


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


if __name__ == "__main__":
    unittest.main()
