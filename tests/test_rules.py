import unittest

from mahjong_agent.rules import RulesBackend


class RulesTest(unittest.TestCase):
    def test_standard_hand(self):
        counts = [0] * 34
        for tile in (0, 1, 2, 3, 4, 5, 9, 9, 9, 27, 27, 27, 31, 31):
            counts[tile] += 1
        rules = RulesBackend()
        self.assertTrue(rules.is_complete_hand(counts))
        self.assertGreaterEqual(rules.fan(counts), 8)

    def test_seven_pairs(self):
        counts = [0] * 34
        for tile in range(7):
            counts[tile] = 2
        self.assertTrue(RulesBackend().is_complete_hand(counts))


if __name__ == "__main__":
    unittest.main()
