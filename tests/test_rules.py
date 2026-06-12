import unittest
from unittest import mock

from mahjong_agent.engine.actions import ActionType, Meld
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

    def test_strict_hu_rejects_official_calculator_error(self):
        rules = RulesBackend()
        rules.has_official = True
        rules.official_fan_calculator = mock.Mock(side_effect=RuntimeError("bad context"))
        self.assertFalse(rules.strict_can_hu(
            [0] * 34, win_tile=0,
            context={"player_id": 0, "seat_wind": 0, "prevalent_wind": 0}))


    def test_official_can_hu_fails_closed_on_calculator_error(self):
        rules = RulesBackend()
        rules.has_official = True
        rules.official_fan_calculator = mock.Mock(side_effect=RuntimeError("bad context"))
        self.assertFalse(rules.can_hu(
            [0] * 34, win_tile=0,
            context={"player_id": 0, "seat_wind": 0, "prevalent_wind": 0}))

    def test_official_fan_uses_multiplicity(self):
        rules = RulesBackend()
        rules.has_official = True
        rules.official_fan_calculator = mock.Mock(return_value=((1, 2, "x", "x"),))
        counts = [0] * 34
        counts[0] = 1
        self.assertEqual(rules.fan(counts, win_tile=0, context={}), 2)


    def test_strict_hu_rejects_if_any_chi_offer_is_below_minimum(self):
        rules = RulesBackend()
        rules.has_official = True

        def calculate(pack, *args, **kwargs):
            fan = 7 if pack[0][2] == 2 else 8
            return ((fan, 1, "x", "x"),)

        rules.official_fan_calculator = calculate
        counts = [0] * 34
        counts[0] = 1
        melds = [Meld(ActionType.CHI, (0, 1, 2), 3)]
        self.assertFalse(rules.strict_can_hu(
            counts, melds, win_tile=0,
            context={"player_id": 0, "seat_wind": 0, "prevalent_wind": 0}))


if __name__ == "__main__":
    unittest.main()
