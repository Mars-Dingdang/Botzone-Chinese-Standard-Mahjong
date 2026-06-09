import unittest

from mahjong_agent.botzone.protocol import ProtocolState, action_to_text
from mahjong_agent.engine.actions import Action


class BotzoneTest(unittest.TestCase):
    def test_replay_and_action_format(self):
        state = ProtocolState()
        state.apply("0 1 0")
        state.apply("1 0 0 0 0 W1 W2 W3 W4 W5 W6 W7 W8 W9 T1 T2 T3 T4")
        state.apply("2 T5")
        self.assertEqual(state.player_id, 1)
        self.assertEqual(len(state.hand), 14)
        self.assertEqual(action_to_text(Action.play(0)), "PLAY W1")


if __name__ == "__main__":
    unittest.main()
