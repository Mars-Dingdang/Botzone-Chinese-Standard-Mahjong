import unittest
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from unittest import mock

from mahjong_agent.botzone.legality import (hu_context, response_to_action, sanitize_action,
                                            strict_legal_actions, validate_action)
from mahjong_agent.botzone.protocol import ProtocolState, action_to_text
from mahjong_agent.engine.actions import Action, ActionType, Meld
from mahjong_agent.rules import default_backend
from scripts.audit_botzone_log import audit_events


class BotzoneTest(unittest.TestCase):
    def test_replay_and_action_format(self):
        state = ProtocolState()
        state.apply("0 1 0")
        state.apply("1 0 0 0 0 W1 W2 W3 W4 W5 W6 W7 W8 W9 T1 T2 T3 T4")
        state.apply("2 T5")
        self.assertEqual(state.player_id, 1)
        self.assertEqual(len(state.hand), 14)
        self.assertEqual(action_to_text(Action.play(0)), "PLAY W1")

    def test_initial_hand_ignores_flower_list(self):
        state = ProtocolState()
        state.apply("1 1 0 0 0 W1 W2 W3 W4 W5 W6 W7 W8 W9 T1 T2 T3 T4 H1")
        self.assertEqual(len(state.hand), 13)
        self.assertEqual(state.phase, "ack")

    def test_claim_and_bugang_phases(self):
        state = ProtocolState()
        state.apply("0 1 0")
        state.apply("1 0 0 0 0 W1 W1 W2 W3 W4 W5 W6 W7 W8 W9 T1 T2 T3")
        state.apply("3 0 PLAY W1")
        self.assertEqual(state.phase, "claim")
        state.apply("3 0 BUGANG W1")
        self.assertTrue(state.claim_hu_only)
        self.assertEqual(state.last_discard, (0, 0))

    def test_flower_replacement_is_acknowledged(self):
        state = ProtocolState()
        state.apply("3 2 BUHUA H1")
        self.assertEqual(state.flower_counts[2], 1)
        self.assertEqual(state.phase, "ack")

    def test_own_concealed_gang_uses_previous_response(self):
        state = ProtocolState()
        state.apply("0 1 0")
        state.apply("1 0 0 0 0 W1 W1 W1 W1 W2 W3 W4 W5 W6 W7 W8 W9 T1")
        state.apply("3 1 GANG", "GANG W1")
        self.assertEqual(state.melds[1][0].kind, ActionType.GANG)
        self.assertNotIn(0, state.hand)

    def test_cannot_claim_own_discard_regression(self):
        state = ProtocolState()
        state.apply("0 2 1")
        state.apply("1 0 0 0 0 T6 T6 T6 B2 B2 B2 B5 B5 B5 F4 F4 F4 F4")
        state.apply("2 F4")
        state.apply("3 2 PLAY F4", "PLAY F4")
        self.assertEqual([action_to_text(action, state)
                          for action in strict_legal_actions(state)], ["PASS"])
        proposed = response_to_action(state, "PENG B5")
        self.assertEqual(validate_action(state, proposed),
                         (False, "cannot claim own discard"))
        self.assertEqual(sanitize_action(state, proposed)[0], Action.pass_())

    def test_own_peng_discard_cannot_be_claimed_again(self):
        state = ProtocolState()
        state.apply("0 1 0")
        state.apply("1 0 0 0 0 W1 W1 W2 W3 W4 W5 W6 W7 W8 W9 T1 T2 T3")
        state.apply("3 0 PLAY W1")
        state.apply("3 1 PENG W2", "PENG W2")
        self.assertEqual([action_to_text(action, state)
                          for action in strict_legal_actions(state)], ["PASS"])

    def test_gang_wire_format_depends_on_phase(self):
        state = ProtocolState()
        state.phase = "claim"
        self.assertEqual(action_to_text(Action(ActionType.GANG, 0), state), "GANG")
        state.phase = "discard"
        self.assertEqual(action_to_text(Action(ActionType.GANG, 0), state), "GANG W1")

    def test_wall_last_forbids_claims_and_kongs(self):
        state = ProtocolState()
        state.player_id = 1
        state.phase = "claim"
        state.last_discard = (0, 0)
        state.hand = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        state.wall_last = True
        self.assertEqual([action.kind for action in strict_legal_actions(state)],
                         [ActionType.PASS])
        state.phase = "discard"
        state.last_discard = None
        state.hand = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.assertNotIn(ActionType.GANG,
                         [action.kind for action in strict_legal_actions(state)])

    def test_bugang_owner_can_only_pass_on_broadcast(self):
        state = ProtocolState()
        state.player_id = 2
        state.hand = [0, 1, 2, 3]
        state.melds[2].append(Meld(ActionType.PENG, (3, 3, 3), 0))
        state.apply("3 2 BUGANG W4")
        self.assertEqual([action.kind for action in strict_legal_actions(state)],
                         [ActionType.PASS])

    def test_hu_is_not_allowed_without_official_calculator(self):
        state = ProtocolState()
        state.player_id = 1
        state.phase = "claim"
        state.last_discard = (0, 13)
        state.hand = [0, 1, 2, 3, 4, 5, 9, 9, 9, 27, 27, 27, 13]
        with mock.patch.object(default_backend, "has_official", False):
            self.assertNotIn(ActionType.HU,
                             [action.kind for action in strict_legal_actions(state)])

    def test_all_generated_actions_validate(self):
        state = ProtocolState()
        state.apply("0 1 0")
        state.apply("1 0 0 0 0 W1 W1 W1 W2 W3 W4 W5 W6 W7 W8 W9 T1 T2")
        state.apply("3 0 PLAY W1")
        for action in strict_legal_actions(state):
            self.assertEqual(validate_action(state, action), (True, "legal"))

    def test_audit_reports_own_discard_claim(self):
        events = [
            {"output": {"command": "request", "content": {
                "2": "0 2 1"}}},
            {"2": {"response": "PASS"}},
            {"output": {"command": "request", "content": {
                "2": "1 0 0 0 0 T6 T6 T6 B2 B2 B2 B5 B5 B5 F4 F4 F4 F4"}}},
            {"2": {"response": "PASS"}},
            {"output": {"command": "request", "content": {"2": "2 F4"}}},
            {"2": {"response": "PLAY F4"}},
            {"output": {"command": "request", "content": {"2": "3 2 PLAY F4"}}},
            {"2": {"response": "PENG B5"}},
        ]
        finding = audit_events(events, players=[2])[-1]
        self.assertEqual(finding["reason"], "cannot claim own discard")
        self.assertEqual(finding["response"], "PENG B5")

    def test_export_contains_main_and_no_model(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = os.path.join(directory, "bot.zip")
            subprocess.check_call([sys.executable, "scripts/export_bot.py", "--model", "missing.pt",
                                   "--output", archive, "--storage-model", os.path.join(directory, "model.pt")])
            with zipfile.ZipFile(archive) as handle:
                names = handle.namelist()
            self.assertIn("__main__.py", names)
            self.assertIn("mahjong_agent/training/rollout.py", names)
            self.assertIn("mahjong_agent/botzone/legality.py", names)
            self.assertFalse(any(name.endswith(".pt") for name in names))
            output = subprocess.check_output(
                [sys.executable, archive],
                input=json.dumps({"requests": ["0 1 0"], "responses": []}).encode("utf-8"))
            self.assertEqual(json.loads(output.decode("utf-8"))["response"], "PASS")


    def test_claimed_discards_do_not_create_false_fourth_tile(self):
        state = ProtocolState()
        state.player_id = 0
        state.phase = "claim"
        state.last_discard = (3, 2)
        state.discards[1] = [2]
        state.discards[2] = [2]
        state.discards[3] = [2]
        state.melds[1] = [Meld(ActionType.CHI, (0, 1, 2), 0)]
        state.melds[2] = [Meld(ActionType.CHI, (1, 2, 3), 1)]
        self.assertFalse(hu_context(state, False)["fourth_tile"])


if __name__ == "__main__":
    unittest.main()
