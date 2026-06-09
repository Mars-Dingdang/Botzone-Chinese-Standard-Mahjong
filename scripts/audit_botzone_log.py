#!/usr/bin/env python3
"""Audit Botzone match logs against the strict outbound legality layer."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mahjong_agent.botzone.legality import response_to_action, validate_action
from mahjong_agent.botzone.protocol import ProtocolState
from mahjong_agent.engine.tiles import tile_to_name


def state_summary(state):
    return {
        "player_id": state.player_id,
        "phase": state.phase,
        "last_discard": (
            [state.last_discard[0], tile_to_name(state.last_discard[1])]
            if state.last_discard else None),
        "hand": [tile_to_name(tile) for tile in state.hand],
        "wall_remaining_by_player": list(state.wall_remaining_by_player),
        "wall_last": state.wall_last,
        "claim_hu_only": state.claim_hu_only,
    }


def audit_events(events, players=None):
    players = set(range(4) if players is None else players)
    states = dict((player, ProtocolState()) for player in players)
    previous = dict((player, None) for player in players)
    findings = []
    turn = 0
    for index, event in enumerate(events):
        output = event.get("output") if isinstance(event, dict) else None
        if not output or output.get("command") != "request":
            continue
        if index + 1 >= len(events) or not isinstance(events[index + 1], dict):
            continue
        responses = events[index + 1]
        turn += 1
        for player in sorted(players):
            key = str(player)
            request = output.get("content", {}).get(key)
            response = responses.get(key, {}).get("response")
            if request is None or response is None:
                continue
            state = states[player]
            try:
                state.apply(request, previous[player])
                action = response_to_action(state, response)
                valid, reason = validate_action(state, action)
            except Exception as exc:
                valid, reason = False, "audit parse/replay error: %s" % exc
            if not valid:
                findings.append({
                    "turn": turn, "player": player, "request": request,
                    "response": response, "reason": reason,
                    "state": state_summary(state),
                })
            previous[player] = response
    return findings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    parser.add_argument("--player", type=int, action="append", choices=range(4))
    parser.add_argument("--all", action="store_true", help="print every finding")
    args = parser.parse_args()
    with open(args.log) as handle:
        events = json.load(handle)
    findings = audit_events(events, args.player)
    shown = findings if args.all else findings[:1]
    print(json.dumps({"findings": shown, "total_findings": len(findings)},
                     indent=2, sort_keys=True))
    raise SystemExit(1 if findings else 0)


if __name__ == "__main__":
    main()
