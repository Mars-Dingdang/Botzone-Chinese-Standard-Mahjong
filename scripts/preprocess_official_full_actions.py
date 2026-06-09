#!/usr/bin/env python3
"""Build candidate-action BC shards from all decisions in official logs."""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mahjong_agent.engine import Action, ActionType, Meld
from mahjong_agent.engine.tiles import is_suited, name_to_tile
from mahjong_agent.features import compact_observation, serialize_action
from mahjong_agent.rules import default_backend
from scripts.preprocess_official_data import match_chunks


class FullActionState(object):
    def reset(self):
        self.wind = 0
        self.hands = [[] for _ in range(4)]
        self.melds = [[] for _ in range(4)]
        self.discards = [[] for _ in range(4)]
        self.events = []
        self.wall = [21] * 4
        self.last = None
        self.pending = {}
        self.claim_pending = {}

    def observation(self, player, phase):
        return {"player_id": player, "current_player": player, "phase": phase,
                "hand": list(self.hands[player]), "melds": [list(x) for x in self.melds],
                "discards": [list(x) for x in self.discards], "wall_remaining": sum(self.wall),
                "wall_remaining_by_player": list(self.wall), "events": list(self.events[-128:]),
                "last_discard": self.last, "prevalent_wind": self.wind}

    def _record(self, player, phase, actions):
        unique = list(dict((action.key(), action) for action in actions).values())
        return {"player": player, "observation": compact_observation(self.observation(player, phase)),
                "legal": unique}

    @staticmethod
    def _can_hu_fast(counts, melds):
        required_tiles = (4 - len(melds)) * 3 + 2
        if sum(counts) != required_tiles:
            return False
        if not any(value >= 2 for value in counts):
            return False
        if default_backend.is_complete_hand(counts, melds):
            return True
        if melds:
            return False
        terminals = {0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33}
        present = {tile for tile, value in enumerate(counts) if value}
        return terminals.issubset(present) and any(counts[tile] >= 2 for tile in terminals)

    def _draw_actions(self, player, drawn):
        actions = [Action.play(tile) for tile in sorted(set(self.hands[player]))]
        counts = Counter(self.hands[player])
        actions.extend(Action(ActionType.GANG, tile) for tile, count in counts.items() if count == 4)
        actions.extend(Action(ActionType.BUGANG, meld.tiles[0]) for meld in self.melds[player]
                       if meld.kind == ActionType.PENG and counts[meld.tiles[0]])
        raw = [counts.get(tile, 0) for tile in range(34)]
        context = {"player_id": player, "seat_wind": player, "prevalent_wind": self.wind,
                   "self_drawn": True, "fourth_tile": False, "about_kong": False,
                   "wall_last": sum(self.wall) == 0, "flower_count": 0}
        if self._can_hu_fast(raw, self.melds[player]) and default_backend.can_hu(raw, self.melds[player], drawn, context=context):
            actions.append(Action.hu())
        return actions

    def _claim_actions(self, player):
        source, tile = self.last
        actions = [Action.pass_()]
        counts = Counter(self.hands[player])
        if counts[tile] >= 2:
            remaining = list(self.hands[player]); remaining.remove(tile); remaining.remove(tile)
            actions.extend(Action(ActionType.PENG, tile, (), discard) for discard in sorted(set(remaining)))
        if counts[tile] >= 3:
            actions.append(Action(ActionType.GANG, tile))
        if player == (source + 1) % 4 and is_suited(tile):
            base, rank = tile - tile % 9, tile % 9
            for start in range(max(0, rank - 2), min(6, rank) + 1):
                seq = (base + start, base + start + 1, base + start + 2)
                needed = list(seq); needed.remove(tile)
                if all(counts[item] >= needed.count(item) for item in set(needed)):
                    for discard in sorted(set(self.hands[player])):
                        if discard not in needed or counts[discard] > needed.count(discard):
                            actions.append(Action(ActionType.CHI, tile, seq, discard))
        raw = [counts.get(index, 0) for index in range(34)]; raw[tile] += 1
        context = {"player_id": player, "seat_wind": player, "prevalent_wind": self.wind,
                   "self_drawn": False, "fourth_tile": False, "about_kong": False,
                   "wall_last": sum(self.wall) == 0, "flower_count": 0}
        if self._can_hu_fast(raw, self.melds[player]) and default_backend.can_hu(raw, self.melds[player], tile, context=context):
            actions.append(Action.hu())
        return actions

    @staticmethod
    def finalize(record, action, action_family):
        actions = record["legal"]
        if action.key() not in {item.key() for item in actions}:
            actions.append(action)
        if len(actions) <= 1:
            return None
        return {"observation": record["observation"], "actions_raw": [serialize_action(x) for x in actions],
                "target": next(i for i, item in enumerate(actions) if item.key() == action.key()),
                "action_family": action_family}

    def apply(self, parts):
        rows = []
        if parts[0] == "Wind": self.wind = int(parts[1]); return rows
        if parts[0] != "Player": return rows
        player, kind = int(parts[1]), parts[2]
        if kind == "Deal": self.hands[player] = sorted(name_to_tile(x) for x in parts[3:16]); return rows
        if kind == "Draw":
            for other in list(self.pending):
                rows.append(self.finalize(self.pending.pop(other), Action.pass_(), "PASS"))
            tile = name_to_tile(parts[3]); self.wall[player] = max(0, self.wall[player] - 1)
            self.hands[player].append(tile); self.hands[player].sort(); self.events.append(("DRAW", player))
            self.pending[player] = self._record(player, "discard", self._draw_actions(player, tile)); return rows
        if kind == "Play":
            tile = name_to_tile(parts[3])
            if player in self.claim_pending:
                record, base = self.claim_pending.pop(player)
                action = Action(base.kind, base.tile, base.sequence, tile)
                rows.append(self.finalize(record, action, base.kind.name))
            elif player in self.pending:
                rows.append(self.finalize(self.pending.pop(player), Action.play(tile), "PLAY"))
            self.hands[player].remove(tile); self.discards[player].append(tile); self.last = (player, tile)
            self.events.append(("PLAY", player, tile))
            for other in range(4):
                if other != player: self.pending[other] = self._record(other, "claim", self._claim_actions(other))
            return rows
        if kind in ("Chi", "Peng"):
            ignored = {int(parts[index + 2]) for index, token in enumerate(parts)
                       if token == "Ignore" and index + 2 < len(parts)}
            for other in list(self.pending):
                if other != player:
                    record = self.pending.pop(other)
                    if other not in ignored: rows.append(self.finalize(record, Action.pass_(), "PASS"))
            record = self.pending.pop(player)
            tile = self.last[1]
            if kind == "Chi":
                middle = name_to_tile(parts[3]); seq = (middle - 1, middle, middle + 1); needed = list(seq); needed.remove(tile)
                for item in needed: self.hands[player].remove(item)
                self.melds[player].append(Meld(ActionType.CHI, seq, self.last[0])); base = Action(ActionType.CHI, tile, seq)
            else:
                for _ in range(2): self.hands[player].remove(tile)
                self.melds[player].append(Meld(ActionType.PENG, (tile,) * 3, self.last[0])); base = Action(ActionType.PENG, tile)
            self.claim_pending[player] = (record, base); self.events.append((kind.upper(), player, tile)); return rows
        if kind in ("Gang", "AnGang", "BuGang", "Hu"):
            ignored = {int(parts[index + 2]) for index, token in enumerate(parts)
                       if token == "Ignore" and index + 2 < len(parts)}
            action_kind = {"Gang": ActionType.GANG, "AnGang": ActionType.GANG,
                           "BuGang": ActionType.BUGANG, "Hu": ActionType.HU}[kind]
            tile = name_to_tile(parts[3]) if len(parts) > 3 else (self.last[1] if self.last else -1)
            action = Action.hu() if kind == "Hu" else Action(action_kind, tile)
            if player in self.pending: rows.append(self.finalize(self.pending.pop(player), action, kind.upper()))
            for other in list(self.pending):
                record = self.pending.pop(other)
                if other not in ignored: rows.append(self.finalize(record, Action.pass_(), "PASS"))
            if kind == "Gang":
                for _ in range(3): self.hands[player].remove(tile)
                self.melds[player].append(Meld(ActionType.GANG, (tile,) * 4, self.last[0]))
            elif kind == "AnGang":
                for _ in range(4): self.hands[player].remove(tile)
                self.melds[player].append(Meld(ActionType.GANG, (tile,) * 4, player))
            elif kind == "BuGang":
                self.hands[player].remove(tile)
            self.events.append((kind.upper(), player, tile)); return rows
        return rows


def process_chunk(task):
    chunk_id, matches, output_dir, compression = task
    import pyarrow as pa
    import pyarrow.parquet as pq
    state = FullActionState(); rows = []; failures = 0; families = Counter()
    for match_id, lines in matches:
        state.reset(); split = "val" if match_id % 20 == 0 else "train"
        for parts in lines:
            try:
                for row in state.apply(parts):
                    if row is None: continue
                    row["split"] = split; rows.append(row); families[row["action_family"]] += 1
            except (ValueError, IndexError, KeyError): failures += 1; state.pending.clear(); state.claim_pending.clear()
    if rows: pq.write_table(pa.Table.from_pylist(rows), os.path.join(output_dir, "part-%05d.parquet" % chunk_id), compression=compression)
    return len(matches), len(rows), failures, dict(families)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--input", default="Chinese-Standard-Mahjong/SL/data/data.txt")
    parser.add_argument("--output-dir", default="artifacts/official_bc_full_v2"); parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--chunk-matches", type=int, default=500); parser.add_argument("--max-matches", type=int, default=0)
    args = parser.parse_args(); os.makedirs(args.output_dir, exist_ok=True); started = time.time(); totals = [0, 0, 0]; families = Counter()
    tasks = ((i, chunk, args.output_dir, "zstd") for i, chunk in enumerate(match_chunks(args.input, args.chunk_matches, args.max_matches)))
    with mp.get_context("fork").Pool(args.workers, maxtasksperchild=8) as pool:
        for done, (matches, samples, failures, counts) in enumerate(pool.imap_unordered(process_chunk, tasks), 1):
            totals = [totals[0] + matches, totals[1] + samples, totals[2] + failures]; families.update(counts)
            if done % 4 == 0:
                elapsed = time.time() - started
                print("chunks=%d matches=%d samples=%d failures=%d rate=%.1f matches/s" % (
                    done, totals[0], totals[1], totals[2], totals[0] / max(elapsed, 1e-6)
                ), flush=True)
    metadata = {"version": 2, "matches": totals[0], "samples": totals[1], "failures": totals[2], "families": dict(families), "seconds": time.time() - started}
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as handle: json.dump(metadata, handle, indent=2, sort_keys=True)
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__": main()
