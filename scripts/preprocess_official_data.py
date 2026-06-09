#!/usr/bin/env python3
"""Parallel conversion of official Botzone logs into compressed Parquet shards."""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mahjong_agent.engine import Action, ActionType, Meld
from mahjong_agent.engine.tiles import name_to_tile
from mahjong_agent.features import encode_action, encode_observation


class State(object):
    def reset(self):
        self.wind = 0
        self.hands = [[] for _ in range(4)]
        self.melds = [[] for _ in range(4)]
        self.discards = [[] for _ in range(4)]
        self.events = []
        self.wall = [21] * 4
        self.last = None

    def observation(self, player):
        return {
            "player_id": player, "current_player": player, "phase": "discard",
            "hand": list(self.hands[player]), "melds": [list(x) for x in self.melds],
            "discards": [list(x) for x in self.discards], "wall_remaining": sum(self.wall),
            "wall_remaining_by_player": list(self.wall), "events": list(self.events[-128:]),
            "last_discard": self.last, "prevalent_wind": self.wind,
        }

    def apply(self, parts):
        if parts[0] == "Wind":
            self.wind = int(parts[1]); return None
        if parts[0] != "Player": return None
        player, kind = int(parts[1]), parts[2]
        if kind == "Deal": self.hands[player] = sorted(name_to_tile(x) for x in parts[3:16])
        elif kind == "Draw":
            self.wall[player] = max(0, self.wall[player] - 1); self.hands[player].append(name_to_tile(parts[3])); self.hands[player].sort(); self.events.append(("DRAW", player))
        elif kind == "Play":
            tile = name_to_tile(parts[3]); legal = [Action.play(x) for x in sorted(set(self.hands[player]))]
            if not legal or tile not in self.hands[player]: return None
            record = {"features": encode_observation(self.observation(player)), "actions": [encode_action(x) for x in legal], "target": next(i for i, x in enumerate(legal) if x.tile == tile)}
            self.hands[player].remove(tile); self.discards[player].append(tile); self.last = (player, tile); self.events.append(("PLAY", player, tile)); return record
        elif kind == "Chi":
            middle = name_to_tile(parts[3]); sequence = (middle - 1, middle, middle + 1); needed = list(sequence); needed.remove(self.last[1])
            for tile in needed: self.hands[player].remove(tile)
            self.melds[player].append(Meld(ActionType.CHI, sequence, self.last[0])); self.events.append(("CHI", player, middle))
        elif kind == "Peng":
            tile = name_to_tile(parts[3]); self.hands[player].remove(tile); self.hands[player].remove(tile); self.melds[player].append(Meld(ActionType.PENG, (tile,) * 3, self.last[0])); self.events.append(("PENG", player, tile))
        elif kind == "Gang":
            tile = name_to_tile(parts[3])
            for _ in range(3): self.hands[player].remove(tile)
            self.melds[player].append(Meld(ActionType.GANG, (tile,) * 4, self.last[0])); self.events.append(("GANG", player, tile))
        elif kind == "AnGang":
            tile = name_to_tile(parts[3])
            for _ in range(4): self.hands[player].remove(tile)
            self.melds[player].append(Meld(ActionType.GANG, (tile,) * 4, player)); self.events.append(("GANG", player, tile))
        elif kind == "BuGang":
            tile = name_to_tile(parts[3]); self.hands[player].remove(tile)
            for index, meld in enumerate(self.melds[player]):
                if meld.kind == ActionType.PENG and meld.tiles[0] == tile:
                    self.melds[player][index] = Meld(ActionType.GANG, (tile,) * 4, meld.from_player); break
            self.events.append(("BUGANG", player, tile))
        return None


def process_chunk(task):
    chunk_id, matches, output_dir, compression = task
    import pyarrow as pa
    import pyarrow.parquet as pq
    state = State(); rows = []; failures = 0; train = val = 0
    for match_id, lines in matches:
        state.reset(); split = "val" if match_id % 20 == 0 else "train"
        for parts in lines:
            try:
                record = state.apply(parts)
                if record is not None:
                    record["split"] = split; rows.append(record)
                    if split == "train": train += 1
                    else: val += 1
            except (ValueError, IndexError): failures += 1
    if rows:
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, os.path.join(output_dir, "part-%05d.parquet" % chunk_id), compression=compression, use_dictionary=False)
    return len(matches), train, val, failures


def match_chunks(path, chunk_matches, max_matches=0):
    chunk = []; current_id = -1; current = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if not parts: continue
            if parts[0] == "Match":
                if current_id >= 0: chunk.append((current_id, current))
                if len(chunk) >= chunk_matches: yield chunk; chunk = []
                current_id += 1; current = []
                if max_matches and current_id >= max_matches: break
            else: current.append(parts)
        else:
            if current_id >= 0: chunk.append((current_id, current))
    if chunk: yield chunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="Chinese-Standard-Mahjong/SL/data/data.txt")
    parser.add_argument("--output-dir", default="artifacts/official_bc")
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--chunk-matches", type=int, default=250)
    parser.add_argument("--max-matches", type=int, default=0)
    parser.add_argument("--compression", default="zstd", choices=("zstd", "snappy", "none"))
    args = parser.parse_args(); os.makedirs(args.output_dir, exist_ok=True)
    started = time.time(); totals = [0, 0, 0, 0]
    tasks = ((i, chunk, args.output_dir, None if args.compression == "none" else args.compression) for i, chunk in enumerate(match_chunks(args.input, args.chunk_matches, args.max_matches)))
    context = mp.get_context("fork")
    with context.Pool(args.workers, maxtasksperchild=8) as pool:
        for done, result in enumerate(pool.imap_unordered(process_chunk, tasks), 1):
            totals = [a + b for a, b in zip(totals, result)]
            if done % 4 == 0:
                elapsed = time.time() - started
                print("matches=%d train=%d val=%d failures=%d rate=%.1f matches/s" % (*totals, totals[0] / max(elapsed, 1e-6)), flush=True)
    metadata = {"matches": totals[0], "train": totals[1], "val": totals[2], "failures": totals[3], "workers": args.workers, "format": "parquet", "seconds": time.time() - started}
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as handle: json.dump(metadata, handle, indent=2, sort_keys=True)
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__": main()
