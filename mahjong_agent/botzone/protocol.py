"""Botzone protocol helpers.

This parser supports the public action vocabulary and is deliberately tolerant
of surrounding whitespace. Full replay is represented by ProtocolState so the
Botzone entry point does not depend on the local simulator's private wall.
"""

from mahjong_agent.engine.actions import Action, ActionType, Meld
from mahjong_agent.engine.tiles import name_to_tile, tile_to_name


def action_to_text(action, state=None):
    # 将内部 Action 序列化为 Botzone 接受的单行动作文本。
    if action.kind in (ActionType.PASS, ActionType.HU):
        return action.kind.name
    if action.kind == ActionType.GANG and state is not None and state.phase == "claim":
        return "GANG"
    if action.kind in (ActionType.PLAY, ActionType.GANG, ActionType.BUGANG):
        return "%s %s" % (action.kind.name, tile_to_name(action.tile))
    if action.kind == ActionType.PENG:
        return "PENG %s" % tile_to_name(action.discard)
    if action.kind == ActionType.CHI:
        return "CHI %s %s" % (
            tile_to_name(action.sequence[1]), tile_to_name(action.discard)
        )
    raise ValueError("unsupported action")


def parse_request(text):
    # 返回 tuple[str,...]；空白请求返回空 tuple。
    parts = text.strip().split()
    if not parts:
        return ()
    return tuple(parts)


class ProtocolState(object):
    def __init__(self):
        # 该状态仅通过 Botzone 历史请求重放得到，不持有完整牌墙或对手暗手。
        self.player_id = 0
        self.hand = []
        self.melds = [[] for _ in range(4)]
        self.discards = [[] for _ in range(4)]
        self.events = []
        self.current_player = 0
        self.prevalent_wind = 0
        self.phase = "ack"
        self.last_discard = None
        self.claim_hu_only = False
        self.drawn_tile = None
        self.flower_counts = [0, 0, 0, 0]
        # Botzone 以每家可摸牌数跟踪牌墙，初始每家21张。
        self.wall_remaining_by_player = [21, 21, 21, 21]
        self.wall_last = False

    def apply(self, request, previous_response=None):
        # request/previous_response 均为协议字符串；本方法原地推进公开状态。
        parts = parse_request(request)
        if not parts:
            return
        code = parts[0]
        # code 0=座位/圈风初始化；1=起手牌；2=自己摸牌；3=公开动作。
        if code == "0":
            self.player_id = int(parts[1])
            if len(parts) > 2:
                self.prevalent_wind = int(parts[2])
            self.phase = "ack"
        elif code == "1":
            # parts[1:5] 为四家花牌数，parts[5:18] 为自己的13张起手牌。
            self.flower_counts = [int(value) for value in parts[1:5]]
            self.hand = sorted(name_to_tile(tile) for tile in parts[5:18])
            self.current_player = self.player_id
            self.phase = "ack"
            self.last_discard = None
        elif code == "2":
            # 自己摸牌：暗手增加一张，进入 discard 阶段。
            self.current_player = self.player_id
            self.wall_remaining_by_player[self.player_id] = max(
                0, self.wall_remaining_by_player[self.player_id] - 1)
            self.hand.append(name_to_tile(parts[1]))
            self.drawn_tile = name_to_tile(parts[1])
            self.hand.sort()
            self.phase = "discard"
            self.wall_last = self.wall_remaining_by_player[(self.player_id + 1) % 4] == 0
            self.last_discard = None
            self.claim_hu_only = False
            self.events.append(("DRAW", self.player_id))
        elif code == "3":
            # 公开动作格式以 player 和 action 起始，后续字段依动作种类变化。
            player = int(parts[1])
            action = parts[2]
            self.current_player = player
            if action == "DRAW":
                self.wall_remaining_by_player[player] = max(
                    0, self.wall_remaining_by_player[player] - 1)
                self.events.append(("DRAW", player))
                self.phase = "ack"
                self.wall_last = self.wall_remaining_by_player[(player + 1) % 4] == 0
                self.last_discard = None
                self.claim_hu_only = False
                self.drawn_tile = None
            elif action == "BUHUA":
                self.flower_counts[player] += 1
                self.events.append(("BUHUA", player))
                self.phase = "ack"
                self.last_discard = None
                self.claim_hu_only = False
                self.drawn_tile = None
            elif action == "HU":
                self.events.append(("HU", player))
                self.phase = "ack"
                self.last_discard = None
                self.claim_hu_only = False
            elif action == "PLAY":
                tile = name_to_tile(parts[3])
                if player == self.player_id and tile in self.hand:
                    self.hand.remove(tile)
                self.discards[player].append(tile)
                self.last_discard = (player, tile)
                self.phase = "claim"
                self.wall_last = self.wall_remaining_by_player[(player + 1) % 4] == 0
                self.claim_hu_only = False
                self.events.append(("PLAY", player, tile))
                self.drawn_tile = None
            elif action in ("PENG", "CHI", "GANG", "BUGANG"):
                # last_source/claimed 是最近一张可被声明的弃牌；不存在时使用 -1 哨兵。
                last_source, claimed = self.last_discard if self.last_discard else (-1, -1)
                if action == "CHI":
                    middle = name_to_tile(parts[3])
                    discard = name_to_tile(parts[4])
                    sequence = (middle - 1, middle, middle + 1)
                    self.melds[player].append(Meld(ActionType.CHI, sequence, last_source))
                    if player == self.player_id:
                        needed = list(sequence)
                        needed.remove(claimed)
                        for tile in needed:
                            self.hand.remove(tile)
                        self.hand.remove(discard)
                    self.discards[player].append(discard)
                    self.last_discard = (player, discard)
                    self.phase = "claim"
                    self.wall_last = self.wall_remaining_by_player[(player + 1) % 4] == 0
                elif action == "PENG":
                    discard = name_to_tile(parts[3])
                    self.melds[player].append(
                        Meld(ActionType.PENG, (claimed,) * 3, last_source)
                    )
                    if player == self.player_id:
                        self.hand.remove(claimed)
                        self.hand.remove(claimed)
                        self.hand.remove(discard)
                    self.discards[player].append(discard)
                    self.last_discard = (player, discard)
                    self.phase = "claim"
                    self.wall_last = self.wall_remaining_by_player[(player + 1) % 4] == 0
                elif action == "GANG":
                    response_parts = parse_request(previous_response or "")
                    tile = claimed
                    if len(parts) > 3:
                        tile = name_to_tile(parts[3])
                    elif player == self.player_id and len(response_parts) > 1:
                        tile = name_to_tile(response_parts[1])
                    if tile >= 0:
                        self.melds[player].append(Meld(ActionType.GANG, (tile,) * 4, last_source))
                        if player == self.player_id:
                            remove_count = 3 if tile == claimed else 4
                            for _ in range(remove_count):
                                if tile in self.hand:
                                    self.hand.remove(tile)
                    self.last_discard = None
                    self.phase = "ack"
                    self.claim_hu_only = False
                elif action == "BUGANG":
                    tile = name_to_tile(parts[3])
                    for index, meld in enumerate(self.melds[player]):
                        if meld.kind == ActionType.PENG and meld.tiles[0] == tile:
                            self.melds[player][index] = Meld(
                                ActionType.GANG, (tile,) * 4, meld.from_player
                            )
                            break
                    if player == self.player_id:
                        self.hand.remove(tile)
                    self.last_discard = (player, tile)
                    self.phase = "claim"
                    self.claim_hu_only = True
                self.events.append(tuple(parts[2:]))

    def observation(self):
        # 输出结构与 MahjongEnv.observe 对齐，供编码器和策略复用。
        return {
            "player_id": self.player_id,
            "current_player": self.current_player,
            "phase": self.phase,
            "hand": list(self.hand),
            "melds": self.melds,
            "discards": self.discards,
            "wall_remaining": max(0, 83 - len(self.events)),
            "wall_remaining_by_player": list(self.wall_remaining_by_player),
            "prevalent_wind": self.prevalent_wind,
            "events": list(self.events[-128:]),
            "last_discard": self.last_discard,
            "wall_last": self.wall_last,
            "about_kong": self.claim_hu_only,
            "claim_hu_only": self.claim_hu_only,
        }
