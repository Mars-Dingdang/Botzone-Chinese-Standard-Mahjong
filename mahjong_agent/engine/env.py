"""Single-hand four-player Mahjong environment.

The environment implements turn progression, draws, discards, chi/peng/gang
claims and a configurable rules backend. Flower replacement and multi-winner
settlement are intentionally outside the first Botzone-focused release.
"""

import random
from collections import Counter

from .actions import Action, ActionType, Meld
from .tiles import full_wall, is_suited
from mahjong_agent.rules import default_backend
from mahjong_agent.rules.legality import action_allowed_in_claim, can_kong


class MahjongEnv(object):
    def __init__(self, rules=None, min_fan=8):
        # rules 提供 can_hu/fan/shanten 等接口；min_fan 是合法和牌最低番数。
        self.rules = rules or default_backend
        self.min_fan = min_fan
        self.reset()

    def reset(self, seed=None, wall=None, prevalent_wind=None):
        # 使用局部 RNG，确保给定 seed 时不污染进程级随机状态。
        rng = random.Random(seed)
        # wall 是 tile id[int] 列表；列表尾部视为下一张摸牌。
        self.wall = list(wall) if wall is not None else full_wall()
        if wall is None:
            rng.shuffle(self.wall)
        if len(self.wall) < 53:
            raise ValueError("wall must contain enough tiles to deal")
        # 四家相关容器均按绝对座位 0..3 索引。
        self.hands = [[] for _ in range(4)]
        self.melds = [[] for _ in range(4)]
        self.discards = [[] for _ in range(4)]
        self.events = []
        self.current_player = 0
        self.prevalent_wind = (seed or 0) % 4 if prevalent_wind is None else int(prevalent_wind)
        # phase 状态机：draw -> discard -> claim -> draw，终局为 terminal。
        self.phase = "draw"
        self.last_discard = None
        self.pending_claimers = []
        self.claim_responses = {}
        self.terminal = False
        self.winner = None
        self.loser = None
        self.fan_count = 0
        self.drawn_tile = None
        self.about_kong = False
        self.claim_hu_only = False
        self.pending_bugang = None
        self.invalid_actions = 0
        # 轮流发牌，每家初始13张，然后庄家/当前玩家再摸一张。
        for _ in range(13):
            for player in range(4):
                self.hands[player].append(self.wall.pop())
        for hand in self.hands:
            hand.sort()
        self._draw()
        return self.observe(0)

    def _draw(self):
        # 摸空牌墙时直接流局，否则当前玩家手牌由13张变为14张。
        if not self.wall:
            self.terminal = True
            self.phase = "terminal"
            self.events.append(("DRAW_GAME",))
            return
        tile = self.wall.pop()
        self.drawn_tile = tile
        self.hands[self.current_player].append(tile)
        self.hands[self.current_player].sort()
        self.phase = "discard"
        self.events.append(("DRAW", self.current_player))

    def _counts(self, player):
        # 返回指定玩家的 34 维整数计数向量。
        counter = Counter(self.hands[player])
        return [counter.get(tile, 0) for tile in range(34)]

    def observe(self, player_id):
        # 仅返回 player_id 可见的公开信息及自己的手牌；不泄漏其他玩家暗手。
        return {
            "player_id": player_id,
            "current_player": self.current_player,
            "phase": self.phase,
            "hand": list(self.hands[player_id]),
            "melds": [list(melds) for melds in self.melds],
            "discards": [list(tiles) for tiles in self.discards],
            "wall_remaining": len(self.wall),
            "wall_remaining_by_player": [len(self.wall) // 4] * 4,
            "prevalent_wind": self.prevalent_wind,
            "events": list(self.events[-128:]),
            "last_discard": self.last_discard,
            "wall_last": not self.wall,
            "about_kong": self.about_kong,
            "claim_hu_only": self.claim_hu_only,
        }

    def _claim_actions(self, player):
        # last_discard 格式为 (source_player:int, tile:int)。
        source, tile = self.last_discard
        actions = [Action.pass_()]
        if player == source:
            return actions
        # 判断荣和时，临时把被弃牌加入 34 维手牌计数。
        counts = self._counts(player)
        counts[tile] += 1
        if self.rules.can_hu(counts, self.melds[player], tile,
                             context=self._fan_context(player, False), min_fan=self.min_fan):
            actions.append(Action.hu())
        if self.claim_hu_only:
            return actions
        counts[tile] -= 1
        hand_counts = self._counts(player)
        if not self.wall and hand_counts[tile] >= 2:
            return actions
        # 碰后必须立即弃一张，因此每种可弃牌形成一个独立候选 Action。
        if hand_counts[tile] >= 2:
            remaining = list(self.hands[player])
            remaining.remove(tile)
            remaining.remove(tile)
            for discard in sorted(set(remaining)):
                actions.append(Action(ActionType.PENG, tile, (), discard))
        if hand_counts[tile] >= 3 and can_kong(not self.wall, len(self.wall)):
            actions.append(Action(ActionType.GANG, tile))
        # 只有弃牌者下家可吃，且字牌不可组成顺子。
        if self.wall and player == (source + 1) % 4 and is_suited(tile):
            base = tile - tile % 9
            rank_index = tile % 9
            for start in range(max(0, rank_index - 2), min(6, rank_index) + 1):
                seq = (base + start, base + start + 1, base + start + 2)
                needed = list(seq)
                needed.remove(tile)
                if all(hand_counts[item] >= needed.count(item) for item in set(needed)):
                    for discard in sorted(set(self.hands[player])):
                        if discard not in needed or hand_counts[discard] > needed.count(discard):
                            actions.append(Action(ActionType.CHI, tile, seq, discard))
        return actions

    def legal_actions(self, player_id=None):
        # 返回 list[Action]；非当前玩家或终局状态没有合法动作。
        player = self.current_player if player_id is None else player_id
        if self.terminal or player != self.current_player:
            return []
        if self.phase == "claim":
            return self._claim_actions(player)
        # discard 阶段默认可打出手中任意一种牌；相同牌只需一个候选动作。
        actions = [Action.play(tile) for tile in sorted(set(self.hands[player]))]
        counts = self._counts(player)
        if self.drawn_tile is not None and self.rules.can_hu(
                counts, self.melds[player], self.drawn_tile,
                context=self._fan_context(player, True), min_fan=self.min_fan):
            actions.append(Action.hu())
        for tile, count in enumerate(counts):
            if count == 4 and can_kong(not self.wall, len(self.wall)):
                actions.append(Action(ActionType.GANG, tile))
        for meld in self.melds[player]:
            if (meld.kind == ActionType.PENG and counts[meld.tiles[0]]
                    and can_kong(not self.wall, len(self.wall))):
                actions.append(Action(ActionType.BUGANG, meld.tiles[0]))
        return actions

    def step(self, action):
        # 用规范 key 验证动作，避免调用方伪造具有相同类型但非法参数的动作。
        legal = dict((item.key(), item) for item in self.legal_actions())
        if action.key() not in legal:
            self.invalid_actions += 1
            raise ValueError("illegal action: %r" % (action,))
        action = legal[action.key()]
        player = self.current_player
        # 下列分支实现状态机迁移；step 最终返回动作玩家视角的 observation。
        if action.kind == ActionType.HU:
            if self.phase == "claim":
                self._record_claim(action)
            else:
                self._finish_hu(player, True)
        elif action.kind == ActionType.PLAY:
            # 出牌后依次询问其余三家是否声明吃碰杠和。
            self.hands[player].remove(action.tile)
            self.discards[player].append(action.tile)
            self.last_discard = (player, action.tile)
            self.events.append(("PLAY", player, action.tile))
            self.about_kong = False
            self.pending_claimers = [(player + offset) % 4 for offset in (1, 2, 3)]
            self.claim_responses = {}
            self.current_player = self.pending_claimers[0]
            self.phase = "claim"
            self.drawn_tile = None
        elif action.kind == ActionType.PASS:
            self._record_claim(action)
        elif action.kind in (ActionType.CHI, ActionType.PENG):
            self._record_claim(action)
        elif action.kind == ActionType.GANG:
            # claim 阶段的明杠要参与声明优先级；暗杠可立即成立并补牌。
            if self.phase == "claim":
                self._record_claim(action)
                return self.observe(player)
            else:
                tile = action.tile
                for _ in range(4):
                    self.hands[player].remove(tile)
                self.melds[player].append(Meld(ActionType.GANG, (tile,) * 4, player))
            self.events.append(("GANG", player, tile))
            self.last_discard = None
            self.phase = "draw"
            self.about_kong = True
            self._draw()
        elif action.kind == ActionType.BUGANG:
            # 补杠先进入只允许 HU/PASS 的抢杠声明阶段。
            tile = action.tile
            self.events.append(("BUGANG", player, tile))
            self.last_discard = (player, tile)
            self.pending_bugang = (player, tile)
            self.pending_claimers = [(player + offset) % 4 for offset in (1, 2, 3)]
            self.claim_responses = {}
            self.current_player = self.pending_claimers[0]
            self.phase = "claim"
            self.claim_hu_only = True
            self.about_kong = True
        return self.observe(player)

    def _fan_context(self, player, self_drawn):
        # 组装规则后端所需的和牌上下文；visible 用于判断是否为绝张。
        tile = self.drawn_tile if self_drawn else (self.last_discard[1] if self.last_discard else -1)
        visible = sum(river.count(tile) for river in self.discards)
        for owner, melds in enumerate(self.melds):
            for meld in melds:
                visible += meld.tiles.count(tile)
                if meld.from_player != owner and tile in meld.tiles:
                    visible -= 1
        return {"player_id": player, "seat_wind": player,
                "prevalent_wind": self.prevalent_wind, "self_drawn": self_drawn,
                "fourth_tile": tile >= 0 and visible + int(self_drawn) >= 4,
                "about_kong": self.about_kong, "wall_last": not self.wall,
                "flower_count": 0}

    def _finish_hu(self, player, self_drawn):
        # 自摸时手牌计数已含和牌；荣和时需临时加入最近弃牌。
        win_tile = self.drawn_tile if self_drawn else self.last_discard[1]
        counts = self._counts(player)
        if not self_drawn:
            counts[win_tile] += 1
        self.fan_count = self.rules.fan(counts, self.melds[player], win_tile,
                                        self._fan_context(player, self_drawn))
        self.terminal = True
        self.phase = "terminal"
        self.winner = player
        self.loser = None if self_drawn else self.last_discard[0]
        self.events.append(("HU", player, self.loser, self.fan_count))

    def _record_claim(self, action):
        # claim_responses: dict[player_id, Action]，收齐三家后统一按优先级裁决。
        self.claim_responses[self.current_player] = action
        remaining = [item for item in self.pending_claimers if item not in self.claim_responses]
        if remaining:
            self.current_player = remaining[0]
        else:
            self._resolve_claims()

    def _resolve_claims(self):
        # 声明优先级为 HU > PENG/GANG > CHI；同类按 pending_claimers 座次顺序。
        source, tile = self.last_discard
        for kind in (ActionType.HU, ActionType.PENG, ActionType.GANG, ActionType.CHI):
            for player in self.pending_claimers:
                action = self.claim_responses[player]
                if action.kind != kind or (kind == ActionType.CHI and player != (source + 1) % 4):
                    continue
                self.current_player = player
                if kind == ActionType.HU:
                    self._finish_hu(player, False)
                    return
                if not action_allowed_in_claim(
                        kind, player, source, not self.wall, self.claim_hu_only,
                        len(self.wall)):
                    continue
                if kind in (ActionType.CHI, ActionType.PENG):
                    # 从暗手移除组成副露所需的牌，再执行动作携带的 discard。
                    required = list(action.sequence if kind == ActionType.CHI else (tile, tile, tile))
                    required.remove(tile)
                    for item in required:
                        self.hands[player].remove(item)
                    meld_tiles = tuple(action.sequence) if kind == ActionType.CHI else (tile,) * 3
                    self.melds[player].append(Meld(kind, meld_tiles, source))
                    self.hands[player].remove(action.discard)
                    self.discards[player].append(action.discard)
                    self.last_discard = (player, action.discard)
                    self.events.append((kind.name, player, tile, action.discard))
                    self.pending_claimers = [(player + offset) % 4 for offset in (1, 2, 3)]
                    self.claim_responses = {}
                    self.current_player = self.pending_claimers[0]
                    self.phase = "claim"
                    return
                for _ in range(3):
                    self.hands[player].remove(tile)
                self.melds[player].append(Meld(ActionType.GANG, (tile,) * 4, source))
                self.events.append(("GANG", player, tile))
                self.last_discard = None
                self.phase = "draw"
                self.about_kong = True
                self._draw()
                return
        if self.pending_bugang is not None:
            # 无人抢杠后，才把原碰牌升级为杠并摸补牌。
            player, tile = self.pending_bugang
            self.hands[player].remove(tile)
            index = next(i for i, meld in enumerate(self.melds[player])
                         if meld.kind == ActionType.PENG and meld.tiles[0] == tile)
            old = self.melds[player][index]
            self.melds[player][index] = Meld(ActionType.GANG, (tile,) * 4, old.from_player)
            self.pending_bugang = None
            self.claim_hu_only = False
            self.current_player = player
            self.last_discard = None
            self.phase = "draw"
            self._draw()
            return
        # 所有人 PASS：轮到弃牌者下家摸牌。
        self.current_player = (source + 1) % 4
        self.last_discard = None
        self.phase = "draw"
        self.about_kong = False
        self.claim_hu_only = False
        self._draw()

    def is_terminal(self):
        return self.terminal

    def result(self):
        # scores 是长度4的整数列表，和为0；按国标麻将基础分结算。
        scores = [0, 0, 0, 0]
        if self.winner is not None:
            fan = max(self.min_fan, self.fan_count)
            if self.loser is None:
                scores = [-(8 + fan) if i != self.winner else 3 * (8 + fan)
                          for i in range(4)]
            else:
                scores = [-8] * 4
                scores[self.winner] = 24 + fan
                scores[self.loser] -= fan
        return {
            "winner": self.winner,
            "loser": self.loser,
            "scores": scores,
            "draw": self.winner is None,
            "invalid_actions": self.invalid_actions,
            "fan_count": self.fan_count,
            "self_drawn": self.winner is not None and self.loser is None,
        }

    def full_state(self):
        """Privileged training labels; never returned by ``observe``."""
        # 该接口含所有暗手和牌墙统计，只能用于训练标签，不能输入策略模型。
        return {
            "hands": [list(hand) for hand in self.hands],
            "wall_counts": dict(Counter(self.wall)),
            "scores": self.result()["scores"] if self.terminal else None,
            "winner": self.winner,
            "loser": self.loser,
            "fan_count": self.fan_count,
        }
