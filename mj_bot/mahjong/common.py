from typing import List
import json
from functools import reduce
from contextlib import contextmanager
import numpy as np

from MahjongGB import MahjongFanCalculator

BUHUA = "BUHUA"
DRAW = "DRAW"
PLAY = "PLAY"
PENG = "PENG"
CHI = "CHI"
GANG = "GANG"
MINGGANG = "MINGGANG"
ANGANG = "ANGANG"
BUGANG = "BUGANG"
PASS = "PASS"
HU = "HU"

CARDS = []
CARDS.extend(f"{t}{n}" for n in range(1, 10) for t in "WBT")
CARDS.extend(f"F{n}" for n in range(1, 5))
CARDS.extend(f"J{n}" for n in range(1, 4))
CARDS.extend(f"H{n}" for n in range(1, 9))

ZH2EN = {
    "摸牌": DRAW,
    "打牌": PLAY,
    "碰": PENG,
    "吃": CHI,
    "补花": BUHUA,
    "补花后摸牌": DRAW,
    "杠": GANG,
    "杠后摸牌": DRAW,
    "补杠": BUGANG,
    "明杠": MINGGANG,
    "暗杠": ANGANG,
    "和牌": HU,
    "自摸": HU,
}

[ZH2EN.update({tk: tk}) for tk in
 ("东", "南", "西", "北", "荒庄",)]
[ZH2EN.update({tk: tk}) for tk in CARDS]


def card_type(card):
    return card[0]


def card_number(card):
    if card[0] in "WBT":
        return int(card[1])
    return None


def make_card_seq(mid_card):
    ct = card_type(mid_card)
    cn = card_number(mid_card)
    if cn and 2 <= cn <= 8:
        return [ct + str(cn - 1), mid_card, ct + str(cn + 1)]
    return None


def chi_space(hand, card):
    ct = card_type(card)
    cn = card_number(card)
    if not cn:
        return []
    space = []
    for n in [cn - 1, cn, cn + 1]:
        if 2 <= n <= 8:
            mid_card = ct + str(n)
            cards = [ct + str(n - 1), ct + str(n), ct + str(n + 1)]
            cards.remove(card)
            if all(c in hand for c in cards):
                tmp_hand: List[str] = hand[:]
                [tmp_hand.remove(c) for c in cards]
                space.extend((CHI, mid_card, c) for c in tmp_hand)
    return space


def peng_space(hand: List[str], card):
    if hand.count(card) >= 2:
        tmp_hand = hand[:]
        [tmp_hand.remove(c) for c in (card, card)]
        return [(PENG, c) for c in tmp_hand]
    return []


def encode_tile(tile):
    if tile[0] == 'W':
        row = int(tile[1]) - 1  # 0-8是万
    elif tile[0] == 'T':
        row = int(tile[1]) + 8  # 9-17是条
    elif tile[0] == 'B':
        row = int(tile[1]) + 17  # 18-26是筒
    elif tile[0] == 'F':
        row = int(tile[1]) + 26  # 27-30分别是东南西北
    elif tile[0] == 'J':
        row = int(tile[1]) + 30  # 31-33分别是红中、发财、白板
    else:
        assert False, tile
    return row


# 将手牌转换成34×4的二维数组
def tiles_to_plane(tiles):
    plane = np.zeros(136, dtype=int).reshape(-1, 4)
    for tile in tiles:
        if tile[0] == 'H':
            continue
        row = encode_tile(tile)
        for i in range(4):
            if plane[row][i] == 0:
                plane[row][i] = 1
                break
    return plane


def melds_to_plane(melds):
    # 将所有吃、碰、杠的牌展开成一维
    return tiles_to_plane([tile for meld in melds for tile in meld.cards])


class Meld:
    def __init__(self, meld_type, cards, src_pid, src_card):
        self.type: str = meld_type
        self.cards: List[str] = cards
        self.src_pid: int = src_pid
        self.src_card: str = src_card

    def cards_from_self(self):
        cards = self.cards[:]
        if self.src_card is not None:
            cards.remove(self.src_card)
        return cards

    def to_pack(self, my_pid):
        pack_type = self.type
        tile_code = self.cards[1]
        if pack_type == CHI:
            data = self.cards.index(self.src_card) + 1
        else:
            data = 4 - (self.src_pid - my_pid) % 4
        return pack_type, tile_code, data


class Player:
    def __init__(self, player_id):
        self.player_id: int = player_id
        self.melds: List[Meld] = []
        self.n_flowers: int = 0
        self.history: List[list] = []
        self.played_cards: List[str] = []


class GameState:
    _eval_glb = {**globals(), **ZH2EN}

    def __init__(self):
        self.my_hand: List[str] = []
        self.my_pid: int = -1
        self.history: List[list] = []
        self.players: List[Player] = []
        self.callback = None
        self.debug_msgs = []

    def log(self, msg):
        self.debug_msgs.append(str(msg))

    @contextmanager
    def save_peng_chi_state(self):
        saved_hand = self.my_hand[:]
        saved_melds = self.players[self.my_pid].melds[:]
        try:
            yield None
        finally:
            self.my_hand = saved_hand
            self.players[self.my_pid].melds = saved_melds

    def reset(self, callback=None):
        self.my_hand = []
        self.my_pid = -1
        self.history = []
        self.players = [Player(i) for i in range(4)]
        self.callback = callback

    def load_replay(self, replay_lines, callback=None):
        if isinstance(replay_lines, str):
            replay_lines = replay_lines.split('\n')
        assert replay_lines[0].strip()[-4:] == ".xml"
        replay_lines = [[eval(tk, self._eval_glb) for tk in rl.strip().split()]
                        for rl in replay_lines[1:]]

        self.reset(callback)

        if replay_lines[0][-1] == HU:
            self.my_pid = replay_lines[-1][0]
        else:
            self.my_pid = 0

        for i, p in enumerate(self.players):
            p.n_flowers = replay_lines[i + 1][-1]
        self.my_hand = replay_lines[self.my_pid + 1][1]

        records: list = replay_lines[5:]
        idx, n_recs = 0, len(records)
        stop = False
        while idx < n_recs and not stop:
            pid, act, cards, *rest = records[idx]
            hidden_card = None
            if act == PENG:
                idx += 1
                card1 = records[idx][2][0]
                req = 3, pid, PENG, card1
            elif act == CHI:
                idx += 1
                card1 = records[idx][2][0]
                req = 3, pid, CHI, cards[1], card1
            elif act == MINGGANG:
                req = 3, pid, GANG
            elif act == ANGANG:
                req = 3, pid, GANG
                hidden_card = cards[0]
            elif act == DRAW:
                if cards[0][0] == 'H' and records[idx + 1][1] == BUHUA:
                    idx += 1
                    req = 3, pid, BUHUA, cards[0]
                elif pid == self.my_pid:
                    req = 2, cards[0]
                else:
                    req = 3, pid, DRAW
            elif act == BUHUA:
                req = 3, pid, BUHUA, cards[0]
            elif act == BUGANG:
                req = 3, pid, BUGANG, cards[0]
            elif act == PLAY:
                req = 3, pid, PLAY, cards[0]
            elif act == HU:
                req = 3, pid, HU
            else:
                assert False, records[idx]
            stop = self._handle_turn(req, hidden_card)
            idx += 1

    def _handle_play(self, pid, args, hidden_card):
        cur_p = self.players[pid]
        act = args[0]
        # 上回合打出的牌
        card0, pid0 = None, None
        if len(self.history) > 0:
            prev_turn = self.history[-1]
            pid0 = prev_turn[0]
            if len(prev_turn) > 2:
                card0 = prev_turn[-1]
        # 该回合打出的牌
        card1 = args[1] if len(args) > 1 else None
        card2 = args[2] if len(args) > 2 else None

        def play(card: str = None, meld: Meld = None):
            # 每次自己的操作前，都会调用callback（参数为当前的GameState，以及己方即将执行的操作）
            # 方便使用replay来进行训练
            if self.callback is not None:
                self.callback(self, pid, args)
            self.history.append([pid, *args])
            cur_p.history.append(args)
            if card is not None:
                cur_p.played_cards.append(card)
                if pid == self.my_pid:
                    self.my_hand.remove(card)
            if meld is not None:
                cur_p.melds.append(meld)
                if pid == self.my_pid:
                    [self.my_hand.remove(c) for c in meld.cards_from_self()]

        stop = False

        if act == BUHUA:  # 补花
            cur_p.n_flowers += 1
            self.history.append([pid, *args])
        elif act == DRAW:  # 抽牌
            self.history.append([pid, *args])
        elif act == PLAY:  # 出牌
            play(card1)
        elif act == PENG:  # 碰
            play(card1, Meld(PENG, [card0] * 3, pid0, card0))
        elif act == CHI:  # 吃
            play(card2, Meld(CHI, make_card_seq(card1), pid0, card0))
        elif act == GANG:  # 杠
            if card0:  # 直杠
                meld = Meld(GANG, [card0] * 4, pid0, card0)
            elif pid == self.my_pid:  # 己方暗杠
                meld = Meld(GANG, [hidden_card] * 4, self.my_pid, hidden_card)
            else:  # 其他玩家暗杠
                meld = Meld(GANG, [], None, None)
            play(meld=meld)
        elif act == BUGANG:  # 补杠
            play(card1)
            peng = next(m for m in cur_p.melds if
                        m.type == PENG and m.cards[0] == card1)
            peng.type = GANG
            peng.cards.append(card1)
        elif act == HU:
            play()
            stop = True

        return stop

    def _handle_turn(self, req, hidden_card=None):
        code = int(req[0])
        args = req[1:]

        stop = False

        if code == 0:  # 开局
            self.my_pid = int(args[0])
        elif code == 1:  # 发牌
            for p, n in zip(self.players, args[:4]):
                p.n_flowers = int(n)
            self.my_hand = args[4:4 + 13]
        elif code == 2:  # 己方抽卡
            self.my_hand.append(args[0])
            self.history.append([self.my_pid, DRAW])
        elif code == 3:  # 行牌
            stop = self._handle_play(int(args[0]), args[1:], hidden_card)

        return stop

    def load_json(self, input_json, callback=None):
        if isinstance(input_json, str):
            input_json = json.loads(input_json)

        self.reset(callback)

        requests = [r.split() for r in input_json["requests"]]
        responses = [None] + [r.split() for r in input_json["responses"]]
        for rq, rs in zip(requests, responses):
            stop = self._handle_turn(
                rq, rs[-1] if rs is not None and rs[0] == GANG else None)
            if stop:
                break

    def calculate_fan(self, win_tile, is_ZIMO=False, is_JUEZHANG=False,
                      is_GANG=False, is_last=False, quan_feng=0, hand=None):
        pack = tuple(
            m.to_pack(self.my_pid) for m in self.players[self.my_pid].melds)
        if hand is None:
            hand = self.my_hand
        hand = tuple(hand)
        flower_cnt = self.players[self.my_pid].n_flowers
        men_feng = self.my_pid
        try:
            res = MahjongFanCalculator(pack, hand, win_tile, flower_cnt,
                                       is_ZIMO, is_JUEZHANG, is_GANG,
                                       is_last, men_feng, quan_feng)
        except Exception as err:
            self.log(err)
            self.log(f"{pack}, {hand}, {win_tile}")
            return 0
        return sum(v for (v, d) in res)

    def action_space(self):
        if len(self.history) <= 0:
            return [(PASS,)]

        prev_turn = self.history[-1]
        pid0 = prev_turn[0]
        act0 = prev_turn[1]
        space = []

        def try_to_hu(n_f):
            if n_f >= 8:
                self.log(f"n_fan: {n_f}")
                space.append((HU, n_f))

        if act0 == DRAW and pid0 == self.my_pid:
            # 直接出牌
            space.extend((PLAY, c) for c in self.my_hand)
            # 补杠
            space.extend(
                (BUGANG, m.cards[0]) for m in self.players[self.my_pid].melds
                if m.type == PENG and m.cards[0] in self.my_hand)
            # 暗杠
            space.extend(
                (GANG, c) for c in set(self.my_hand) if
                self.my_hand.count(c) >= 4)
            # 自摸胡牌
            n_fan = self.calculate_fan(
                self.my_hand[-1], is_ZIMO=True, hand=self.my_hand[:-1])
            try_to_hu(n_fan)
        elif act0 in (PLAY, PENG, CHI) and pid0 != self.my_pid:
            card0 = prev_turn[-1]
            if card0 in self.my_hand:
                # 碰
                space.extend(peng_space(self.my_hand, card0))
                # 直杠
                if self.my_hand.count(card0) >= 3:
                    space.append((GANG,))
            if (pid0 + 1) % 4 == self.my_pid:
                # 吃
                space.extend(chi_space(self.my_hand, card0))
            # 点炮胡牌
            n_fan = self.calculate_fan(card0)
            try_to_hu(n_fan)
            # 过
            space.append((PASS,))
        else:
            space.append((PASS,))
        return space

    def encode(self):
        my_pid = self.my_pid
        players = self.players
        encoded_state = np.concatenate((
            np.array([tiles_to_plane(self.my_hand)], dtype="int8"),
            *(np.array([tiles_to_plane(players[(my_pid + i) % 4].played_cards)], dtype="int8")
              for i in range(4)),
            *(np.array([melds_to_plane(players[(my_pid + i) % 4].melds)], dtype="int8")
              for i in range(4)),
        ))
        return encoded_state
