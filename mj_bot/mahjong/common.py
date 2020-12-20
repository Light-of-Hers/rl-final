from typing import List
import json

from MahjongGB import MahjongFanCalculator

BUHUA = "BUHUA"
DRAW = "DRAW"
PLAY = "PLAY"
PENG = "PENG"
CHI = "CHI"
GANG = "GANG"
BUGANG = "BUGANG"
PASS = "PASS"
HU = "HU"


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
        self.played: List[str] = []


class GameState:
    def __init__(self):
        self.my_hand: List[str] = []
        self.my_pid: int = -1
        self.history: List[list] = []
        self.players: List[Player] = []
        self.debug_msgs = []

    def log(self, msg):
        self.debug_msgs.append(str(msg))

    def load_json(self, input_json, callback=None):
        if isinstance(input_json, str):
            input_json = json.loads(input_json)

        self.my_hand = []
        self.my_pid = -1
        self.history = []
        self.players = [Player(i) for i in range(4)]

        def handle_play(pid, args, res):
            players = self.players
            history = self.history
            my_pid = self.my_pid
            my_hand = self.my_hand

            cur_p = players[pid]
            act = args[0]
            # 上回合打出的牌
            card0, pid0 = None, None
            if len(history) > 0:
                prev_turn = history[-1]
                pid0 = prev_turn[0]
                if len(prev_turn) > 2:
                    card0 = prev_turn[-1]
            # 该回合打出的牌
            card1 = args[1] if len(args) > 1 else None
            card2 = args[2] if len(args) > 2 else None

            def play(card: str = None, meld: Meld = None):
                # 每次自己的操作前，都会调用callback（参数为当前的GameState，以及己方即将执行的操作）
                # 方便使用replay来进行训练
                if pid == my_pid and callback is not None:
                    callback(self, args)
                history.append([pid, *args])
                cur_p.history.append(args)
                if card is not None:
                    cur_p.played.append(card)
                    if pid == my_pid:
                        my_hand.remove(card)
                if meld is not None:
                    cur_p.melds.append(meld)
                    if pid == my_pid:
                        [my_hand.remove(c) for c in meld.cards_from_self()]

            if act == BUHUA:  # 补花
                cur_p.n_flowers += 1
                history.append([pid, *args])
            elif act == DRAW:  # 抽牌
                history.append([pid, *args])
            elif act == PLAY:  # 出牌
                play(card1)
            elif act == PENG:  # 碰
                play(card1, Meld(PENG, [card0] * 3, pid0, card0))
            elif act == CHI:  # 吃
                play(card2, Meld(CHI, make_card_seq(card1), pid0, card0))
            elif act == GANG:  # 杠
                if card0:  # 直杠
                    meld = Meld(GANG, [card0] * 4, pid0, card0)
                elif pid == my_pid:  # 己方暗杠
                    meld = Meld(GANG, [res[-1]] * 4, None, None)
                else:  # 其他玩家暗杠
                    meld = Meld(GANG, None, None, None)
                play(meld=meld)
            elif act == BUGANG:  # 补杠
                play(card1)
                peng = next(m for m in cur_p.melds if
                            m.type == PENG and m.cards[0] == card1)
                peng.type = GANG
                peng.cards.append(card1)

        def handle_turn(req, res):
            code = int(req[0])
            args = req[1:]

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
                handle_play(int(args[0]), args[1:], res)

        requests = [r.split() for r in input_json["requests"]]
        responses = [None] + [r.split() for r in input_json["responses"]]
        [handle_turn(rq, rs) for (rq, rs) in zip(requests, responses)]

    def calculate_fan(self, win_tile, is_ZIMO, is_JUEZHANG=False,
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
                self.my_hand[-1], True, hand=self.my_hand[:-1])
            if n_fan >= 8:
                self.log(f"n_fan: {n_fan}")
                space.append((HU, n_fan))

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
            n_fan = self.calculate_fan(
                card0, False, hand=self.my_hand)
            if n_fan >= 8:
                self.log(f"n_fan: {n_fan}")
                space.append((HU, n_fan))
            # 过
            space.append((PASS,))
        else:
            space.append((PASS,))
        return space
