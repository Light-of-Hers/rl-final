import random
from argparse import ArgumentParser

import numpy as np

from mahjong.common import *


class DatasetGenerator:
    _scalars = [np.array(x) for x in range(34)]

    def __init__(self):
        self.play_data = []
        self.play_label = []
        self.peng_data = []
        self.peng_label = []
        self.chi_data = []
        self.chi_label = []
        self._history = [np.zeros((9, 34, 4)) for _ in range(7)]
        self._other_played = False

    def reset(self):
        self.__init__()

    def _push_state(self, game_state: GameState):
        self._history.append(game_state.encode())
        self._history.pop(0)
        assert len(self._history) == 7
        return self._get_cur_data()

    def _get_cur_data(self):
        return np.concatenate(self._history)

    def handle_state_and_action(self, game_state: GameState, pid: int, action: tuple):
        if action[0] in (DRAW, BUHUA):
            return

        my_pid = game_state.my_pid

        if self._other_played:
            self._other_played = False
            if pid != my_pid:
                space = game_state.action_space()
                can_peng = next(
                    (act for act in space if act[0] == PENG), None) is not None
                can_chi = next(
                    (act for act in space if act[0] == CHI), None) is not None
                if can_peng or can_chi:
                    cur_data = self._push_state(game_state)
                    # 仅在可以吃/碰的情况下考虑不吃/不碰
                    if can_peng:
                        self.peng_data.append(cur_data)
                        self.peng_label.append(self._scalars[0])
                    if can_chi:
                        self.chi_data.append(cur_data)
                        self.chi_label.append(self._scalars[0])

        if pid == my_pid:

            if action[0] == PLAY:
                cur_data = self._push_state(game_state)
                tile_code = encode_tile(action[-1])
                self.play_data.append(cur_data)
                self.play_label.append(self._scalars[tile_code])

            elif action[0] == PENG:
                src_act = game_state.history[-1][1]
                if src_act not in (PLAY, PENG, CHI):
                    return

                cur_data = self._push_state(game_state)
                self.peng_data.append(cur_data)
                self.peng_label.append(self._scalars[1])

                # 碰后出牌
                src_pid = game_state.history[-1][0]
                src_card = game_state.history[-1][-1]
                meld = Meld(PENG, [src_card] * 3, src_pid, src_card)
                with game_state.save_peng_chi_state():
                    [game_state.my_hand.remove(c)
                        for c in meld.cards_from_self()]
                    game_state.players[my_pid].melds.append(meld)
                    cur_data = self._push_state(game_state)
                tile_code = encode_tile(action[-1])
                self.play_data.append(cur_data)
                self.play_label.append(self._scalars[tile_code])

            elif action[0] == CHI:
                src_act = game_state.history[-1][1]
                if src_act not in (PLAY, PENG, CHI):
                    return

                cur_data = self._push_state(game_state)
                src_pid = game_state.history[-1][0]
                src_card = game_state.history[-1][-1]
                mid_card = action[1]
                rel_pos = card_number(src_card) - card_number(mid_card) + 2
                self.chi_data.append(cur_data)
                self.chi_label.append(self._scalars[rel_pos])

                # 吃后出牌
                meld = Meld(CHI, make_card_seq(mid_card), src_pid, src_card)
                with game_state.save_peng_chi_state():
                    [game_state.my_hand.remove(c)
                     for c in meld.cards_from_self()]
                    game_state.players[my_pid].melds.append(meld)
                    cur_data = self._push_state(game_state)
                tile_code = encode_tile(action[-1])
                self.play_data.append(cur_data)
                self.play_label.append(self._scalars[tile_code])

        elif action[0] in (PLAY, PENG, CHI):
            self._other_played = True


def main():
    game_state = GameState()
    generator = DatasetGenerator()
    for i in range(3):
        generator.reset()  # reset generator
        game_state.load_replay(
            list(open(f"../data/sample{i + 1}.txt")), generator.handle_state_and_action)

        print(generator.play_data)
        print(generator.play_label)
        print(generator.chi_data)
        print(generator.chi_label)
        print(generator.peng_data)
        print(generator.peng_label)


if __name__ == "__main__":
    main()
