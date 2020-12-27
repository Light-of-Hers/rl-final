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

    def handle_state_and_action(self, game_state: GameState, pid: int, action: tuple):
        if action[0] in (DRAW, BUHUA):
            return

        my_pid = game_state.my_pid

        cur_state = game_state.encode()
        self._history.append(cur_state)
        self._history.pop(0)
        cur_data = np.concatenate(self._history)

        if self._other_played:
            self._other_played = False
            if pid != my_pid:
                self.peng_data.append(cur_data)
                self.peng_label.append(self._scalars[0])
                if (pid + 1) % 4 == my_pid:
                    self.chi_data.append(cur_data)
                    self.chi_label.append(self._scalars[0])

        if pid == my_pid:
            if action[0] == PLAY:
                tile_code = encode_tile(action[-1])
                self.play_data.append(cur_data)
                self.play_label.append(self._scalars[tile_code])
            elif action[0] == PENG:
                self.peng_data.append(cur_data)
                self.peng_label.append(self._scalars[1])
            elif action[0] == CHI:
                src_card = game_state.history[-1][-1]
                mid_card = action[1]
                rel_pos = card_number(src_card) - card_number(mid_card) + 2
                self.chi_data.append(cur_data)
                self.chi_label.append(self._scalars[rel_pos])

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
