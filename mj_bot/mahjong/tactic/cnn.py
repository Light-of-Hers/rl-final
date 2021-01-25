import random

import numpy as np
import tensorflow.keras
from keras.layers import Conv2D, BatchNormalization, \
    Dense, Dropout, Activation, Flatten
from keras.models import Sequential

from ..common import *


def cnn_model(action, train=False):
    model = Sequential()

    # （1个手牌，4个出的牌，4个吃碰杠的牌）×过去7手
    model.add(Conv2D(100, (5, 2), input_shape=(34, 4, 63)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Conv2D(100, (5, 2)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Conv2D(100, (5, 2)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(300))
    model.add(Activation('relu'))

    if action == "play":
        model.add(Dense(34))  # 判断出什么牌
    elif action == "peng":
        model.add(Dense(2))  # 判断碰还是不碰
    elif action == "chi":
        model.add(Dense(4))  # 判断不吃，吃左边、中间、右边
    else:
        assert False
    model.add(Activation('softmax'))
    # model.summary()  # 输出网络结构信息

    if train:
        opt = tensorflow.keras.optimizers.Adam(lr=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=opt,
                      metrics=['accuracy'])

    return model


class CNNTactic:
    _weight_path = {
        "play": "data/play_model_weight2.0.hdf5",
        "chi": "data/chi_model_weight2.0.hdf5",
        "peng": "data/peng_model_weight2.0.hdf5",
    }

    def __init__(self):
        self._history = [np.zeros((9, 34, 4), dtype="int8") for _ in range(7)]
        self._other_played = False
        self._model = dict()

    def _push_state(self, game_state: GameState):
        self._history.append(game_state.encode())
        self._history.pop(0)
        assert len(self._history) == 7
        return self._get_cur_data()

    def _get_cur_data(self):
        data = np.concatenate(self._history)
        data = np.swapaxes(data, 0, 2)
        data = np.swapaxes(data, 0, 1)
        return np.array([data])

    def _load_model(self, action):
        if action not in self._model:
            model = cnn_model(action)
            model.load_weights(self._weight_path[action])
            self._model[action] = model
        return self._model[action]

    def _predict(self, action, input_data):
        model = self._load_model(action)
        res = model.predict(input_data)
        res = sorted(list(enumerate(res[0])), key=lambda t: t[1], reverse=True)
        res = [i for (i, _) in res]
        return res

    def loading(self, game_state: GameState, pid: int, action: tuple):
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
                    self._push_state(game_state)

        if pid == my_pid:

            if action[0] == PLAY:
                self._push_state(game_state)

            elif action[0] == PENG:
                src_act = game_state.history[-1][1]
                if src_act not in (PLAY, PENG, CHI):
                    return

                self._push_state(game_state)

                # 碰后出牌
                src_pid = game_state.history[-1][0]
                src_card = game_state.history[-1][-1]
                meld = Meld(PENG, [src_card] * 3, src_pid, src_card)
                with game_state.save_peng_chi_state():
                    [game_state.my_hand.remove(c)
                        for c in meld.cards_from_self()]
                    game_state.players[my_pid].melds.append(meld)
                    self._push_state(game_state)

            elif action[0] == CHI:
                src_act = game_state.history[-1][1]
                if src_act not in (PLAY, PENG, CHI):
                    return

                self._push_state(game_state)
                src_pid = game_state.history[-1][0]
                src_card = game_state.history[-1][-1]
                mid_card = action[1]

                # 吃后出牌
                meld = Meld(CHI, make_card_seq(mid_card), src_pid, src_card)
                with game_state.save_peng_chi_state():
                    [game_state.my_hand.remove(c)
                     for c in meld.cards_from_self()]
                    game_state.players[my_pid].melds.append(meld)
                    self._push_state(game_state)

        elif action[0] in (PLAY, PENG, CHI):
            self._other_played = True

    def __call__(self, game_state: GameState):
        my_pid = game_state.my_pid

        space = game_state.action_space()

        if next((act for act in space if act[0] == HU), None) is not None:
            return HU,

        for act in (act for act in space if act[0] == GANG and len(act) > 1):
            card = act[1]
            card_n = card_number(card)
            if card_n is not None:
                okay = True
                for shifts in [(-2, -1), (-1, 1), (1, 2)]:
                    if all(next_card(card, s) in game_state.my_hand for s in shifts):
                        okay = False
                        break
                if okay:
                    return act
            else:
                return act

        if next((act for act in space if act[0] == PLAY), None) is not None:
            input_data = self._get_cur_data()
            for i, code in enumerate(self._predict("play", input_data)):
                tile = decode_tile(code)
                act = PLAY, tile
                if act in space:
                    game_state.log(f"paly-predicting valid at iter {i}: {act}")
                    return act

        if next((act for act in space if act[0] == CHI), None) is not None:
            input_data = self._get_cur_data()
            for i, rel_pos in enumerate(self._predict("chi", input_data)):
                if rel_pos == 0:
                    break

                src_pid = game_state.history[-1][0]
                src_card = game_state.history[-1][-1]
                mid_card_n = card_number(src_card) - rel_pos + 2
                mid_card = card_type(src_card) + str(mid_card_n)

                if next((act for act in space if act[0] == CHI
                         and act[1] == mid_card), None) is None:
                    continue

                meld = Meld(CHI, make_card_seq(mid_card), src_pid, src_card)
                with game_state.save_peng_chi_state():
                    [game_state.my_hand.remove(c)
                     for c in meld.cards_from_self()]
                    game_state.players[my_pid].melds.append(meld)
                    input_data = self._push_state(game_state)

                for j, code in enumerate(self._predict("play", input_data)):
                    tile = decode_tile(code)
                    act = CHI, mid_card, tile
                    if act in space:
                        game_state.log(
                            f"chi-predicting valid at iter ({i}, {j}): {act}")
                        return act

        if next((act for act in space if act[0] == PENG), None) is not None:
            input_data = self._get_cur_data()
            for i, do_peng in enumerate(self._predict("peng", input_data)):
                if do_peng == 0:
                    break

                src_pid = game_state.history[-1][0]
                src_card = game_state.history[-1][-1]
                meld = Meld(PENG, [src_card] * 3, src_pid, src_card)
                with game_state.save_peng_chi_state():
                    [game_state.my_hand.remove(c)
                        for c in meld.cards_from_self()]
                    game_state.players[my_pid].melds.append(meld)
                    input_data = self._push_state(game_state)

                for j, code in enumerate(self._predict("play", input_data)):
                    tile = decode_tile(code)
                    act = PENG, tile
                    if act in space:
                        game_state.log(
                            f"peng-predicting valid at iter ({i}, {j}): {act}")
                        return act

        return random.choice(space)
