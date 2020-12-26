import random

import numpy as np

from mahjong.common import GameState, Player, HU


def sl_tactic(game_state: GameState):
    space = game_state.action_space()
    if next((act for act in space if act[0] == HU), None) is not None:
        return HU,
    return random.choice(space)


# 将手牌转换成34×4的二维数组
def tiles_to_plane(hand):
    plane = np.zeros(136, dtype=int).reshape(-1, 4)
    for tile in hand:
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
            assert False
        for i in range(4):
            if plane[row][i] == 0:
                plane[row][i] = 1
                break
    return plane


def melds_to_plane(player):
    # 将所有吃、碰、杠的牌展开成一维
    return tiles_to_plane([tile for meld in player.melds for tile in meld.cards])


def print_player(player: Player):
    print("player_id:", player.player_id)
    print("melds:", [m.cards for m in player.melds])
    print("n_flowers:", player.n_flowers)


def handle_state_and_action(game_state: GameState, action):
    print("my_player_id:", game_state.my_pid)
    print("my_hand:", game_state.my_hand)
    # "我"的手牌
    hand_plane = np.array([tiles_to_plane(game_state.my_hand)])
    # TODO 四名玩家之前出过的所有牌，从"我"开始

    # 四名玩家的吃、碰、杠，从"我"开始
    my_meld_plane = np.array([melds_to_plane(p)
                              for p in game_state.players
                              if p.player_id == game_state.my_pid])
    next_meld_plane = np.array([melds_to_plane(p)
                                for p in game_state.players
                                if p.player_id == (game_state.my_pid + 1) % 4])
    opposite_meld_plane = np.array([melds_to_plane(p)
                                    for p in game_state.players
                                    if p.player_id == (game_state.my_pid + 2) % 4])
    preceding_meld_plane = np.array([melds_to_plane(p)
                                     for p in game_state.players
                                     if p.player_id == (game_state.my_pid + 3) % 4])

    # TODO 四名玩家之前六轮出过的牌、之前六轮吃、碰、杠的牌

    plane = np.concatenate((hand_plane, my_meld_plane, next_meld_plane,
                            opposite_meld_plane, preceding_meld_plane))
    print("plane", plane)
    print("action:", action)


def main():
    game_state = GameState()

    game_state.load_replay(
        list(open("../../../Documents/rl-final/data/sample1.txt", "r")),
        callback=handle_state_and_action
    )


if __name__ == "__main__":
    main()
