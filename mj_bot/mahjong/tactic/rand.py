from ..common import *
import random


def random_tactic(game_state: GameState):
    space = game_state.action_space()
    if next((act for act in space if act[0] == HU), None) is not None:
        return (HU,)
    return random.choice(space)
