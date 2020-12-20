from ..common import *
import random


def random_tactic(game_state: GameState):
    space = game_state.action_space()
    if (HU,) in space:
        return (HU,)
    return random.choice(space)
