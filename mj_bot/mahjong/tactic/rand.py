from ..common import *
import random


def random_tactic(game_state: GameState):
    space = game_state.action_space()
    return random.choice(space)
