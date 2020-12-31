from ..common import *
import random


class RandomTactic:
    def loading(self, game_state: GameState, pid: int, action: tuple):
        pass

    def __call__(self, game_state: GameState):
        space = game_state.action_space()
        if next((act for act in space if act[0] == HU), None) is not None:
            return (HU,)
        return random.choice(space)
