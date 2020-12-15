from mahjong.common import GameState
from mahjong.tactic import random_tactic
import json


def main():
    tactic = random_tactic
    game_state = GameState()

    input_json = json.loads(input())
    game_state.load_json(input_json)

    action = tactic(game_state)
    if not isinstance(action, str):
        action = " ".join(str(x) for x in action)

    output_json = json.dumps({
        "response": action,
    })
    print(output_json)


if __name__ == '__main__':
    main()
