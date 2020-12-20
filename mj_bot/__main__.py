from mahjong.common import GameState
from mahjong.tactic import random_tactic
import json
import sys


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
        "debug": game_state.debug_msgs,
    })
    print('\n'.join(game_state.debug_msgs), file=sys.stderr)
    print(output_json)


if __name__ == '__main__':
    main()
