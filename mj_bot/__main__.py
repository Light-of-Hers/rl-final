from mahjong.common import GameState
from mahjong.tactic import RandomTactic, CNNTactic, RuleBasedTactic
import json
import sys


def main():
    # tactic = CNNTactic()
    tactic = RuleBasedTactic()
    game_state = GameState()

    input_json = json.loads(input())
    game_state.load_json(
        input_json, tactic.loading if hasattr(tactic, "loading") else None)

    action = tactic(game_state)
    if not isinstance(action, str):
        assert isinstance(action, (list, tuple))
        action = " ".join(str(x) for x in action)

    game_state.log(action)

    output_json = json.dumps({
        "response": action,
        "debug": game_state.debug_msgs,
    })
    print('\n'.join(game_state.debug_msgs), file=sys.stderr)
    print(output_json)


if __name__ == '__main__':
    main()
