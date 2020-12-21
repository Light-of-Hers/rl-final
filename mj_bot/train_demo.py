from mahjong.common import GameState


class Trainer:

    def __init__(self):
        pass

    def process_state_and_action(self, game_state, action):
        print(action)


def main():
    game_state = GameState()
    trainer = Trainer()

    game_state.load_replay(list(open("../data/sample1.txt", "r")),
                           callback=trainer.process_state_and_action)


if __name__ == "__main__":
    main()
