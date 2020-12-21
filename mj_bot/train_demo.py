from mahjong.common import GameState, Player


class Trainer:

    def __init__(self):
        pass

    def print_player(self, player: Player):
        print("player_id:", player.player_id)
        print("melds:", [m.cards for m in player.melds])
        print("n_flowers:", player.n_flowers)

    def handle_state_and_action(self, game_state: GameState, action):
        print("my_player_id:", game_state.my_pid)
        print("my_hand:", game_state.my_hand)
        [self.print_player(p) for p in game_state.players]
        print("action:", action)


def main():
    game_state = GameState()
    trainer = Trainer()

    game_state.load_replay(list(open("../data/sample1.txt", "r")),
                           callback=trainer.handle_state_and_action)


if __name__ == "__main__":
    main()
