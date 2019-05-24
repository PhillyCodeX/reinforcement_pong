import src.game_objects as go
import src.strategies as strat

def main():
    want_new_players = True
    want_new_game = True
    new_game = go.Game()

    while want_new_game:
        want_new_game = False
        print('Let\'s Play Pong!!!')
        
        if want_new_players:
            want_new_players = False
            new_game.setPlayers()
            print('Players are set!')
            print('Player 1, Welcome!: '+new_game.player1.name)
            print('Player 2, Welcome!: '+new_game.player2.name)
        
        new_game.play()
        winner = new_game.winner
        
        print("Game finished!!!")
        print(winner.name + " won! Congratulations")
        
        user_input = input("Do you want to play a new game?[Yes/No] ")

        if user_input == "Yes":
            want_new_game = True

            user_input = input("Do you want to set new players?[Yes/No] ")

            if user_input == "Yes":
                want_new_players = True
            elif user_input == "No":
                new_game.player1.points = 0
                new_game.player2.points = 0
    
if __name__ == '__main__': 
    main() 