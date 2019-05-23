import src.game_objects as go
import src.strategies as strat

def main():

    print('Let\'s Play Pong!!!')
    new_game = go.Game()

    new_game.setPlayers()
    print('Players are set!')
    print('Player 1, Welcome!: '+new_game.player1.name)
    print('Player 2, Welcome!: '+new_game.player2.name)
    
    new_game.play()

if __name__ == '__main__': 
    main() 