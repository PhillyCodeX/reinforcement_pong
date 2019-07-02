import src.game_objects as go
import src.strategies as strat
import datetime

TRAIN_MODE = True
DELIMITER = ";"
RESUME = False

def train(p_nepisodes):
    logging_row = "timestamp;episode_no;p1_points;p2_points;p1_avg_loss;p1_sum_reward;p2_avg_loss;p2_sum_reward\n"

    with open("train.log", "a") as myfile:
            myfile.write(logging_row)

    new_game = go.Game()
    new_game.setPlayers(TRAIN_MODE, RESUME)

    for i in range(p_nepisodes):
        today = datetime.datetime.today() 
        dt=datetime.datetime.strftime(today,'%Y%m%d%H%M%S')

        logging_row = dt
        logging_row += DELIMITER
        logging_row += str(i)
        logging_row += DELIMITER
        
        new_game.play()

        loss1 = new_game.player1.strategy.avg_loss
        reward1 = new_game.player1.strategy.sum_reward
        loss2 = new_game.player2.strategy.avg_loss
        reward2 = new_game.player2.strategy.sum_reward

        logging_row += str(new_game.player1.points)
        logging_row += DELIMITER
        logging_row += str(new_game.player2.points)
        logging_row += DELIMITER
        logging_row += str(loss1)
        logging_row += DELIMITER
        logging_row += str(reward1)
        logging_row += DELIMITER
        logging_row += str(loss2)
        logging_row += DELIMITER
        logging_row += str(reward2)
        logging_row += "\n"
        
        if i % 100 == 0:
            new_game.player1.strategy.safe_state()
            new_game.player2.strategy.safe_state()

        new_game.reset()

        with open("train.log", "a") as myfile:
            myfile.write(logging_row)

    print("------FINISHED TRAINING------")

def normal():
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

def main():
    if TRAIN_MODE:
<<<<<<< HEAD
        train(2)
=======
        train(500)
>>>>>>> 1e885c1a72899cdd4e9b00a31296b8147d7a0519
    else:
        normal()
    

if __name__ == '__main__': 
    main() 