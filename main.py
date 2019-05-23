import src.game_objects as go
import src.strategies as strat
import pygame 

def main():

    print('Let\'s Play Pong!!!')
    new_game = go.Game()

    new_game.setPlayers()
    print('Players are set!')
    print('Player 1, Welcome!: '+new_game.player1.name)
    print('Player 2, Welcome!: '+new_game.player2.name)
    
    pygame.init()
    clock = pygame.time.Clock()

    resolution = (new_game.area.width, new_game.area.height)
    screen = pygame.display.set_mode(resolution)
    pygame.display.set_caption("Smart Pong")
    score_font = pygame.font.SysFont("Clear Sans Regular", 30)

    cancel = False

    while not cancel:
        pressed_down = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cancel = True
                pygame.quit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_DOWN:
                    dir_up = False
                if event.key == pygame.K_UP:
                    dir_up = True

            if event.type == pygame.KEYUP:
                dir_up = None

        new_game.player1.next_pos(dir_up)

if __name__ == '__main__': 
    main() 