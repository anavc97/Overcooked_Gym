# import pygame module in this program
import pygame
from overcooked2 import Overcooked, SingleAgentWrapper, LAYOUTS
from yaaf.agents import HumanAgent
from teammates.JackOfAllTrades import JackOfAllTrades
from teammates.Astro import AstroHandcoded, AstroSmart, JOINT_ACTION_SPACE
import numpy as np
import time
from PIL import Image

# activate the pygame library .
# initiate pygame and give permission
# to use pygame's functionality.
pygame.init()
MPF = 100
# define the RGB value
# for white colour
white = (255, 255, 255)
start_game = time.time()

# assigning values to X and Y variable
X = 900
Y = 700
X_resize = 900
Y_resize = 700

left_wall = pygame.Rect(0, 0, 55, Y_resize)
right_wall = pygame.Rect(X_resize-55, 0, 55, Y_resize)
down_wall = pygame.Rect(0, Y_resize-55, X_resize, 55)
up_wall = pygame.Rect(0, 0, X_resize, 47)
# create the display surface object
# of specific dimension..e(X, Y).
# display_surface = pygame.display.set_mode((X, Y))
display_surface = pygame.display.set_mode((X, Y), pygame.RESIZABLE)
# set the pygame window name
pygame.display.set_caption('Overcooked')
single_agent = False
render = True
render_mode = "silent"  # Available: window (pop-up) and matplotlib (plt.imshow). Video rendering planned for the future.

layout = "Lab"
env = Overcooked(layout=layout)
#agent = HumanAgent(action_meanings=env.action_meanings, name="Player 1")  # 1 - selects robot; 0 - selects human
teammate = AstroSmart(LAYOUTS[layout], 1, env=env)
env = SingleAgentWrapper(env, teammate)
state = env.reset()
frame = np.rot90(env.render(render_mode))
#frame = imresize(frame, [500, 500], 'bilinear')
image = pygame.surfarray.make_surface(frame)
terminal = False
new_frame = pygame.USEREVENT + 1
pygame.time.set_timer(new_frame, MPF)
action = 5
font = pygame.font.SysFont('didot.ttc', 50)
text = font.render('Time: ' + str(time.time()), True, (255,255,255))
text2 = font.render('Time with ball: ' + str(time.time()), True, (255,255,255))
textRect = text.get_rect()
textRect2 = text.get_rect()
textRect.center = (int(X*(0.27)), int(Y*(0.95)))
textRect2.center = (int(X*(0.57)), int(Y*(0.95)))
game_time = 0

"""while not terminal:
    print("Blue hat goes first")
    print(f"State: {state}")
    action = agent.action(state)
    next_state, reward, terminal, info = env.step(action)
    env.render(render_mode)
    print(f"State: {next_state}")
    print(f"Reward: {reward}")"""


# infinite loop
while not terminal:
    
    action = None
    # completely fill the surface object
    # with white colour
    display_surface.fill(white)

    # copying the image surface object
    # to the display surface object at
    # (0, 0) coordinate.
    display_surface.blit(pygame.transform.flip(image, True, False), (0, 0))

    # iterate over the list of Event objects
    # that was returned by pygame.event.get() method.
    event = pygame.event.get()[-1]
    print("Action beggining : ", action)
    pygame.draw.rect(display_surface, (64,64,64), left_wall)
    pygame.draw.rect(display_surface, (64,64,64), right_wall)
    pygame.draw.rect(display_surface, (64,64,64), up_wall)
    pygame.draw.rect(display_surface, (64,64,64), down_wall)

    # if event object type is QUIT
    # then quitting the pygame
    # and program both.
    if event.type == pygame.QUIT:
        # deactivates the pygame library
        pygame.quit()

        # quit the program.
        quit()

    if event.type == pygame.KEYDOWN:
        #pygame.time.set_timer(new_frame, 0)
        #pygame.time.set_timer(new_frame, MPF)
        print("Key Pressed")
        if event.key == pygame.K_UP:
            action = 0

        if event.key == pygame.K_DOWN:
            action = 1

        if event.key == pygame.K_LEFT:
            action = 2

        if event.key == pygame.K_RIGHT:
            action = 3

        if event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
            action = 4
            
    if event.type == new_frame or action == None:
        #action = agent.action(state)
        print("Time passed")
        action = 5
        
    print("Action end: ", action)
    state, _, terminal, _ = env.step(action)
    frame = np.rot90(env.render(render_mode))
    frame = np.array(Image.fromarray(frame).resize(size=(Y_resize, X_resize)))
    image = pygame.surfarray.make_surface(frame)

    # Draws the surface object to the screen.
    game_time = int(time.time()-start_game)
    text = font.render('Time: ' + str(round(time.time()-start_game, 1)), True, (255,255,255))
    text2 = font.render('Time with ball: ' + str(round(teammate.onion_time, 1)), True, (255,255,255))
    display_surface.blit(text, textRect)
    display_surface.blit(text2, textRect2)

    pygame.display.update()
    
print("Game time: ", game_time)
print("Time with onion in hand: ", round(teammate.onion_time, 1))
print("Final Score: ", 100 - game_time - round(teammate.onion_time))