# import pygame module in this program
import pygame
from overcooked2 import Overcooked, SingleAgentWrapper, LAYOUTS
from yaaf.agents import HumanAgent
from teammates.JackOfAllTrades import JackOfAllTrades
from teammates.Astro import AstroHandcoded, AstroSmart, AstroFake, ACTION_MEANINGS_MDP, log_file, LVL
import numpy as np
import itertools
import time
from PIL import Image
import glob
import pickle
import copy


log = []
ACTION_SPACE = tuple(range(len(ACTION_MEANINGS_MDP[LVL-1])))
JOINT_ACTION_SPACE = list(itertools.product(ACTION_SPACE, repeat=2))
print("READING FROM: ", log_file)

class LogFrame:
  def __init__(self, timestep:int , state_env, state_mdp:list, action_env:tuple, action_mdp:tuple, onion_time:float, game_time:float):
    self.timestep = timestep
    self.state_env = state_env
    self.state_mdp = state_mdp
    self.action_env = action_env
    self.action_mdp = action_mdp
    self.onion_time = onion_time
    self.game_time = game_time


def get_font(size): # Returns Press-Start-2P in the desired size
    return pygame.font.Font('/home/anavc/Overcooked_Gym/overcooked-gym/overcooked_ai_py/data/fonts/PublicPixel-0W5Kv.ttf', size)


# activate the pygame library .
# initiate pygame and give permission
# to use pygame's functionality.
pygame.init()
MPF = 100
# define the RGB value
# for white colour
white = (255, 255, 255)
GREY = (100,100,100)

# assigning values to X and Y variable
X = 900
Y = 700
X_resize = 900
Y_resize = 700

left_wall = pygame.Rect(0, 47, 60, Y_resize)
right_wall = pygame.Rect(X_resize-60, 47, 60, Y_resize)
down_wall = pygame.Rect(0, Y_resize-47, X_resize, Y_resize)
up_wall_1 = pygame.Rect(60, 0, 360, 47)
up_wall_2 = pygame.Rect(480, 0, 360, 47)
# create the display surface object
# of specific dimension..e(X, Y).
# display_surface = pygame.display.set_mode((X, Y))
display_surface = pygame.display.set_mode((X, Y), pygame.RESIZABLE)
# set the pygame window name
pygame.display.set_caption('Overcooked')
single_agent = False
render = True
render_mode = "silent"  # Available: window (pop-up) and matplotlib (plt.imshow). Video rendering planned for the future.

layout = f"Lab{LVL}"
env = Overcooked(layout=layout)
#agent = HumanAgent(action_meanings=env.action_meanings, name="Player 1")  # 1 - selects robot; 0 - selects human
teammate = AstroFake(layout, 1, env=env)
env = SingleAgentWrapper(env, teammate)
state = env.reset()
frame = np.rot90(env.render(render_mode))
#frame = imresize(frame, [500, 500], 'bilinear')
image = pygame.surfarray.make_surface(frame)
terminal = False
new_frame = pygame.USEREVENT + 1
pygame.time.set_timer(new_frame, MPF)
action = 5
font = pygame.font.Font('/home/anavc/Overcooked_Gym/overcooked-gym/overcooked_ai_py/data/fonts/PublicPixel-0W5Kv.ttf', 25)
text = get_font(25).render('Time: ' + str(time.time()), True, (255,173,1))
text2 = get_font(25).render('Time with ball: ' + str(time.time()), True, (255,173,1))
textRect = text.get_rect()
textRect2 = text.get_rect()
textRect3 = text.get_rect()
textRect.center = (int(X*(0.40)), int(Y*(0.965)))
textRect2.center = (int(X*(0.75)), int(Y*(0.965)))
game_time = 0
orig_surf = get_font(25).render("???", True, (20,255,20))
txt_list = []
slipped = False

def fade_in_text(txt_list):
    for txt in txt_list:
        display_surface.blit(txt[0], txt[1])
        if txt[0].get_alpha() <=0:
            txt_list.remove(txt)
        else: txt[0].set_alpha(txt[0].get_alpha()-50)

# infinite loop

with open(log_file, "rb") as f:
    log = pickle.load(f)

print(isinstance(log[0], LogFrame))

start_game = time.time()
for logframe in log:
    
    state = logframe.state_env
    a_joint = logframe.action_env
    state_mdp = logframe.state_mdp
    action = a_joint[0]
    action_r = a_joint[1]
    print("state: ", state_mdp, " action: ", action_r)
    # completely fill the surface object
    # with white colour
    display_surface.fill(white)
    
    # copying the image surface object
    # to the display surface object at
    # (0, 0) coordinate.
    display_surface.blit(pygame.transform.flip(image, True, False), (0, 0))

    # iterate over the list of Event objects
    # that was returned by pygame.event.get() method.
    pygame.draw.rect(display_surface, GREY, left_wall)
    pygame.draw.rect(display_surface, GREY, right_wall)
    pygame.draw.rect(display_surface, GREY, up_wall_1)
    pygame.draw.rect(display_surface, GREY, up_wall_2)
    pygame.draw.rect(display_surface, GREY, down_wall)
    
    #state, _, _, _ = env.step(action)
    textRect3.center = (state[3]*15, state[2]*15)
    
    frame = np.rot90(env.render_log(log_file,render_mode, logframe.timestep))
    frame = np.array(Image.fromarray(frame).resize(size=(Y_resize, X_resize)))
    image = pygame.surfarray.make_surface(frame)

    # Draws the surface object to the screen.
    game_time = logframe.game_time
    text = get_font(25).render('Time: ' + str(game_time), True,(255,173,1) )
    text2 = get_font(25).render('Time with ball: ' + str(round(logframe.onion_time, 1)), True, (255,173,1))
    display_surface.blit(text, textRect)
    display_surface.blit(text2, textRect2)

#str(round(time.time()-start_game, 1))
    if state[8] == 1:
        pos = (state[3]*(X/15), (state[2]-1)*(Y/15))
        txt_surf = orig_surf.copy()
        txt_surf.set_alpha(200)
        txt_list.append([txt_surf,pos])
    fade_in_text(txt_list)

    pygame.display.update()



print("Game time: ", game_time)
print("Time with onion in hand: ", round(logframe.onion_time, 1))
print("Final Score: ", 100 - game_time - round(logframe.onion_time))