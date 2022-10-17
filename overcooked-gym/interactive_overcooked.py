# import pygame module in this program
import pygame
from overcooked2 import Overcooked, SingleAgentWrapper, LAYOUTS
from yaaf.agents import HumanAgent
from teammates.JackOfAllTrades import JackOfAllTrades
from teammates.Astro import AstroHandcoded, AstroSmart, JOINT_ACTION_SPACE
import numpy as np
import time
from PIL import Image
import glob
import pickle

fileCounter = len(glob.glob1("/home/anavc/Overcooked_Gym/overcooked-gym/","logfile_AstroHuman_*"))

log = []
log_file = f"logfile_AstroHuman_{fileCounter}.pickle"
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

layout = "Lab2"
env = Overcooked(layout=layout)
#agent = HumanAgent(action_meanings=env.action_meanings, name="Player 1")  # 1 - selects robot; 0 - selects human
teammate = AstroSmart(LAYOUTS[layout], 1, env=env)
#teammate = AstroHandcoded(LAYOUTS[layout], 1, env=env)
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
text = font.render('Time: ' + str(time.time()), True, (255,173,1))
text2 = font.render('Time with ball: ' + str(time.time()), True, (255,173,1))
textRect = text.get_rect()
textRect2 = text.get_rect()
textRect3 = text.get_rect()
textRect.center = (int(X*(0.40)), int(Y*(0.965)))
textRect2.center = (int(X*(0.75)), int(Y*(0.965)))
game_time = 0
orig_surf = font.render("*slip*", True, (0,0,255))
txt_list = []
t = 0
slipped = False

class LogFrame:
  def __init__(self, timestep:int , state_env, state_mdp:list, action_env:tuple, action_mdp:tuple, onion_time:float, game_time:float):
    self.timestep = timestep
    self.state_env = state_env
    self.state_mdp = state_mdp
    self.action_env = action_env
    self.action_mdp = action_mdp
    self.onion_time = onion_time
    self.game_time = game_time

def fade_in_text(txt_list):
    for txt in txt_list:
        display_surface.blit(txt[0], txt[1])
        if txt[0].get_alpha() <=0:
            txt_list.remove(txt)
        else: txt[0].set_alpha(txt[0].get_alpha()-50)  
        
# infinite loop
start_game = time.time()
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
    for event in pygame.event.get()[-1:]:
        pygame.draw.rect(display_surface, GREY, left_wall)
        pygame.draw.rect(display_surface, GREY, right_wall)
        pygame.draw.rect(display_surface, GREY, up_wall_1)
        pygame.draw.rect(display_surface, GREY, up_wall_2)
        pygame.draw.rect(display_surface, GREY, down_wall)

        # if event object type is QUIT
        # then quitting the pygame
        # and program both.
        if event.type == pygame.QUIT:
            # deactivates the pygame library
            pygame.quit()

            # quit the program.
            quit()

        if event.type == new_frame or action == None:
            #action = agent.action(state)
            print("Time passed")
            action = 5

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
            
        state, _, terminal, _ = env.step(action)
        
        #LOG: timestep, environment state, mdp state, environment action, mdp_action, onion_time, game_time
        game_time = int(time.time()-start_game)
        timestep_log = LogFrame(t, state, teammate.last_state, env.unpack_joint_action(env.j_a), teammate.last_action, teammate.onion_time, game_time)
        log.append(timestep_log)
        t += 1

        textRect3.center = (state[3]*15, state[2]*15)
        
        frame = np.rot90(env.render(render_mode))
        #frame = np.array(Image.fromarray(frame).resize(size=(Y_resize, X_resize)))
        image = pygame.surfarray.make_surface(frame)
        image = pygame.transform.scale(image, (X_resize, Y_resize))

        # Draws the surface object to the screen.
        text = font.render('Time: ' + str(round(time.time()-start_game, 1)), True,(255,173,1) )
        text2 = font.render('Time with ball: ' + str(round(teammate.onion_time)), True, (255,173,1))
        display_surface.blit(text, textRect)
        display_surface.blit(text2, textRect2)

        if state[8] == 1:
            pos = (state[3]*(X/15), (state[2]-1)*(Y/15))
            txt_surf = orig_surf.copy()
            txt_surf.set_alpha(200)
            txt_list.append([txt_surf,pos])
        fade_in_text(txt_list)

        pygame.display.update()



with open(log_file, "wb") as a:
    pickle.dump(log, a)

print("TIMESTEPS: ", t)
print("Game time: ", game_time)
print("Time with onion in hand: ", round(teammate.onion_time, 1))
print("Final Score: ", 100 - game_time - round(teammate.onion_time))