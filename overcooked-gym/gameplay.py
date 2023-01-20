# import pygame module in this program
import pygame, sys
from overcooked2 import Overcooked, SingleAgentWrapper
from yaaf.agents import HumanAgent
from teammates.JackOfAllTrades import JackOfAllTrades
from teammates.Astro import AstroHandcoded, AstroSmart
import numpy as np
import time
from PIL import Image
import glob
import pickle
from overcooked_ai_py.visualization.button import Button

pygame.init()
    
MPF = 100
GAME_OVER = False
COMPLETION_CODE = 12345
# assigning values to X and Y variable
X = 900
Y = 700
X_resize = 900
Y_resize = 700

def get_font(size): # Returns Press-Start-2P in the desired size
    return pygame.font.Font('/home/anavc/Overcooked_Gym/overcooked-gym/overcooked_ai_py/data/fonts/PublicPixel-0W5Kv.ttf', size)

#CONSTANTS
left_wall = pygame.Rect(0, 47, 60, Y_resize)
right_wall = pygame.Rect(X_resize-60, 47, 60, Y_resize)
down_wall = pygame.Rect(0, Y_resize-47, X_resize, Y_resize)
up_wall_1 = pygame.Rect(60, 0, 360, 47)
up_wall_2 = pygame.Rect(480, 0, 360, 47)

single_agent = False
render = True
render_mode = "silent"  # Available: window (pop-up) and matplotlib (plt.imshow). Video rendering planned for the future.

text = get_font(25).render('Time: ' + str(time.time()), True, (255,173,1))
text2 = get_font(25).render('Time with ball: ' + str(time.time()), True, (255,173,1))
textRect = text.get_rect()
textRect2 = text.get_rect()
textRect3 = text.get_rect()
textRect.center = (int(X*(0.40)), int(Y*(0.965)))
textRect2.center = (int(X*(0.75)), int(Y*(0.965)))
game_time = 0
orig_surf = get_font(25).render("*slip*", True, (0,0,255))
txt_list = []
t = 0
slipped = False
clock = pygame.time.Clock()
user_id = ''

class LogFrame:
  def __init__(self, timestep:int , state_env, state_mdp:list, action_env:tuple, action_mdp:tuple, onion_time:float, game_time:float):
    self.timestep = timestep
    self.state_env = state_env
    self.state_mdp = state_mdp
    self.action_env = action_env
    self.action_mdp = action_mdp
    self.onion_time = onion_time
    self.game_time = game_time

display_surface = pygame.display.set_mode((X, Y), pygame.RESIZABLE)
pygame.display.set_caption("Menu")
white = (255, 255, 255)
GREY = (100,100,100)
BG = pygame.image.load("/home/anavc/Overcooked_Gym/overcooked-gym/overcooked_ai_py/data/graphics/Background.png")

def fade_in_text(txt_list, dec):
    for txt in txt_list:
        display_surface.blit(txt[0], txt[1])
        if txt[0].get_alpha() <=0:
            txt_list.remove(txt)
        else: txt[0].set_alpha(txt[0].get_alpha()-dec)
    
    if len(txt_list)>200:
        del txt_list[-100:] 

def valid_id():
    i = 0
    if not user_id == '':
        i = len(glob.glob1("/home/anavc/Overcooked_Gym/overcooked-gym/logfiles/", "logfile_{}_*".format(user_id)))
        print(user_id)
    if  i > 0:
        return False
    return user_id.isalnum()

def main_menu():

    global user_id

    invalid_txt_list = []

    while not GAME_OVER:
        display_surface.fill(white)
        display_surface.blit(BG, (0, 0))

        MENU_MOUSE_POS = pygame.mouse.get_pos()

        MENU_TEXT = get_font(50).render("TOXIC WASTE GAME", True, "#b68f40")
        MENU_RECT = MENU_TEXT.get_rect(center=(455, 150))
        
        MENU_TEXT2 = get_font(30).render("ENTER ID: ", True, "#d7fcd4")
        MENU_RECT2 = MENU_TEXT.get_rect(center=(455, 250))

        MENU_TEXT3 = get_font(30).render(user_id, True, "#d7fcd4")
        MENU_RECT3 = MENU_TEXT.get_rect(center=(750, 250))
        
        MENU_TEXT4 = get_font(30).render("INVALID USER ID", True, "#ff0000")

        PLAY_BUTTON = Button(image=pygame.image.load("/home/anavc/Overcooked_Gym/overcooked-gym/overcooked_ai_py/data/graphics/Play Rect.png"), pos=(455, 400), 
                            text_input="TUTORIAL", font=get_font(35), base_color="#d7fcd4", hovering_color="White")
        QUIT_BUTTON = Button(image=pygame.image.load("/home/anavc/Overcooked_Gym/overcooked-gym/overcooked_ai_py/data/graphics/Quit Rect.png"), pos=(455, 550), 
                            text_input="QUIT", font=get_font(35), base_color="#d7fcd4", hovering_color="White")

        display_surface.blit(MENU_TEXT, MENU_RECT)
        display_surface.blit(MENU_TEXT2, MENU_RECT2)
        display_surface.blit(MENU_TEXT3, MENU_RECT3)

        for button in [PLAY_BUTTON, QUIT_BUTTON]:
            button.changeColor(MENU_MOUSE_POS)
            button.update(display_surface)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE : user_id = user_id[:-1]
                elif event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                    if valid_id():
                        play_tutorial()
                    else:
                        pos = (250, 300)
                        txt_surf = MENU_TEXT4.copy()
                        txt_surf.set_alpha(200)
                        invalid_txt_list.append([txt_surf,pos])
                elif event.key!= pygame.K_SPACE: user_id += event.unicode
            if event.type == pygame.MOUSEBUTTONDOWN:
                if valid_id():
                    if PLAY_BUTTON.checkForInput(MENU_MOUSE_POS):
                        play_tutorial()
                else:
                    pos = (250, 300)
                    txt_surf = MENU_TEXT4.copy()
                    txt_surf.set_alpha(200)
                    invalid_txt_list.append([txt_surf,pos])
                    
                if QUIT_BUTTON.checkForInput(MENU_MOUSE_POS):
                    pygame.quit()
                    sys.exit()

        fade_in_text(invalid_txt_list, 5)
        pygame.display.update()
        clock.tick(60)

def play_tutorial():
    layout = "kitchen2"
    env = Overcooked(layout=layout)
    #agent = HumanAgent(action_meanings=env.action_meanings, name="Player 1")  # 1 - selects robot; 0 - selects human
    teammate = AstroHandcoded(layout, 1, env=env)
    #teammate = AstroHandcoded(layout, 1, env=env)
    env = SingleAgentWrapper(env, teammate)
    state = env.reset()
    frame = np.rot90(env.render(render_mode))
    #frame = imresize(frame, [500, 500], 'bilinear')
    image = pygame.surfarray.make_surface(frame)
    terminal = False
    new_frame = pygame.USEREVENT + 1
    pygame.time.set_timer(new_frame, MPF)
    action = 5
    start_game = time.time()

    while not terminal:
        display_surface.fill(white)
        display_surface.blit(pygame.transform.flip(image, True, False), (0, 0))

        action = None
    
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
                #print("Time passed")
                action = 5

            if event.type == pygame.KEYDOWN:
                #pygame.time.set_timer(new_frame, 0)
                #pygame.time.set_timer(new_frame, MPF)
                #print("Key Pressed")
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

            textRect3.center = (state[3]*15, state[2]*15)
            
            frame = np.rot90(env.render(render_mode))
            #frame = np.array(Image.fromarray(frame).resize(size=(Y_resize, X_resize)))
            image = pygame.surfarray.make_surface(frame)
            image = pygame.transform.scale(image, (X_resize, Y_resize))

            # Draws the surface object to the screen.
            text = get_font(25).render('Time: ' + str(round(time.time()-start_game, 1)), True,(255,173,1) )
            text2 = get_font(25).render('Time with ball: ' + str(round(teammate.onion_time)), True, (255,173,1))
            display_surface.blit(text, textRect)
            display_surface.blit(text2, textRect2)

            if state[8] == 1:
                pos = (state[3]*(X/15), (state[2]-1)*(Y/15))
                txt_surf = orig_surf.copy()
                txt_surf.set_alpha(200)
                txt_list.append([txt_surf,pos])
            fade_in_text(txt_list, 50)

            pygame.display.update()
    
    mid_screen0_1()

def mid_screen0_1():

    while True:
        display_surface.fill(white)
        display_surface.blit(BG, (0, 0))

        MENU_MOUSE_POS = pygame.mouse.get_pos()

        MENU_TEXT = get_font(40).render("READY FOR LEVEL 1?", True, "#b68f40")
        MENU_RECT = MENU_TEXT.get_rect(center=(455, 220))

        PLAY_BUTTON = Button(image=pygame.image.load("/home/anavc/Overcooked_Gym/overcooked-gym/overcooked_ai_py/data/graphics/Quit Rect.png"), pos=(455, 350), 
                            text_input="LEVEL 1", font=get_font(35), base_color="#d7fcd4", hovering_color="White")
        QUIT_BUTTON = Button(image=pygame.image.load("/home/anavc/Overcooked_Gym/overcooked-gym/overcooked_ai_py/data/graphics/Quit Rect.png"), pos=(455, 500), 
                            text_input="TUTORIAL", font=get_font(35), base_color="#d7fcd4", hovering_color="White")

        display_surface.blit(MENU_TEXT, MENU_RECT)

        for button in [PLAY_BUTTON, QUIT_BUTTON]:
            button.changeColor(MENU_MOUSE_POS)
            button.update(display_surface)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if PLAY_BUTTON.checkForInput(MENU_MOUSE_POS):
                    play_lvl1()
                if QUIT_BUTTON.checkForInput(MENU_MOUSE_POS):
                    play_tutorial()
                    sys.exit()

        pygame.display.update()

def play_lvl1():
    
    layout = "Lab1"
    env = Overcooked(layout=layout)
    #agent = HumanAgent(action_meanings=env.action_meanings, name="Player 1")  # 1 - selects robot; 0 - selects human
    teammate = AstroSmart(layout, 1, env=env)
    #teammate = AstroHandcoded(layout, 1, env=env)
    env = SingleAgentWrapper(env, teammate)
    state = env.reset()
    frame = np.rot90(env.render(render_mode))
    #frame = imresize(frame, [500, 500], 'bilinear')
    image = pygame.surfarray.make_surface(frame)
    terminal = False
    new_frame = pygame.USEREVENT + 1
    pygame.time.set_timer(new_frame, MPF)
    action = 5
    t = 0
    start_game = time.time()
    log = []

    while not terminal:
        display_surface.fill(white)
        display_surface.blit(pygame.transform.flip(image, True, False), (0, 0))

        action = None
    
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
                #print("Time passed")
                action = 5

            if event.type == pygame.KEYDOWN:
                #pygame.time.set_timer(new_frame, 0)
                #pygame.time.set_timer(new_frame, MPF)
                #print("Key Pressed")
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
            text = get_font(25).render('Time: ' + str(round(time.time()-start_game, 1)), True,(255,173,1) )
            text2 = get_font(25).render('Time with ball: ' + str(round(teammate.onion_time)), True, (255,173,1))
            display_surface.blit(text, textRect)
            display_surface.blit(text2, textRect2)

            if state[8] == 1:
                pos = (state[3]*(X/15), (state[2]-1)*(Y/15))
                txt_surf = orig_surf.copy()
                txt_surf.set_alpha(200)
                txt_list.append([txt_surf,pos])
            fade_in_text(txt_list, 50)

            pygame.display.update()

    log_file = f"/home/anavc/Overcooked_Gym/overcooked-gym/logfiles/logfile_{user_id}_lvl1.pickle"

    with open(log_file, "wb") as a:
        pickle.dump(log, a)

    print("TIMESTEPS: ", t)
    print("Game time: ", game_time)
    print("Time with onion in hand: ", round(teammate.onion_time, 1))
    print("Score: ", 100 - game_time - round(teammate.onion_time))
    mid_screen1_2(teammate.onion_time, game_time, 100 - game_time - teammate.onion_time)

def mid_screen1_2(onion_time, game_time, score):

    while True:
        display_surface.fill(white)
        display_surface.blit(BG, (0, 0))

        MENU_MOUSE_POS = pygame.mouse.get_pos()

        MENU_TEXT = get_font(47).render("LEVEL 1 COMPLETED", True, "#b68f40")
        MENU_RECT = MENU_TEXT.get_rect(center=(455, 100))

        MENU_TEXT2 = get_font(30).render("GAME TIME: {}".format(round(game_time,1)), True, "#b68f40")
        MENU_RECT2 = MENU_TEXT2.get_rect(center=(455, 250))

        MENU_TEXT3 = get_font(30).render("TIME WITH ONION ", True, "#b68f40")
        MENU_RECT3 = MENU_TEXT2.get_rect(center=(455, 320))

        MENU_TEXT4 = get_font(30).render("SCORE: {}".format(round(score,1)), True, "#b68f40")
        MENU_RECT4 = MENU_TEXT2.get_rect(center=(455, 420))

        MENU_TEXT5 = get_font(30).render("IN HAND: {}".format(round(onion_time,1)), True, "#b68f40")
        MENU_RECT5 = MENU_TEXT2.get_rect(center=(455, 350))

        PLAY_BUTTON = Button(image=pygame.image.load("/home/anavc/Overcooked_Gym/overcooked-gym/overcooked_ai_py/data/graphics/Play Rect.png"), pos=(455, 550), 
                            text_input="LEVEL 2", font=get_font(40), base_color="#d7fcd4", hovering_color="White")
        #QUIT_BUTTON = Button(image=pygame.image.load("/home/anavc/Overcooked_Gym/overcooked-gym/overcooked_ai_py/data/graphics/Quit Rect.png"), pos=(455, 650), 
        #                    text_input="LEVEL 1", font=get_font(40), base_color="#d7fcd4", hovering_color="White")

        display_surface.blit(MENU_TEXT, MENU_RECT)
        display_surface.blit(MENU_TEXT2, MENU_RECT2)
        display_surface.blit(MENU_TEXT3, MENU_RECT3)
        display_surface.blit(MENU_TEXT4, MENU_RECT4)
        display_surface.blit(MENU_TEXT5, MENU_RECT5)

        for button in [PLAY_BUTTON]:
            button.changeColor(MENU_MOUSE_POS)
            button.update(display_surface)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if PLAY_BUTTON.checkForInput(MENU_MOUSE_POS):
                    play_lvl2()
                #if QUIT_BUTTON.checkForInput(MENU_MOUSE_POS):
                #    play_lvl1()
                #    sys.exit()

        pygame.display.update()

def play_lvl2():
    
    layout = "Lab2"
    env = Overcooked(layout=layout)
    #agent = HumanAgent(action_meanings=env.action_meanings, name="Player 1")  # 1 - selects robot; 0 - selects human
    teammate = AstroSmart(layout, 1, env=env)
    #teammate = AstroHandcoded(layout, 1, env=env)
    env = SingleAgentWrapper(env, teammate)
    state = env.reset()
    frame = np.rot90(env.render(render_mode))
    #frame = imresize(frame, [500, 500], 'bilinear')
    image = pygame.surfarray.make_surface(frame)
    terminal = False
    new_frame = pygame.USEREVENT + 1
    pygame.time.set_timer(new_frame, MPF)
    action = 5
    t = 0
    start_game = time.time()
    log = []

    while not terminal:
        display_surface.fill(white)
        display_surface.blit(pygame.transform.flip(image, True, False), (0, 0))

        action = None
    
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
                #print("Time passed")
                action = 5

            if event.type == pygame.KEYDOWN:
                #pygame.time.set_timer(new_frame, 0)
                #pygame.time.set_timer(new_frame, MPF)
                #print("Key Pressed")
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
            text = get_font(25).render('Time: ' + str(round(time.time()-start_game, 1)), True,(255,173,1) )
            text2 = get_font(25).render('Time with ball: ' + str(round(teammate.onion_time)), True, (255,173,1))
            display_surface.blit(text, textRect)
            display_surface.blit(text2, textRect2)

            if state[8] == 1:
                pos = (state[3]*(X/15), (state[2]-1)*(Y/15))
                txt_surf = orig_surf.copy()
                txt_surf.set_alpha(200)
                txt_list.append([txt_surf,pos])
            fade_in_text(txt_list, 50)

            pygame.display.update()

    log_file = f"/home/anavc/Overcooked_Gym/overcooked-gym/logfiles/logfile_{user_id}_lvl2.pickle"

    with open(log_file, "wb") as a:
        pickle.dump(log, a)

    print("TIMESTEPS: ", t)
    print("USER ID: ", user_id)
    print("Game time: ", game_time)
    print("Time with onion in hand: ", round(teammate.onion_time, 1))
    print("Score: ", 100 - game_time - round(teammate.onion_time))
    game_over(teammate.onion_time, game_time, 100 - game_time - teammate.onion_time)

def game_over(onion_time, game_time, score):
    
    while True:
        display_surface.fill(white)
        display_surface.blit(BG, (0, 0))

        MENU_MOUSE_POS = pygame.mouse.get_pos()

        MENU_TEXT = get_font(50).render("GAME COMPLETE", True, "#b68f40")
        MENU_RECT = MENU_TEXT.get_rect(center=(455, 100))

        MENU_TEXT2 = get_font(30).render("GAME TIME: {}".format(round(game_time,1)), True, "#b68f40")
        MENU_RECT2 = MENU_TEXT2.get_rect(center=(455, 210))

        MENU_TEXT3 = get_font(30).render("TIME WITH ONION ", True, "#b68f40")
        MENU_RECT3 = MENU_TEXT2.get_rect(center=(455, 280))

        MENU_TEXT4 = get_font(30).render("SCORE: {}".format(round(score,1)), True, "#b68f40")
        MENU_RECT4 = MENU_TEXT2.get_rect(center=(455, 380))

        MENU_TEXT5 = get_font(30).render("IN HAND: {}".format(round(onion_time,1)), True, "#b68f40")
        MENU_RECT5 = MENU_TEXT2.get_rect(center=(455, 310))

        MENU_TEXT6 = get_font(20).render("COMPLETION CODE: {}".format(COMPLETION_CODE), True, "#b68f40")
        MENU_RECT6 = MENU_TEXT2.get_rect(center=(455, 450))

        QUIT_BUTTON = Button(image=pygame.image.load("/home/anavc/Overcooked_Gym/overcooked-gym/overcooked_ai_py/data/graphics/Quit Rect.png"), pos=(455, 570), 
                            text_input="QUIT", font=get_font(40), base_color="#d7fcd4", hovering_color="White")

        display_surface.blit(MENU_TEXT, MENU_RECT)
        display_surface.blit(MENU_TEXT2, MENU_RECT2)
        display_surface.blit(MENU_TEXT3, MENU_RECT3)
        display_surface.blit(MENU_TEXT4, MENU_RECT4)
        display_surface.blit(MENU_TEXT5, MENU_RECT5)
        display_surface.blit(MENU_TEXT6, MENU_RECT6)
        
        for button in [QUIT_BUTTON]:
            button.changeColor(MENU_MOUSE_POS)
            button.update(display_surface)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if QUIT_BUTTON.checkForInput(MENU_MOUSE_POS):
                    pygame.quit()
                    sys.exit()

        pygame.display.update()
    
    GAME_OVER = True

main_menu()
