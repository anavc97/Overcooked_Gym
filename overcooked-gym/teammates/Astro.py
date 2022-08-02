from numpy import ndarray
import numpy as np
import pickle
import itertools
from yaaf import Timestep
from yaaf.policies import deterministic_policy
from overcooked2 import Overcooked, HOLDING_ONION, HOLDING_NOTHING, HOLDING_DISH, ACTION_MEANINGS
from teammates.HandcodedTeammate import HandcodedTeammate
import time, random, copy
import glob


fileCounter = len(glob.glob1("/home/anavc/Overcooked_Gym/overcooked-gym/","logfile_AstroHuman_*"))
log_file = f"logfile_AstroHuman_{fileCounter-1}.pickle"
print("READING FROM: ", log_file)


OFFSETS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
agents = ['Human', 'Astro']
S_COEF = 0.9 #prob of slipping
STATE_MAP = np.array([
["2", "2", "2", "2", "2", "2", "2", "4", "7", "7", "7", "7", "7", "7", "7"],
["2", "2", "2", "2", "2", "2", "2", "4", "7", "7", "7", "7", "7", "7", "7"],
["2", "2", "2", "2", "2", "2", "2", "4", "4", "4", "7", "7", "7", "7", "7"],
["2", "2", "2", "2", "2", "2", "2", "4", "4", "4", "7", "7", "7", "7", "7"],
["2", "2", "2", "2", "2", "3", "3", "4", "4", "4", "7", "7", "7", "7", "7"],
["2", "2", "2", "2", "2", "3", "3", "4", "4", "4", "4", "4", "7", "7", "7"],
["1", "1", "1", "1", "3", "3", "3", "4", "4", "4", "4", "4", "4", "7", "7"],
["1", "1", "1", "1", "3", "3", "3", "4", "4", "4", "4", "4", "4", "7", "7"],
["1", "1", "1", "1", "3", "3", "3", "4", "4", "4", "4", "4", "4", "7", "7"],
["1", "1", "1", "1", "3", "3", "3", "4", "4", "6", "4", "4", "6", "6", "6"],
["1", "1", "1", "1", "1", "3", "3", "3", "4", "6", "6", "6", "6", "6", "6"],
["0", "0", "0", "0", "0", "5", "5", "5", "5", "5", "5", "5", "5", "5", "5"],
["0", "0", "0", "0", "0", "5", "5", "5", "5", "5", "5", "5", "5", "5", "5"],
["0", "0", "0", "0", "0", "5", "5", "5", "5", "5", "5", "5", "5", "5", "5"],
["0", "0", "0", "0", "0", "5", "5", "5", "5", "5", "5", "5", "5", "5", "5"]])

ADJACENCY_MATRIX = np.array(
    [
        [0, 1, 0, 0, 0, 1, 0, 0],
        [1, 0, 1, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 1, 1],
        [1, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 1, 0, 1],
        [0, 0, 0, 0, 1, 0, 1, 0],
    ]
)
ACTION_MEANINGS_MDP = [
    "move to lower-index node",
    "move to second-lower-index node",
    "move to third-lower-index node",
    "move to fourth-lower-index node",
    "stay",
    "act"
]

ACTION_SPACE = tuple(range(len(ACTION_MEANINGS_MDP)))
JOINT_ACTION_SPACE = list(itertools.product(ACTION_SPACE, repeat=2))


class AstroHandcoded(HandcodedTeammate):
    """Agent that fetches the items needed to make and serve the soup and places them in a balcony, so that a teammate
    can make the soup and serving without having to fetch anything."""

    def __init__(self, layout, index, env=None):
        super().__init__(layout, index)
        self.num_rows, self.num_columns = self.layout.shape
        self.env = env
        self.onion_time = 0
        self.curr_time = 0
        self.prev_time = 0
    

    def policy(self, state: ndarray):
        a0_row, a0_column, a1_row, a1_column, a0_heading, a1_heading, a0_hand, a1_hand, pan = state[:9] #a0 - players a1 - teammate
        hand = a0_hand if self.index == 0 else a1_hand
        onions = self.env.onions
        dist_onion = np.zeros(len(onions.pos))

        if a0_hand == HOLDING_ONION:

            if self.prev_time == 0:
                self.prev_time = time.time()

            self.curr_time = time.time()

            self.onion_time += self.curr_time-self.prev_time
            self.prev_time = self.curr_time    
        
        if a0_hand != HOLDING_ONION:
            self.curr_time = 0
            self.prev_time = 0
            

        dist_tm = np.linalg.norm(np.array([a0_row,a0_column])-np.array([a1_row,a1_column]))
        for i in range(0,len(onions.pos)):
            if onions.status[i] == 0: 
                dist_onion[i] = np.linalg.norm(np.array([onions.pos[i][0], onions.pos[i][1]])-np.array([a1_row,a1_column]))
            else:
                dist_onion[i] = 100
        print(dist_onion)
        if self.index == 1 and a0_hand == HOLDING_ONION:
            pos = (a1_row, a1_column)
            action = self._action_to_move_to(state, (a0_row, a0_column))
        else:
            action = self._action_to_move_to(state, onions.pos[np.argmin(dist_onion)])
            print("Moving to target: ", onions.pos[np.argmin(dist_onion)])
            '''if (a1_column == 12 and 9<=a1_row<=13) or (a1_row == 13 and 6<=a1_column<=12) or (a1_row == 10 and 9<=a1_column<=13) or (a1_row == 9 and 12<=a1_column<=13) or (a1_column == 9 and a1_row == 9) or (a1_column == 8 and a1_row == 12):
                print("Slipery Slope")
                if random.random()>=(1-S_COEF):
                    print("Slipped")
                    x,y =  self.cell_facing_agent(a1_row, a1_column, a1_heading)
                    action = self._action_to_move_to(state,(x,y-1))'''
        

        return deterministic_policy(action, len(ACTION_MEANINGS))

    def _reinforce(self, timestep: Timestep):
        pass

    ####################################################################################################################
    #                                              AUXILIARY METHODS
    ####################################################################################################################

    def cell_facing_agent(self, row, column, direction):

        dr, dc = OFFSETS[direction]
        object_row = row + dr
        object_column = column + dc

        if object_row < 0: object_row = 0
        if object_row > self.num_rows: object_row = self.num_rows - 1

        if object_column < 0: object_column = 0
        if object_column > self.num_columns: object_column = self.num_columns - 1

        return object_row, object_column


#############################################################################################################################################################
#############################################################################################################################################################
#############################################################################################################################################################
#############################################################################################################################################################


class AstroSmart(HandcodedTeammate):
    """Agent that fetches the items needed to make and serve the soup and places them in a balcony, so that a teammate
    can make the soup and serving without having to fetch anything."""

    def __init__(self, layout, index, env=None):
        super().__init__(layout, index)
        self.layout = layout
        self.num_rows, self.num_columns = self.layout.shape
        self.env = env
        self.onion_time = 0
        self.curr_time = 0
        self.prev_time = 0
        self.balcony_contents = []
        self.state_map = STATE_MAP
        self.onions = self.env.onions
        self.human_pos = [(13,2), (9,1), (2,3), (6,6), (8,8), (9,9), (13,9), (5,12)]
        self.agent_pos = [(13,3), (7,2), (1,3), (7,5), (10,8), (12,8), (10,9), (4,10)]
        self.dist = np.zeros(len(self.onions.pos))
        self.last_state = [10,10,10,10,10,10]
        self.last_action = 0
        self.target = [False, False]
        self.index = index
        self.state_mdp = None
        self.t = 0

        with open("/home/anavc/Overcooked_Gym/overcooked-gym/mmdp.pickle", "rb") as f:
            self.mdp = pickle.load(f)
        print("Initializing MDP")
        with open("/home/anavc/Overcooked_Gym/overcooked-gym/policy.pickle", "rb") as f:
            self.p_joint =  pickle.load(f)
        print("Done MDP")

    

    def policy(self, state: ndarray):
        self.env.state[8] = 4
        a0_row, a0_column, a1_row, a1_column, a0_heading, a1_heading, a0_hand, a1_hand, pan = state[:9] #a0 - human 1 - astro
        self.balcony_contents = state[9:]
        self.state_mdp = self.state_converter(state[:9])
        #print("state: ", self.state_mdp)

        if self.last_state[0] != self.state_mdp[0]:#ROBOT
            self.target[1] = False
        
        if self.last_state[1] != self.state_mdp[1]:#HUMAN
            self.target[0] = False

        if self.last_state == self.state_mdp:
            a_joint = self.last_action
        else:
            a_joint = JOINT_ACTION_SPACE[np.random.choice(range(len(JOINT_ACTION_SPACE)), p=self.p_joint[self.mdp.state_index(self.state_mdp)])]
        #a_joint = JOINT_ACTION_SPACE[np.argmax(self.p_joint[self.mdp.state_index(self.state_mdp)])]

        self.last_state = copy.copy(self.state_mdp)
        self.last_action = copy.copy(a_joint)

        #print("ACTION for ", agents[self.index], ": ", a_joint[1-self.index])

        if a0_hand == HOLDING_ONION:

            if self.prev_time == 0:
                self.prev_time = time.time()

            self.curr_time = time.time()

            self.onion_time += self.curr_time-self.prev_time
            self.prev_time = self.curr_time
        
        if a0_hand != HOLDING_ONION:
            self.curr_time = 0
            self.prev_time = 0

        action = self.action_converter(state, self.state_mdp, a_joint[1-self.index]) # Human = player 0 Astro = player 1  
        self.last_state = copy.copy(self.state_mdp)
        print("T: ", self.t)
        self.t += 1
        return deterministic_policy(action, len(ACTION_MEANINGS))

    def _reinforce(self, timestep: Timestep):
        pass

    ####################################################################################################################
    #                                              AUXILIARY METHODS                                                   #
    ####################################################################################################################

    def go_to_node(self, node, state_env):
        
        if self.index == 0: # HUMAN
            if node == 0:
                action_env = self._action_to_move_to(state_env, self.human_pos[0])        
            elif node == 1:
                action_env = self._action_to_move_to(state_env, self.onions.pos[0])
            elif node == 2:
                action_env = self._action_to_move_to(state_env, self.onions.pos[1])
            elif node == 3:
                action_env = self._action_to_move_to(state_env, self.human_pos[3])
            elif node == 4:
                action_env = self._action_to_move_to(state_env, self.human_pos[4])
            elif node == 5:
                action_env = self._action_to_move_to(state_env, self.onions.pos[2])
            elif node == 6:
                action_env = self._action_to_move_to(state_env, self.onions.pos[3])
            elif node == 7:
                action_env = self._action_to_move_to(state_env, self.human_pos[7])

        if self.index == 1: # ROBOT
            if node == 0:
                action_env = self._action_to_move_to(state_env, self.agent_pos[0])        
            elif node == 1:
                action_env = self._action_to_move_to(state_env, self.agent_pos[1], is_teammate_obstacle=False)
            elif node == 2:
                action_env = self._action_to_move_to(state_env, self.agent_pos[2], is_teammate_obstacle=False)
            elif node == 3:
                action_env = self._action_to_move_to(state_env, self.agent_pos[3])
            elif node == 4:
                action_env = self._action_to_move_to(state_env, self.agent_pos[4])
            elif node == 5:
                action_env = self._action_to_move_to(state_env, self.agent_pos[5], is_teammate_obstacle=False)
            elif node == 6:
                action_env = self._action_to_move_to(state_env, self.agent_pos[6], is_teammate_obstacle=False)
            elif node == 7:
                action_env = self._action_to_move_to(state_env, self.agent_pos[7])
        
        return action_env

    def action_converter(self, state_env, state_mdp, action_mdp):
        
        if self.index == 1 and state_env[6] == HOLDING_ONION:
            pos = (state_env[2], state_env[3])
            if random.random()<S_COEF and  (state_mdp[1-self.index] == 5 or state_mdp[1-self.index] == 6):
                self.env.state[8] = 1
                x,y =  self.cell_facing_agent(pos[0], pos[1],  state_env[5])
                off = copy.copy(OFFSETS)
                off.pop(state_env[5])
                i,j = off[np.random.choice(range(len(off)))]
                return self._action_to_move_to(state_env,(x+i,y+j))
            return self._action_to_move_to(state_env, (state_env[0], state_env[1]))

        #IF STAY
        if action_mdp == 4:
            if self.index == 0: # if human
                pos = (state_env[0], state_env[1])
                if pos in self.human_pos or self.target[self.index]:
                    self.target[self.index] = True                                                             #IF HUMAN IS IN TARGET POSITION OF NODE
                    #print(agents[self.index], " faces onion: ", state_env[0], state_env[1]) 
                    return ACTION_MEANINGS.index("stay")
                elif not self.target[self.index]:
                    #print(agents[self.index], " goes to target in node: ", self.state_mdp[1], " with prob: ",  self.p_joint[self.mdp.state_index(self.state_mdp)]) # human goes to target position of node
                    return self.go_to_node(self.state_mdp[1-self.index], state_env)

            elif self.index == 1: #if robot
                pos = (state_env[2], state_env[3])
                if pos in self.agent_pos or self.target[self.index]:                                                      #IF robot IS IN TARGET POSITION OF NODE  
                    #print(agents[self.index], "faces onion.")
                    self.target[self.index] = True
                    return ACTION_MEANINGS.index("stay")
                elif not self.target[self.index]: 
                    #print(agents[self.index], " goes to target in node: ", self.state_mdp[0], " with prob: ",  self.p_joint[self.mdp.state_index(self.state_mdp)]) # robot goes to target position of node
                    if random.random()<S_COEF and  (self.state_mdp[1-self.index] == 5 or self.state_mdp[1-self.index] == 6):
                        self.env.state[8] = 1
                        x,y =  self.cell_facing_agent(pos[0], pos[1],  state_env[5])
                        off = copy.copy(OFFSETS)
                        off.pop(state_env[5])
                        i,j = off[np.random.choice(range(len(off)))]
                        return self._action_to_move_to(state_env,(x+i,y+j))

                    return self.go_to_node(self.state_mdp[1-self.index], state_env)

        #IF ACT
        elif action_mdp == 5:                                                               
            if self.index == 1: # ROBOT
                pos = (state_env[2], state_env[3])
                if pos in self.agent_pos or self.target[self.index]:                                                      #IF robot IS IN TARGET POSITION OF NODE  
                    #print(agents[self.index], "faces onion.")
                    self.target[self.index] = True
                    return ACTION_MEANINGS.index("stay")
                elif not self.target[self.index]: 
                    #print(agents[self.index], " goes to target in node: ", self.state_mdp[0], " with prob: ",  self.p_joint[self.mdp.state_index(self.state_mdp)]) # robot goes to target position of node
                    if random.random()<S_COEF and (self.state_mdp[1-self.index] == 5 or self.state_mdp[1-self.index] == 6):
                        self.env.state[8] = 1
                        x,y =  self.cell_facing_agent(pos[0], pos[1],  state_env[5])
                        off = copy.copy(OFFSETS)
                        off.pop(state_env[5])
                        i,j = off[np.random.choice(range(len(off)))]
                        return self._action_to_move_to(state_env,(x+i,y+j))

                    return self.go_to_node(self.state_mdp[1-self.index], state_env)

            elif self.index == 0: # HUMAN
                x,y = self.cell_facing_agent(state_env[0], state_env[1], state_env[4])
                pos = (state_env[0], state_env[1])
                for i in range(0,len(self.onions.pos)):
                    self.dist[i] = np.linalg.norm(np.array([pos[0],pos[1]])-np.array([self.onions.pos[i][0], self.onions.pos[i][1]]))
      
                if state_env[6] == HOLDING_ONION:                                           # if onion is already in hand
                    if (x,y) != (state_env[2], state_env[3]):                               #if not near robot
                        #print(agents[self.index], " meets robot")
                        return self._action_to_move_to(state_env, (state_env[2],state_env[3]))
                    else:
                        #print("Human is dropping onion.")
                        return ACTION_MEANINGS.index("act")
                
                elif pos in self.human_pos or min(self.dist)<=1 or self.target[self.index]:   
                    self.target[self.index]=True                                          #IF HUMAN IS IN TARGET POSITION OF NODE OR NEAR ONION
                    if (x,y) in self.env.balconies:
                        balcony_index = self.env.balconies.index((x,y))
                        if self.balcony_contents[balcony_index] != HOLDING_ONION: return self.face_balcony(state_env[0], state_env[1]) # IF NOT FACING ONION -> FACE ONION
                    elif self.layout[x,y] != 'B': return self.face_balcony(state_env[0], state_env[1]) # IF NOT FACING ONION -> FACE ONION
                    else: 
                        #print("Human is picking up onion") 
                        return ACTION_MEANINGS.index("act") #IF FACING ONION -> PICK IT UP
                else:
                    #print(agents[self.index], " goes to target in node: ", self.state_mdp[1], " with prob: ",  self.p_joint[self.mdp.state_index(self.state_mdp)]) # human goes to target position of node
                    return self.go_to_node(self.state_mdp[1-self.index], state_env)

                return ACTION_MEANINGS.index("act")
        
        #IF MOVE
        else:
            if self.index == 0: # if human
                pos = (state_env[0], state_env[1])
                if pos in self.human_pos or self.target[self.index]:
                    self.target[self.index] = True
                    adjacencies = np.where(ADJACENCY_MATRIX[self.state_mdp[1-self.index]] == 1)[0]
                    downgrade_to_lower_index = int(action_mdp) >= len(adjacencies)
                    action_mdp = 0 if downgrade_to_lower_index else action_mdp
                    node = adjacencies[action_mdp]
                    #print(agents[self.index], " goes to node: ", node, " with prob: ",  self.p_joint[self.mdp.state_index(self.state_mdp)])
                    return self.go_to_node(node, state_env)
                elif not self.target[self.index]:
                    #print(agents[self.index], " goes to target in node: ", self.state_mdp[1], " with prob: ",  self.p_joint[self.mdp.state_index(self.state_mdp)]) # human goes to target position of node
                    return self.go_to_node(self.state_mdp[1-self.index], state_env)

            elif self.index == 1: #if robot
                pos = (state_env[2], state_env[3])

                for i in range(0,len(self.onions.pos)):
                    self.dist[i] = np.linalg.norm(np.array([pos[0],pos[1]])-np.array([self.onions.pos[i][0], self.onions.pos[i][1]]))
                
                if random.random()<S_COEF and (self.state_mdp[1-self.index] == 5 or self.state_mdp[1-self.index] == 6):
                    print("Slipped - ", action_mdp)
                    self.env.state[8] = 1
                    x,y =  self.cell_facing_agent(pos[0], pos[1],  state_env[5])
                    off = copy.copy(OFFSETS)
                    off.pop(state_env[5])
                    i,j = off[np.random.choice(range(len(off)))]
                    return self._action_to_move_to(state_env,(x+i,y+j))     
                                   
                if pos in self.agent_pos or self.target[self.index] or min(self.dist) <= 1:
                    self.target[self.index] = True
                    adjacencies = np.where(ADJACENCY_MATRIX[self.state_mdp[1-self.index]] == 1)[0]
                    downgrade_to_lower_index = int(action_mdp) >= len(adjacencies)
                    action_mdp = 0 if downgrade_to_lower_index else action_mdp
                    node = adjacencies[action_mdp]
                    #print(agents[self.index], " goes to node: ", node, " with prob: ",  self.p_joint[self.mdp.state_index(self.state_mdp)
                                        
                    return self.go_to_node(node, state_env)
                else:
                    #print(agents[self.index], " goes to target in node: ", self.state_mdp[1-self.index], " with prob: ",  self.p_joint[self.mdp.state_index(self.state_mdp)]) # human goes to target position of node
                    return self.go_to_node(self.state_mdp[1-self.index], state_env)

    def state_converter(self,state_env):

        p1 = int(self.state_map[state_env[2], state_env[3]]) #Robot pos
        p2 = int(self.state_map[state_env[0], state_env[1]]) #Human pos

        state_mdp = [p1,p2,self.onions.status[0], self.onions.status[1], self.onions.status[2], self.onions.status[3]]

        return state_mdp

    def face_balcony(self, a_row, a_column):

        for a in range(0,4):
            x,y = self.cell_facing_agent(a_row, a_column, a)
            #print(agents[self.index], " is in position: ", a_row, a_column)                 
            #print(agents[self.index], " is facing cell: ", x,y)                 
            if (x,y) not in self.env.balconies:
                continue
            balcony_index = self.env.balconies.index((x,y))
            if self.balcony_contents[balcony_index] == HOLDING_ONION:
                #print("Cell facing ONION: ", x,y)
                heading = a
                break

        return heading

    def cell_facing_agent(self, row, column, direction):

        dr, dc = OFFSETS[direction]
        object_row = row + dr
        object_column = column + dc

        if object_row < 0: object_row = 0
        if object_row > self.num_rows: object_row = self.num_rows - 1

        if object_column < 0: object_column = 0
        if object_column > self.num_columns: object_column = self.num_columns - 1

        return object_row, object_column

#############################################################################################################################################################
#############################################################################################################################################################
#############################################################################################################################################################
#############################################################################################################################################################


class AstroFake(HandcodedTeammate):
    """Agent that fetches the items needed to make and serve the soup and places them in a balcony, so that a teammate
    can make the soup and serving without having to fetch anything."""

    def __init__(self, layout, index, env=None):
        super().__init__(layout, index)
        self.layout = layout
        self.num_rows, self.num_columns = self.layout.shape
        self.env = env
        self.onion_time = 0
        self.curr_time = 0
        self.prev_time = 0
        self.cur_frame = 0
        with open(log_file, "rb") as f:
            self.log = pickle.load(f)
        
    def policy(self, state: ndarray):

        self.env.state[8] = 4
        a0_row, a0_column, a1_row, a1_column, a0_heading, a1_heading, a0_hand, a1_hand, pan = state[:9] #a0 - human 1 - astro

        if a0_hand == HOLDING_ONION:

            if self.prev_time == 0:
                self.prev_time = time.time()

            self.curr_time = time.time()

            self.onion_time += self.curr_time-self.prev_time
            self.prev_time = self.curr_time
        
        if a0_hand != HOLDING_ONION:
            self.curr_time = 0
            self.prev_time = 0

        a_joint = self.log[self.cur_frame].action_env
        action = a_joint[1]
        print("ACTIONNNN: ", action)

        self.cur_frame += 1

        return deterministic_policy(action, len(ACTION_MEANINGS))
    
    def _reinforce(self, timestep: Timestep):
        pass