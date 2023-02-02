from numpy import ndarray
import numpy as np
import pickle
import itertools
from yaaf import Timestep
from yaaf.policies import deterministic_policy
from overcooked2 import Overcooked, HOLDING_ONION, HOLDING_NOTHING, HOLDING_DISH, ACTION_MEANINGS, LAYOUTS
from teammates.HandcodedTeammate import HandcodedTeammate
import time, random, copy
import glob

LVL=1
LOG_NR = 215
log_file = f"/home/anavc/Overcooked_Gym/overcooked-gym/logfiles/logfile_{LOG_NR}_lvl{LVL}.pickle"

OFFSETS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
agents = ['Human', 'Astro']
S_COEF = 0 #prob of slipping
STATE_MAPS = []
ADJACENCY_MATRIXS = []
#LAB STATE MAP
STATE_MAPS.append(np.array([
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
["0", "0", "0", "0", "0", "5", "5", "5", "5", "5", "5", "5", "5", "5", "5"]]))

#LAB2 STATE MAP
STATE_MAPS.append(np.array([
["5", "5", "5", "5", "5", "5", "5", "6", "6", "6", "6", "7", "7", "7", "7"],
["5", "5", "5", "5", "5", "5", "5", "6", "6", "6", "6", "7", "7", "7", "7"],
["5", "5", "5", "5", "5", "5", "5", "6", "6", "6", "6", "7", "7", "7", "7"],
["5", "5", "5", "5", "5", "5", "5", "6", "6", "6", "6", "7", "7", "7", "7"],
["5", "5", "5", "5", "5", "5", "5", "6", "6", "6", "6", "7", "7", "7", "7"],
["5", "5", "5", "5", "5", "5", "5", "6", "6", "6", "6", "7", "7", "7", "7"],
["5", "5", "5", "5", "5", "5", "8", "8", "8", "8", "8", "7", "7", "7", "7"],
["4", "4", "4", "4", "4", "4", "8", "8", "8", "8", "3", "3", "3", "3", "3"],
["4", "4", "4", "4", "4", "4", "8", "8", "8", "8", "3", "3", "3", "3", "3"],
["4", "4", "4", "4", "4", "4", "2", "2", "2", "2", "3", "3", "3", "3", "3"],
["0", "0", "2", "2", "2", "4", "2", "2", "2", "2", "3", "3", "3", "3", "3"],
["0", "0", "0", "0", "2", "2", "2", "1", "1", "1", "3", "3", "3", "3", "3"],
["0", "0", "0", "0", "1", "1", "1", "1", "1", "1", "1", "3", "3", "3", "3"],
["0", "0", "0", "0", "1", "1", "1", "1", "1", "1", "1", "3", "3", "3", "3"],
["0", "0", "0", "0", "1", "1", "1", "1", "1", "1", "1", "3", "3", "3", "3"]]))

#LAB AD_MAT
ADJACENCY_MATRIXS.append(np.array(
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
))

#LAB2 AD_MAT
ADJACENCY_MATRIXS.append(np.array(
    [
        [0, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 0, 0, 0, 0, 1],
        [0, 1, 1, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 1, 1, 0, 1, 0],

    ]
))

ACTION_MEANINGS_MDP = []

ACTION_MEANINGS_MDP.append([
    "move to lower-index node",
    "move to second-lower-index node",
    "move to third-lower-index node",
    "move to fourth-lower-index node",
    "stay",
    "act"
])

ACTION_MEANINGS_MDP.append([
    "move to lower-index node",
    "move to second-lower-index node",
    "move to third-lower-index node",
    "move to fourth-lower-index node",
    "stay",
    "act"
])

class AstroHandcoded(HandcodedTeammate):
    """Agent that fetches the items needed to make and serve the soup and places them in a balcony, so that a teammate
    can make the soup and serving without having to fetch anything."""

    def __init__(self, layout, index, env=None):
        super().__init__(LAYOUTS[layout], index)
        self.layout = LAYOUTS[layout]
        self.layout_name = layout
        self.S_COEF = 0
        if self.layout_name == 'kitchen2':
            self.S_COEF = 0.90
        self.num_rows, self.num_columns = self.layout.shape
        self.env = env
        self.onion_time = 0
        self.curr_time = 0
        self.prev_time = 0
    

    def policy(self, state: ndarray):
        self.env.state[8] = 4
        a0_row, a0_column, a1_row, a1_column, a0_heading, a1_heading, a0_hand, a1_hand, pan = state[:9] #a0 - players a1 - teammate
        hand = a0_hand if self.index == 0 else a1_hand
        onions = self.env.onions
        pos = (state[2], state[3])
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
        
        if self.index == 1 and a0_hand == HOLDING_ONION:
            pos = (a1_row, a1_column)
            action = self._action_to_move_to(state, (a0_row, a0_column))
        else:
            action = self._action_to_move_to(state, onions.pos[np.argmin(dist_onion)])
           
        if random.random()<self.S_COEF and self.layout[pos[0],pos[1]] == 'I':
            next_move = self.slip_move(pos,state)
            action = self._action_to_move_to(state,next_move) 
        

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

    def slip_move(self, pos, state_env):
        print("Slipped")
        self.env.state[8] = 1
        x,y =  self.cell_facing_agent(pos[0], pos[1],  state_env[5])
        off = copy.copy(OFFSETS)
        off.pop(state_env[5])
        i,j = off[np.random.choice(range(len(off)))]
        next_move = (x+i,y+j)

        return next_move


#############################################################################################################################################################
#############################################################################################################################################################
#############################################################################################################################################################
#############################################################################################################################################################


class AstroSmart(HandcodedTeammate):
    """Agent that fetches the items needed to make and serve the soup and places them in a balcony, so that a teammate
    can make the soup and serving without having to fetch anything."""

    def __init__(self, layout, index, env=None):
        super().__init__(LAYOUTS[layout], index)
        self.layout = LAYOUTS[layout]
        self.layout_name = layout
        self.S_COEF = 0
        if self.layout_name == 'kitchen2':
            self.S_COEF = 0.75
        self.num_rows, self.num_columns = self.layout.shape
        self.env = env
        self.onion_time = 0
        self.curr_time = 0
        self.prev_time = 0
        self.balcony_contents = []
        if self.layout_name == "Lab1":
            self.state_map = STATE_MAPS[0]
            self.ad_matrix = ADJACENCY_MATRIXS[0]
            self.human_pos = [(13,2), (9,1), (2,3), (6,5), (8,8), (13,9), (13,9), (5,12)]
            self.agent_pos = [(13,3), (7,2), (1,3), (7,5), (10,8), (12,8), (10,9), (4,10)]
            self.ACTION_SPACE = tuple(range(len(ACTION_MEANINGS_MDP[0])))
            
        elif self.layout_name == "Lab2":
            self.state_map = STATE_MAPS[1]
            self.ad_matrix = ADJACENCY_MATRIXS[1]
            self.human_pos = [(13,2), (13,9), (10,6), (8,11), (8,1), (4,5), (1,8), (2,13), (8,7)]
            self.agent_pos = [(12,2), (12,9), (10,9), (11,10), (8,1), (3,5), (1,10),(6,12), (6,7)]
            self.ACTION_SPACE = tuple(range(len(ACTION_MEANINGS_MDP[1])))

        self.onions = self.env.onions

        self.dist = np.zeros(len(self.onions.pos))
        self.last_state = [10,10,10,10,10,10]
        self.last_action = 0
        self.target = [False, False]
        self.index = index
        self.state_mdp = None
        self.t = 0
        self.mdp_ind = np.load('/home/anavc/Overcooked_Gym/overcooked-gym/mdp_ind_{}.npy'.format(self.layout_name), allow_pickle=True)
        self.p_joint =  np.load('/home/anavc/Overcooked_Gym/overcooked-gym/policy_{}.npy'.format(self.layout_name), allow_pickle=True)
        self.JOINT_ACTION_SPACE = list(itertools.product(self.ACTION_SPACE, repeat=2))

    def check_index(self, state):
        for ind, val in enumerate(self.mdp_ind):
            if np.array_equal(np.array(val),np.array(state)):
                return ind

    def prime_action_choice(self, prob):
        # Bias favouring robot actions == stay or act

        aux = copy.copy(prob)
        for i in range(len(prob)):
            if i >=24 and aux[i] != 0:
                aux[i] = aux[i]*1.1
        prob = aux / sum(aux)
        print("prob after: ", prob)
        #return np.random.choice(range(len(self.JOINT_ACTION_SPACE)), p=prob)
        return np.argmax(prob)

    def policy(self, state: ndarray):
        self.env.state[8] = 4
        a0_row, a0_column, a1_row, a1_column, a0_heading, a1_heading, a0_hand, a1_hand, pan = state[:9] #a0 - human 1 - astro
        self.balcony_contents = state[9:]
        self.state_mdp = self.state_converter(state[:9])
        print("state: ", self.state_mdp)

        if self.last_state[0] != self.state_mdp[0]:#ROBOT
            self.target[1] = False
        
        if self.last_state[1] != self.state_mdp[1]:#HUMAN
            self.target[0] = False

        if self.last_state == self.state_mdp:
            a_joint = self.last_action
        else:
            print("Prob before: ", self.p_joint[self.check_index(self.state_mdp)])
            a_joint = self.JOINT_ACTION_SPACE[self.prime_action_choice(self.p_joint[self.check_index(self.state_mdp)])]

        print("A JOINT: ",a_joint) # a0 - robot a1 - human
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
        self.t += 1
        return deterministic_policy(action, len(ACTION_MEANINGS))

    def _reinforce(self, timestep: Timestep):
        pass

    ####################################################################################################################
    #                                              AUXILIARY METHODS                                                   #
    ####################################################################################################################

    def go_to_node(self, node, state_env):
        
        if self.index == 0: # HUMAN
            action_env = self._action_to_move_to(state_env, self.human_pos[node])
            print("Target: ", self.human_pos[node], "Node: ", node)

        if self.index == 1: # ROBOT
            action_env = self._action_to_move_to(state_env, self.agent_pos[node], is_teammate_obstacle=False)
            print("Going to Node: ", node)
            '''  
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
            '''
        return action_env

    def action_converter(self, state_env, state_mdp, action_mdp):
        #if self.state_mdp is None:
        self.state_mdp = state_mdp
                
        if self.index == 1 and state_env[6] == HOLDING_ONION:
            pos = (state_env[2], state_env[3])
            if random.random()<self.S_COEF and self.layout[pos[0],pos[1]] == 'I':
                next_move = self.slip_move(pos,state_env)
                return self._action_to_move_to(state_env,next_move) 
            return self._action_to_move_to(state_env, (state_env[0], state_env[1]))

        #IF STAY
        if action_mdp == ACTION_MEANINGS.index("stay"):
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
                    if random.random()<self.S_COEF and self.layout[pos[0],pos[1]] == 'I':
                        next_move = self.slip_move(pos,state_env)
                        return self._action_to_move_to(state_env,next_move) 

                    return self.go_to_node(self.state_mdp[1-self.index], state_env)

        #IF ACT
        elif action_mdp == ACTION_MEANINGS.index("act"):                                                               
            if self.index == 1: # ROBOT
                pos = (state_env[2], state_env[3])
                if pos in self.agent_pos or self.target[self.index]:                                                      #IF robot IS IN TARGET POSITION OF NODE  
                    #print(agents[self.index], "faces onion.")
                    self.target[self.index] = True
                    return ACTION_MEANINGS.index("stay")
                elif not self.target[self.index]: 
                    #print(agents[self.index], " goes to target in node: ", self.state_mdp[0], " with prob: ",  self.p_joint[self.mdp.state_index(self.state_mdp)]) # robot goes to target position of node
                    if random.random()<self.S_COEF and self.layout[pos[0],pos[1]] == 'I':
                        next_move = self.slip_move(pos,state_env)
                        return self._action_to_move_to(state_env,next_move) 

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
                '''
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
                '''
                return ACTION_MEANINGS.index("act")
        
        #IF MOVE
        else:
            if self.index == 0: # if human
                pos = (state_env[0], state_env[1])
                if pos in self.human_pos or self.target[self.index]:
                    self.target[self.index] = True
                    adjacencies = np.where(self.ad_matrix[self.state_mdp[1-self.index]] == 1)[0]
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
                
                if random.random()<self.S_COEF and self.layout[pos[0],pos[1]] == 'I':
                    next_move = self.slip_move(pos,state_env)
                    return self._action_to_move_to(state_env,next_move)     
                                   
                if pos in self.agent_pos or self.target[self.index] or min(self.dist) <= 1:
                    self.target[self.index] = True
                    adjacencies = np.where(self.ad_matrix[self.state_mdp[1-self.index]] == 1)[0]
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
            if (x,y) not in self.env.balconies:
                continue
            balcony_index = self.env.balconies.index((x,y)) 
            if balcony_index >= len(self.balcony_contents):
                balcony_index = len(self.balcony_contents)
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

    def slip_move(self, pos, state_env):
        print("Slipped")
        self.env.state[8] = 1
        x,y =  self.cell_facing_agent(pos[0], pos[1],  state_env[5])
        off = copy.copy(OFFSETS)
        off.pop(state_env[5])
        i,j = off[np.random.choice(range(len(off)))]
        next_move = (x+i,y+j)

        return next_move
        
#############################################################################################################################################################
#############################################################################################################################################################
#############################################################################################################################################################
#############################################################################################################################################################


class AstroFake(HandcodedTeammate):
    """Agent that fetches the items needed to make and serve the soup and places them in a balcony, so that a teammate
    can make the soup and serving without having to fetch anything."""

    def __init__(self, layout, index, env=None):
        super().__init__(LAYOUTS[layout], index)
        self.layout = LAYOUTS[layout]
        self.layout_name = layout
        self.S_COEF = 0
        if self.layout_name == 'kitchen2':
            self.S_COEF = 0.75
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
        pos = (state[2], state[3])
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
        if random.random()<self.S_COEF and self.layout[pos[0],pos[1]] == 'I':
            next_move = self.slip_move(pos,state)
            action = self._action_to_move_to(state,next_move) 
        action = a_joint[1]

        self.cur_frame += 1

        return deterministic_policy(action, len(ACTION_MEANINGS))
    
    def _reinforce(self, timestep: Timestep):
        pass

    def slip_move(self, pos, state_env):
        print("Slipped")
        self.env.state[8] = 1
        x,y =  self.cell_facing_agent(pos[0], pos[1],  state_env[5])
        off = copy.copy(OFFSETS)
        off.pop(state_env[5])
        i,j = off[np.random.choice(range(len(off)))]
        next_move = (x+i,y+j)

        return next_move