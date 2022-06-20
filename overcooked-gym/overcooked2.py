import random
from itertools import product

import yaaf
from matplotlib import pyplot as plt
from PIL import Image

import numpy as np
from gym import Env, Wrapper
from gym.envs.registration import EnvSpec
from gym.spaces import Discrete, Box

from yaaf.agents import Agent

from overcooked_ai_py.mdp.actions import Direction
from overcooked_ai_py.mdp.overcooked_mdp import ObjectState, SoupState, Recipe, PlayerState
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

"""
Assigning constants to relevant information so we're not using magic numbers in the env's code
"""

# Rewards
MOVE_REWARD = -1.0
DELIVER_SOUP_REWARD = 1000  # before was 100
PICKUP_SOUP_REWARD = 250  # before was 25
ONION_ADDED_REWARD = 50  # before was 5

# Actions
ACTION_MEANINGS = ["up", "down", "left", "right", "act", "stay"]
OFFSETS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
JOINT_ACTION_MEANINGS = list(product(ACTION_MEANINGS, repeat=2))

# Features
BASE_FEATURES_MEANING = ["a0_row", "a0_column", "a1_row", "a1_column", "a0_heading", "a1_heading", "a0_hand", "a1_hand", "pan"]
EMPTY, ONE_ONION, TWO_ONIONS, THREE_ONIONS, SOUP_READY = range(5)       # Soup pan state
NORTH, SOUTH, WEST, EAST = range(4)                                     # Agent headings (turn direction)
HOLDING_NOTHING, HOLDING_ONION, HOLDING_DISH, HOLDING_SOUP = range(4)   # Agent hand state and balconies
HIGHEST_VALUE = 0   # EMPTY soup pan and agent HOLDING_NOTHING
LOWEST_VALUE = 4    # SOUP_READY


"""
Layouts
Design your maps here!

    X = Wall
    B = Balcony
    P = Soup Pan
    S = Soup Delivery Window
    O = Onion Supply
    D = Dish Supply
      = Walking Space for Both Agents
    1 = Walking Space for Agents 1
    2 = Walking Space for Agents 2

"""

LAYOUTS = {
    "small": np.array([
        ["X", "X", "X", "P", "X"],
        ["O", "1", "B", "2", "X"],
        ["D", "1", "B", "2", "X"],
        ["X", "X", "X", "S", "X"],
    ]),
    "open_space": np.array([
        ["X", "X", "X", "X", "P", "X"],
        ["X", " ", " ", " ", " ", "X"],
        ["O", " ", "B", "B", " ", "X"],
        ["D", " ", "B", "B", " ", "X"],
        ["X", " ", " ", " ", " ", "X"],
        ["X", "X", "X", "X", "S", "X"],
    ]),
    "kitchen": np.array([
        ["X", "X", "X", "X", "P", "X", "S", "X"],
        ["X", " ", " ", " ", " ", " ", " ", "X"],
        ["X", " ", "B", "B", "B", "B", " ", "X"],
        ["X", " ", " ", " ", " ", " ", " ", "X"],
        ["X", "X", "X", "O", "X", "D", "X", "X"],
    ]),
    "Lab": np.array([
        ["D", "X", "X", "X", "X", "X", "X", "P", "X", "X", "X", "X", "O", "X", "S"],
        ["X", " ", " ", " ", " ", " ", " ", "B", " ", " ", " ", " ", "B", "B", "X"],
        ["X", "B", "B", " ", " ", "B", "B", "B", "B", "B", " ", "B", "B", "B", "X"],
        ["X", "B", "B", " ", " ", "B", "B", "B", "B", "B", " ", "B", "B", "B", "X"],
        ["X", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", "X"],
        ["X", " ", " ", " ", " ", " ", " ", " ", " ", " ", "B", "B", " ", " ", "X"],
        ["X", " ", " ", " ", " ", " ", " ", " ", " ", "B", "B", "B", "B", " ", "X"],
        ["X", " ", " ", " ", " ", " ", " ", " ", " ", "B", "B", "B", "B", " ", "X"],
        ["X", "B", " ", " ", " ", " ", " ", " ", " ", "B", "B", "B", "B", " ", "X"],
        ["X", " ", " ", " ", " ", " ", " ", " ", " ", " ", "B", "B", " ", " ", "X"],
        ["X", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", "X"],
        ["X", " ", " ", " ", " ", " ", "B", "B", "B", "B", "B", "B", " ", "B", "X"],
        ["X", " ", " ", " ", " ", " ", "B", "B", " ", "B", "B", "B", " ", "B", "X"],
        ["X", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", "B", "X"],
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"]
    ])
}

class Onion_list:

    def __init__(self, pos, status):
        self.pos = pos
        self.status = status

class Overcooked(Env):

    def __init__(self, layout="small", id="Overcooked-v2"):

        # Attributes
        self.layout: np.ndarray = np.copy(LAYOUTS[layout]) if isinstance(layout, str) else np.copy(layout)
        self.num_rows, self.num_columns = self.layout.shape
        self.action_meanings = ACTION_MEANINGS
        self.joint_action_meanings = JOINT_ACTION_MEANINGS
        self.num_actions = len(self.action_meanings)
        self.num_joint_actions = len(self.joint_action_meanings)
        self.valid_start_cells_a0 = self._valid_start_cells(self.layout, 0)
        self.valid_start_cells_a1 = self._valid_start_cells(self.layout, 1)
        self.layout[self.layout == '1'] = ' '   # For rendering and transitions
        self.layout[self.layout == '2'] = ' '   # For rendering and transitions
        self.balconies = self._balconies(self.layout)
        self.num_balconies = len(self.balconies)
        self.pan = self._pan_location(self.layout)
        self.features_meaning = BASE_FEATURES_MEANING + [f"balcony_{b}" for b in range(self.num_balconies)]
        self.onion_picked = None
        self.onions = Onion_list([(8,1), (2,2), (12,9), (9,10)], [0,0,0,0])

        # OpenAI Gym
        self.spec = EnvSpec(id=id)
        self.num_features = len(self.features_meaning)
        self.observation_space = Box(
            low=LOWEST_VALUE, high=HIGHEST_VALUE,
            shape=(len(self.features_meaning),), dtype=int)
        self.action_space = Discrete(self.num_joint_actions)
        self.reward_range = (MOVE_REWARD, DELIVER_SOUP_REWARD)
        self.metadata = {}

        self.state = self._random_initial_state()
        self.frame = self.render_state(self.state)
        self._timestep = 0

    # Main interface

    def reset(self):
        self.state = self._random_initial_state()
        self.frame = self.render_state(self.state)
        return self.state

    def step(self, joint_action: int):
        next_state, reward, terminal, info = self.transition(self.state, joint_action)
        self.state = next_state
        self._timestep += 1
        return next_state, reward, terminal, info

    def render(self, mode="human"):
        if mode == "human" or mode == "pygame" or mode == "window":
            self.frame = self.render_state(self.state, show=True)
        elif mode == "matplotlib" or mode == "plt":
            self.frame = self.render_state(self.state)
            plt.imshow(self.frame)
            plt.show()
        elif mode == "file":
            self.frame = self.render_state(self.state)
            image = Image.fromarray(self.frame)
            directory = "videocapture"
            yaaf.mkdir(directory)
            filename = f"{directory}/step_{self._timestep}.png"
            image.save(filename)
        elif mode == "silent":
            self.frame = self.render_state(self.state)
            return self.frame

    # Auxiliary methods

    def transition(self, state, joint_action):

        a0, a1 = self.unpack_joint_action(joint_action)
        a0_meaning, a1_meaning = self.joint_action_meanings[joint_action]

        a0_row, a0_column, a1_row, a1_column, a0_heading, a1_heading, a0_hand, a1_hand, pan = state[:9]
        balcony_contents = state[9:]

        def interact(object_row, object_column, hand, next_pan, next_balconies, reward):

            descriptor = self.layout[object_row, object_column]
            
            # Player interacts with teammate
            if cell_facing_agent(a0_row, a0_column, a0_heading) == (a1_row, a1_column) and hand == HOLDING_ONION:
                next_hand = HOLDING_NOTHING
                if self.onion_picked is not None:
                    self.onions.status[self.onions.pos.index(self.onion_picked)] = 2
                    print("ONION STATUS UPDATED: ", self.onions.status)
                print("RFID: Beep!")
            # Nothing
            
            elif descriptor == 'X' or descriptor == ' ':
                next_hand = hand

            # Kitchen balcony
            elif descriptor == 'B':
                balcony_cell = object_row, object_column
                balcony_index = self.balconies.index(balcony_cell)
                balcony_contents = next_balconies[balcony_index]
                able_to_pick = balcony_contents != HOLDING_NOTHING and hand == HOLDING_NOTHING
                able_to_drop = balcony_contents == HOLDING_NOTHING and hand != HOLDING_NOTHING
                if able_to_pick:
                    next_hand = balcony_contents
                    next_balconies[balcony_index] = HOLDING_NOTHING
                    if balcony_contents == HOLDING_ONION:
                        for a in self.onions.pos:
                            dist_onion = np.linalg.norm(np.array([a[0], a[1]])-self.balconies[balcony_index])
                            if dist_onion <= 1:
                                self.onion_picked = a
                                self.onions.status[self.onions.pos.index(self.onion_picked)] = 1
                                print("Onion picked: ", self.onions.status)
                elif able_to_drop:
                    next_balconies[balcony_index] = hand
                    next_hand = HOLDING_NOTHING
                    self.onions.status[self.onions.pos.index(self.onion_picked)] = 0
                    self.onions.pos[self.onions.pos.index(self.onion_picked)] = self.balconies[balcony_index]
                    self.onion_picked = None
                    print("Onion droppped: ", self.onions.status, self.onions.pos)
                else:
                    next_hand = hand
 
            # Onion Supply
            elif descriptor == 'O':
                able_to_pick_onion = hand == HOLDING_NOTHING
                able_to_drop_onion = hand == HOLDING_ONION
                if able_to_pick_onion: next_hand = HOLDING_ONION
                elif able_to_drop_onion: next_hand = HOLDING_NOTHING
                else: next_hand = hand

            # Dish Supply
            elif descriptor == 'D':
                able_to_pick_dish = hand == HOLDING_NOTHING
                able_to_drop_dish = hand == HOLDING_DISH
                if able_to_pick_dish: next_hand = HOLDING_DISH
                elif able_to_drop_dish: next_hand = HOLDING_NOTHING
                else: next_hand = hand

            # Soup pan
            elif descriptor == 'P':

                able_to_add_onion = hand == HOLDING_ONION and next_pan in [EMPTY, ONE_ONION, TWO_ONIONS]
                able_to_pickup_soup = hand == HOLDING_DISH and next_pan == SOUP_READY

                if able_to_add_onion:
                    next_pan = next_pan + 1
                    next_hand = HOLDING_NOTHING
                    reward = ONION_ADDED_REWARD
                elif able_to_pickup_soup:
                    next_pan = EMPTY
                    next_hand = HOLDING_SOUP
                    reward = PICKUP_SOUP_REWARD
                else:
                    next_hand = hand

            # Soup delivery window
            elif descriptor == "S":
                able_to_deliver_soup = hand == HOLDING_SOUP
                if able_to_deliver_soup:
                    next_hand = HOLDING_NOTHING
                    reward = DELIVER_SOUP_REWARD
                else:
                    next_hand = hand
        
            else:
                raise ValueError("Should be unreachable")

            return next_hand, next_pan, next_balconies, reward

        def cell_facing_agent(row, column, direction):

            dr, dc = OFFSETS[direction]
            object_row = row + dr
            object_column = column + dc

            if object_row < 0: object_row = 0
            if object_row > self.num_rows: object_row = self.num_rows - 1

            if object_column < 0: object_column = 0
            if object_column > self.num_columns: object_column = self.num_columns - 1

            return object_row, object_column

        def try_move_agent(row, column, direction, teammate_row, teammate_column):

            dr, dc = OFFSETS[direction]
            next_row = row + dr
            next_column = column + dc

            if next_row < 0: next_row = 0
            if next_row > self.num_rows: next_row = self.num_rows - 1

            if next_column < 0: next_column = 0
            if next_column > self.num_columns: next_column = self.num_columns - 1

            # TODO fix out of bounds error when indexing numpy array with values larger than rows/columns and lower than zero
            # in_bounds = next_row < self.layout.shape[0] and next_column < self.layout.shape[1]
            walking_area = self.layout[next_row, next_column] == ' '
            blocked_by_teammate = teammate_row == next_row and teammate_column == next_column

            if walking_area and not blocked_by_teammate:
                return next_row, next_column
            else:
                return row, column

        def step_agent(row, column, heading, hand, action, action_meaning, teammate_row, teammate_column, next_pan, next_balconies, reward):
            moving = action_meaning != "act"

            if action_meaning == "stay":
                return row, column, heading, hand, pan, next_balconies, MOVE_REWARD
            
            if moving and heading != action:
                # Turn
                next_heading = action
                next_row, next_column = row, column
                next_hand = hand
            elif moving and heading == action:
                # Move
                next_row, next_column = try_move_agent(row, column, action, teammate_row, teammate_column)
                next_heading = action
                next_hand = hand
            else:
                # Act
                next_row, next_column = row, column
                next_heading = heading
                object_row, object_column = cell_facing_agent(row, column, heading)
                next_hand, next_pan, next_balconies, reward = interact(object_row, object_column, hand, next_pan, next_balconies, reward)

            return next_row, next_column, next_heading, next_hand, next_pan, next_balconies, reward

        next_pan, next_balconies = pan, balcony_contents
        reward = MOVE_REWARD

        next_a0_row, next_a0_column, next_a0_heading, next_a0_hand, next_pan, next_balconies, reward0 = step_agent(a0_row, a0_column, a0_heading, a0_hand, a0, a0_meaning, a1_row, a1_column, next_pan, next_balconies, reward)
        next_a1_row, next_a1_column, next_a1_heading, next_a1_hand, next_pan, next_balconies, reward1 = step_agent(a1_row, a1_column, a1_heading, a1_hand, a1, a1_meaning, next_a0_row, next_a0_column, next_pan, next_balconies, reward)

        cooking = pan == THREE_ONIONS
        if cooking: next_pan = SOUP_READY
        #terminal = (reward0 == DELIVER_SOUP_REWARD or reward1 == DELIVER_SOUP_REWARD)
        terminal = (all([a!=HOLDING_ONION for a in balcony_contents]) == True) and next_a0_hand != HOLDING_ONION
        if terminal:
            print("Game has ended.")
        info = {}
        next_state = np.array([next_a0_row, next_a0_column, next_a1_row, next_a1_column, next_a0_heading, next_a1_heading, next_a0_hand, next_a1_hand, next_pan] + list(next_balconies))

        return next_state, max(reward0,reward1), terminal, info

    def _random_initial_state(self):

        #a0_cell = random.choice(self.valid_start_cells_a0)
        a1_cell = (7,6)
        a0_cell = (6,6)
        '''
        a1_cell = a0_cell
        while a0_cell == a1_cell:
            a1_cell = random.choice(self.valid_start_cells_a1)
        '''
        a0_row, a0_column = a0_cell
        a1_row, a1_column = a1_cell
        a0_heading, a1_heading = random.choice([NORTH, SOUTH, WEST, EAST]), NORTH
        a0_hand, a1_hand = HOLDING_NOTHING, HOLDING_NOTHING
        pan = EMPTY
        balconies = [EMPTY for _ in range(self.num_balconies)]
        balconies[4] = HOLDING_ONION #define onion initial pos
        balconies[33] = HOLDING_ONION
        balconies[38] = HOLDING_ONION
        balconies[49] = HOLDING_ONION
        state = np.array([a0_row, a0_column, a1_row, a1_column, a0_heading, a1_heading, a0_hand, a1_hand, pan] + balconies)
        return state

    def unpack_joint_action(self, joint_action: int):
        a0_meaning, a1_meaning = self.joint_action_meanings[joint_action]
        a0 = self.action_meanings.index(a0_meaning)
        a1 = self.action_meanings.index(a1_meaning)
        return a0, a1

    def pack_joint_action(self, a0, a1):
        a0_meaning = self.action_meanings[a0]
        a1_meaning = self.action_meanings[a1]
        joint_action_meaning = a0_meaning, a1_meaning
        joint_action = self.joint_action_meanings.index(joint_action_meaning)
        return joint_action

    @staticmethod
    def get_cells_for(item: str, layout: np.ndarray):
        rows, columns = np.where(layout == item)
        num_cells = len(rows)
        assert len(columns) == num_cells
        cells = [(rows[i], columns[i]) for i in range(num_cells)]
        return cells

    @staticmethod
    def _valid_start_cells(layout: np.ndarray, player: int):
        walking_space = ' '
        player = str(player + 1)
        shared_cells = Overcooked.get_cells_for(walking_space, layout)
        individual_cells = Overcooked.get_cells_for(player, layout)
        return list(set(shared_cells + individual_cells))

    @staticmethod
    def _balconies(layout: np.ndarray):
        balconies = 'B'
        return Overcooked.get_cells_for(balconies, layout)

    @staticmethod
    def _pan_location(layout):
        pan = 'P'
        pans = Overcooked.get_cells_for(pan, layout)
        assert len(pans) == 1, "Layout can only contain one pan"
        return pans[0]

    def render_state(self, state, show=False):

        a0_row, a0_column, a1_row, a1_column, a0_heading, a1_heading, a0_hand, a1_hand, pan = state[:9]

        p1_pos = a0_column, a0_row
        p2_pos = a1_column, a1_row

        if a0_hand == HOLDING_NOTHING:
            p1_hand = None
        elif a0_hand == HOLDING_ONION:
            p1_hand = ObjectState("onion", p1_pos)
        elif a0_hand == HOLDING_DISH:
            p1_hand = ObjectState("dish", p1_pos)
        elif a0_hand == HOLDING_SOUP:
            p1_hand = SoupState(p1_pos, ingredients=[ObjectState("onion", p1_pos) for _ in range(THREE_ONIONS)])
        else:
            raise ValueError("Should be unreachable")

        if a1_hand == HOLDING_NOTHING:
            p2_hand = None
        elif a1_hand == HOLDING_ONION:
            p2_hand = ObjectState("onion", p2_pos)
        elif a1_hand == HOLDING_DISH:
            p2_hand = ObjectState("dish", p2_pos)
        elif a1_hand == HOLDING_SOUP:
            p2_hand = SoupState(p2_pos, ingredients=[ObjectState("onion", p2_pos) for _ in range(THREE_ONIONS)])
        else:
            raise ValueError("Should be unreachable")

        if a0_heading == NORTH:
            p1_dir = Direction.NORTH
        elif a0_heading == SOUTH:
            p1_dir = Direction.SOUTH
        elif a0_heading == WEST:
            p1_dir = Direction.WEST
        elif a0_heading == EAST:
            p1_dir = Direction.EAST
        else:
            raise ValueError("Should be unreachable")

        if a1_heading == NORTH:
            p2_dir = Direction.NORTH
        elif a1_heading == SOUTH:
            p2_dir = Direction.SOUTH
        elif a1_heading == WEST:
            p2_dir = Direction.WEST
        elif a1_heading == EAST:
            p2_dir = Direction.EAST
        else:
            raise ValueError("Should be unreachable")

        objects = {}
        balconies = state[9:]
        for b, balcony in enumerate(self.balconies):
            balcony_pos = balcony[1], balcony[0]
            balcony_contents = balconies[b]
            if balcony_contents == HOLDING_ONION:
                contents = ObjectState("onion", balcony_pos)
            elif balcony_contents == HOLDING_DISH:
                contents = ObjectState("dish", balcony_pos)
            elif balcony_contents == HOLDING_SOUP:
                contents = SoupState(balcony_pos, ingredients=[ObjectState("onion", balcony_pos) for _ in range(THREE_ONIONS)])
            elif balcony_contents == HOLDING_NOTHING:
                continue
            else:
                raise ValueError("Should be unreachable")
            objects[balcony_pos] = contents

        pan_pos = self.pan[1], self.pan[0]
        if pan > EMPTY and pan < SOUP_READY:
            objects[pan_pos] = SoupState(pan_pos, ingredients=[ObjectState("onion", pan_pos) for _ in range(pan)], cooking_tick=pan, cook_time=THREE_ONIONS)
        elif pan == SOUP_READY:
            objects[pan_pos] = SoupState(pan_pos, ingredients=[ObjectState("onion", pan_pos) for _ in range(THREE_ONIONS)], cooking_tick=THREE_ONIONS, cook_time=THREE_ONIONS)

        Recipe.configure({})
        vis = StateVisualizer()

        grid = np.copy(self.layout)
        grid[grid == 'B'] = 'X'
        p1 = PlayerState(p1_pos, p1_dir, p1_hand)
        p2 = PlayerState(p2_pos, p2_dir, p2_hand)
        players = (p1, p2)

        filename = None
        rgb_image = vis.render_clean_state(players, objects, grid, filename, show)
        return rgb_image


class SingleAgentWrapper(Wrapper):

    def __init__(self, env: Overcooked, teammate: Agent):
        self.teammate = teammate
        super().__init__(env)
        self.i = 0

    def reset(self):
        return super(SingleAgentWrapper, self).reset()

    def step(self, a0: int):
        self.i += 1
        state = self.env.state
        
        if self.i % 2 == 0:
            a1 = 5
        else:
            a1 = self.teammate.action(state)
        a1 = self.teammate.action(state)
        joint_action = self.env.pack_joint_action(a0, a1)
        next_state, reward, terminal, info = super().step(joint_action)
        timestep = yaaf.Timestep(state, a1, reward, next_state, terminal, info)
        teammate_info = self.teammate.reinforcement(timestep)
        info["teammate"] = teammate_info
        return next_state, reward, terminal, info
