from numpy import ndarray
from yaaf import Timestep

from teammates.HandcodedTeammate import HandcodedTeammate
from yaaf.policies import deterministic_policy
import random as rnd
from overcooked import HOLDING_NOTHING, HOLDING_ONION, HOLDING_DISH, HOLDING_SOUP, SOUP_READY, ACTION_MEANINGS


class RandomyMovingTeammate(HandcodedTeammate):
    """Agent that fetches the items needed to make and serve the soup and places them in a balcony, so that a teammate
    can make the soup and serving without having to fetch anything."""

    def __init__(self, layout, index):
        super().__init__(layout, index)
        self._moves_left_count = 0
        self._last_action = None

    def policy(self, state: ndarray):
        # if the last action is still supposed to be performed, act according to it
        if self._moves_left_count > 0:
            if self._facing_empty_space(state) and not self._facing_teammate(state):
                self._moves_left_count -= 1
                return deterministic_policy(self._last_action, len(ACTION_MEANINGS))
            else:
                self._moves_left_count = 0

        # Check out whether the agent can or not use act. If it can, it will use it with probability 0.5
        if self._last_action != 4 and self._can_interact(state):
            p = rnd.uniform(0, 1)
            if p < 0.5:
                self._last_action = 4
                return deterministic_policy(ACTION_MEANINGS.index("act"), len(ACTION_MEANINGS))

        # if act was not chosen, chose between one of the movement actions
        action = rnd.randrange(0, 4)
        self._last_action = action
        self._moves_left_count = rnd.randrange(0, 4)

        return deterministic_policy(action, len(ACTION_MEANINGS))

    def _reinforce(self, timestep: Timestep):
        pass

    def _can_interact(self, state):
        a0_row, a0_column, a1_row, a1_column, a0_heading, a1_heading, a0_hand, a1_hand, pan = state[:9]
        hand = a0_hand if self.index == 0 else a1_hand

        # if the agent is not holding anything, it can only pickup onions and dishes, as long as they are in from of it
        if hand == HOLDING_NOTHING:
            if self._facing_dish_supply(state) or self._facing_onion_supply(state) or \
                    self._facing_balcony_with_soup(state) or self._facing_balcony_with_onion(state) or \
                    self._facing_balcony_with_dish(state):
                return True
            else:
                return False

        # regardless of what the agent is holding, as long as it is holding something, it can interact with an empty
        # balcony
        elif self._facing_empty_balcony(state):
            return True

        # if the agent is holding an onion, it can always put it back onto the onion supply or in the pan, if there are
        # not enough onions there yet
        elif hand == HOLDING_ONION:
            if self._facing_onion_supply(state):
                return True
            elif pan < 3 and self._facing_pan(state):
                return True

        # a dish can always be placed back onto the dish supply or used to get a soup from the pan, if the soup is ready
        elif hand == HOLDING_DISH:
            if self._facing_dish_supply(state):
                return True
            elif pan == SOUP_READY and self._facing_pan(state):
                return True

        # if the agent is holding a soup and not facing an empty balcony, it can only interact with the delivery window
        elif hand == HOLDING_SOUP and self._facing_delivery_window(state):
            return True

        return False