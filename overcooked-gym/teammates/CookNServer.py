from numpy import ndarray
from yaaf import Timestep
from yaaf.policies import deterministic_policy
from overcooked import Overcooked, HOLDING_ONION, HOLDING_NOTHING, HOLDING_DISH, ACTION_MEANINGS, HOLDING_SOUP, SOUP_READY, LAYOUTS
from teammates.HandcodedTeammate import HandcodedTeammate


class CookNServer(HandcodedTeammate):
    """Agent that puts the onions on the pan, as long as they are placed on the balcony first. When the soup is ready,
    it serves it, as soon as a plate is placed in the balcony. It only interacts with objects placed in the balcony or
    in the pan. It should be a good teammate for the fetcher."""

    def __init__(self, layout, index, resting_cell=None):
        super().__init__(layout, index)
        if resting_cell is None:
            resting_cell = Overcooked.get_cells_for("P", self.layout)[0]
        self._resting_cell = resting_cell

    def policy(self, state: ndarray):
        a0_row, a0_column, a1_row, a1_column, a0_heading, a1_heading, a0_hand, a1_hand, pan = state[:9]
        hand = a0_hand if self.index == 0 else a1_hand
        stove = Overcooked.get_cells_for("P", self.layout)[0]
        delivery_window = Overcooked.get_cells_for("S", self.layout)[0]
        dish_supply = Overcooked.get_cells_for("D", self.layout)[0]

        if hand == HOLDING_SOUP:
            if self._facing_delivery_window(state):
                action = ACTION_MEANINGS.index("act")
            else:
                action = self._action_to_move_to(state, delivery_window)

        elif hand == HOLDING_DISH:
            if pan == SOUP_READY:
                if self._facing_pan(state):
                    action = ACTION_MEANINGS.index("act")
                else:
                    action = self._action_to_move_to(state, stove)
            else:
                if self._facing_empty_balcony(state) or self._facing_dish_supply(state):
                    action = ACTION_MEANINGS.index("act")
                elif self._free_balconies(state):
                    action = self._action_to_move_to_empty_balcony(state)
                else:
                    action = self._action_to_move_to(state, dish_supply)

        elif hand == HOLDING_ONION:
            if pan < 3:
                if self._facing_pan(state):
                    action = ACTION_MEANINGS.index("act")
                else:
                    action = self._action_to_move_to(state, stove)
            else:
                if self._facing_empty_balcony(state) or self._facing_onion_supply(state):
                    action = ACTION_MEANINGS.index("act")
                elif self._free_balconies(state):
                    action = self._action_to_move_to_empty_balcony(state)
                else:
                    action = self._action_to_move_to(state, dish_supply)

        else:
            if pan < 3:
                if self._facing_balcony_with_onion(state):
                    action = ACTION_MEANINGS.index("act")
                else:
                    action = self._action_to_move_to_balcony_with_onion(state)
            elif pan == SOUP_READY:
                if self._facing_balcony_with_dish(state):
                    action = ACTION_MEANINGS.index("act")
                else:
                    action = self._action_to_move_to_balcony_with_dish(state)
            else:
                action = None

            action = self._action_to_move_to(state, self._resting_cell, alternative_target=delivery_window) if action \
                                                                                                               is None \
                else action

        return deterministic_policy(action, len(ACTION_MEANINGS))

    def _reinforce(self, timestep: Timestep):
        pass

