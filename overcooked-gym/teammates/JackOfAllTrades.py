from numpy import ndarray
from yaaf import Timestep
from yaaf.policies import deterministic_policy
from overcooked import Overcooked, HOLDING_ONION, HOLDING_NOTHING, HOLDING_DISH, ACTION_MEANINGS, HOLDING_SOUP, SOUP_READY
from teammates.HandcodedTeammate import HandcodedTeammate


class JackOfAllTrades(HandcodedTeammate):
    """Agent that fetches the ingredients, puts the onions in the pan, gets the soup when ready and serves the soup. A
    true Jack of all trades"""

    def __init__(self, layout, index):
        super().__init__(layout, index)

    def policy(self, state: ndarray):
        a0_row, a0_column, a1_row, a1_column, a0_heading, a1_heading, a0_hand, a1_hand, pan = state[:9]
        hand = a0_hand if self.index == 0 else a1_hand
        stove = Overcooked.get_cells_for("P", self.layout)[0]
        delivery_window = Overcooked.get_cells_for("S", self.layout)[0]
        dish_supply = Overcooked.get_cells_for("D", self.layout)[0]
        onion_supply = Overcooked.get_cells_for("O", self.layout)[0]

        if hand == HOLDING_SOUP:
            if self._facing_delivery_window(state):
                action = ACTION_MEANINGS.index("act")
            else:
                action = self._action_to_move_to(state, delivery_window, alternative_target=stove)

        elif pan == SOUP_READY:
            if hand == HOLDING_ONION:
                if self._facing_empty_balcony(state) or self._facing_onion_supply(state):
                    action = ACTION_MEANINGS.index("act")
                else:
                    action = self._action_to_move_to_empty_balcony(state)
                    if action is None:
                        action = self._action_to_move_to(state, onion_supply, alternative_target=dish_supply)

            elif hand == HOLDING_DISH:
                if self._facing_pan(state):
                    action = ACTION_MEANINGS.index("act")
                else:
                    # sometimes handcoded agents block each other targets. When that happens, the best is to have an
                    # alternative target in order to get out of the way
                    action = self._action_to_move_to(state, stove, alternative_target=delivery_window)

            elif self._facing_balcony_with_dish(state) or self._facing_dish_supply(state):
                action = ACTION_MEANINGS.index("act")

            else:
                action = self._action_to_move_to_balcony_with_dish(state)
                if action is None:
                    # sometimes handcoded agents block each other targets. When that happens, the best is to have an
                    # alternative target in order to get out of the way
                    action = self._action_to_move_to(state, dish_supply, onion_supply)

        # soup not ready
        else:
            if hand == HOLDING_ONION:
                if self._facing_pan(state):
                    action = ACTION_MEANINGS.index("act")
                else:
                    action = self._action_to_move_to(state, stove, alternative_target=delivery_window)
                    if action is None:
                        action = self._action_to_move_to(state, dish_supply, alternative_target=onion_supply)

            elif hand == HOLDING_DISH:
                if self._facing_dish_supply(state) or self._facing_empty_balcony(state):
                    action = ACTION_MEANINGS.index("act")
                else:
                    action = self._action_to_move_to_empty_balcony(state)

            # holding nothing
            else:
                """if self._onion_required(state):"""
                if self._facing_onion_supply(state) or self._facing_balcony_with_onion(state):
                    action = ACTION_MEANINGS.index("act")
                else:
                    action = self._action_to_move_to_balcony_with_onion(state)
                    if action is None:
                        action = self._action_to_move_to(state, onion_supply, alternative_target=dish_supply)
                """elif self._dish_required(state):
                    if self._facing_dish_supply(state) or self._facing_balcony_with_dish(state):
                        action = ACTION_MEANINGS.index("act")
                    else:
                        action = self._action_to_move_to_balcony_with_dish(state)
                        if action is None:
                            action = self._action_to_move_to(state, dish_supply, alternative_target=onion_supply)
                else:
                    action = self._action_to_move_to(state, onion_supply, alternative_target=dish_supply)"""

        return deterministic_policy(action, len(ACTION_MEANINGS))

    def _reinforce(self, timestep: Timestep):
        pass

