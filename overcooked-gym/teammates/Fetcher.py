from numpy import ndarray
from yaaf import Timestep
from yaaf.policies import deterministic_policy
from overcooked import Overcooked, HOLDING_ONION, HOLDING_NOTHING, HOLDING_DISH, ACTION_MEANINGS
from teammates.HandcodedTeammate import HandcodedTeammate


class Fetcher(HandcodedTeammate):
    """Agent that fetches the items needed to make and serve the soup and places them in a balcony, so that a teammate
    can make the soup and serving without having to fetch anything."""

    def __init__(self, layout, index):
        super().__init__(layout, index)

    def policy(self, state: ndarray):
        a0_row, a0_column, a1_row, a1_column, a0_heading, a1_heading, a0_hand, a1_hand, pan = state[:9]
        hand = a0_hand if self.index == 0 else a1_hand
        onion_supply = Overcooked.get_cells_for("O", self.layout)[0]
        dish_supply = Overcooked.get_cells_for("D", self.layout)[0]

        if hand == HOLDING_NOTHING:
            if self._onion_required(state):
                if self._facing_onion_supply(state):
                    action = ACTION_MEANINGS.index("act")
                else:
                    action = self._action_to_move_to(state, onion_supply)

            elif self._dish_required(state):
                if self._facing_dish_supply(state):
                    action = ACTION_MEANINGS.index("act")
                else:
                    action = self._action_to_move_to(state, dish_supply)

            # went the agent has nothing to do, it moves to the onion supply and stays put.
            else:
                action = self._action_to_move_to(state, onion_supply)
        else:
            if hand == HOLDING_ONION:
                if self._onion_required(state):
                    if self._facing_empty_balcony(state):
                        action = ACTION_MEANINGS.index("act")
                    else:
                        action = self._action_to_move_to_empty_balcony(state)

                # if the onion is not needed, put it back to the supply
                elif self._facing_onion_supply(state):
                    action = ACTION_MEANINGS.index("act")

                else:
                    action = self._action_to_move_to(state, onion_supply)

            elif hand == HOLDING_DISH:

                if self._dish_required(state):
                    if self._facing_empty_balcony(state):
                        action = ACTION_MEANINGS.index("act")

                    else:
                        action = self._action_to_move_to_empty_balcony(state)

                elif self._facing_dish_supply(state):
                    action = ACTION_MEANINGS.index("act")

                else:
                    action = self._action_to_move_to(state, onion_supply)

            # this would be the case in which the agent has the soup. It should never happen
            else:
                #TODO: test and garantee this is never reached
                pass

        return deterministic_policy(action, len(ACTION_MEANINGS))

    def _reinforce(self, timestep: Timestep):
        pass

    ####################################################################################################################
    #                                              AUXILIARY METHODS
    ####################################################################################################################
