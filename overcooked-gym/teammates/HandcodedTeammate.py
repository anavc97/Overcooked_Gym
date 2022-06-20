from numpy import ndarray
from yaaf import Timestep
from yaaf.agents import Agent
import numpy as np
from yaaf.policies import deterministic_policy

# TODO: _balcony contents indexes should be changed for regular overcooked
# TODO:changed: it picks the finite overcooked instead of the original
from finite_overcooked import Overcooked, HOLDING_ONION, SOUP_READY, HOLDING_NOTHING, HOLDING_DISH, ACTION_MEANINGS, SOUTH, \
    NORTH, EAST, HOLDING_SOUP
from search2 import A_star_search_overcooked_wrapper
from abc import ABC


def distance(x_0, y_0, x_1, y_1):
    return (x_0 - x_1) ** 2 + (y_0 - y_1) ** 2

class HandcodedTeammate(Agent, ABC):
    """Agent that fetches the items needed to make and serve the soup and places them in a balcony, so that a teammate
    can make the soup and serving without having to fetch anything."""

    def __init__(self, layout, index):
        super().__init__("Fetcher Teammate")
        self.layout = np.copy(layout)
        self.index = index
        self.balconies = Overcooked._balconies(self.layout)

    def _action_to_move_to(self, state, target, alternative_target=None, is_teammate_obstacle=True):
        """ Returns the action index that allows the agent to get closer to the target. If the agent is already by the
        target, the action will be to turn arround"""
        a0_row, a0_column, a1_row, a1_column, a0_heading, a1_heading, a0_hand, a1_hand, pan = state[:9]
        player = (a0_row, a0_column) if self.index == 0 else (a1_row, a1_column)
        player_heading = a0_heading if self.index == 0 else a1_heading

        walls = Overcooked.get_cells_for("X", self.layout)
        teammate = (a1_row, a1_column) if self.index == 0 else (a0_row, a0_column)
        pan = Overcooked.get_cells_for("P", self.layout)
        balconies = Overcooked.get_cells_for("B", self.layout)
        window = Overcooked.get_cells_for("S", self.layout)
        onion_supplies = Overcooked.get_cells_for("O", self.layout)
        dish_supplies = Overcooked.get_cells_for("D", self.layout)
        obstacles = walls + balconies + pan + window + onion_supplies + dish_supplies
        obstacles = obstacles +[teammate] if is_teammate_obstacle else obstacles
        action_meaning = A_star_search_overcooked_wrapper(player, target, set(obstacles), self.layout.shape[0],
                                                          self.layout.shape[1])

        # since the action stay does not equivalent in the overcooked interface, we simply make the agent change the
        # direction it is facing
        if action_meaning == "stay":
            if alternative_target is not None:
                return self._action_to_move_to(state, alternative_target)
            elif player_heading == NORTH:
                return ACTION_MEANINGS.index("down")
            else:
                return ACTION_MEANINGS.index("up")
        else:
            return ACTION_MEANINGS.index(action_meaning)

    def _action_to_move_to_empty_balcony(self, state):
        """ Returns the index of the action that allows the agent to get closer to the closest empty balcony.
        If there are no empty balconies, it returns None!"""

        a0_row, a0_column, a1_row, a1_column, a0_heading, a1_heading, a0_hand, a1_hand, pan = state[:9]
        a_row, a_column = (a0_row, a0_column) if self.index == 0 else (a1_row, a1_column)

        balcony_contents = self._balcony_contents(state)
        empty_balconies = []
        for index, contents in enumerate(balcony_contents):
            if contents == HOLDING_NOTHING:
                empty_balconies.append(self.balconies[index])

        min_dist = float('inf')
        chosen_balcony = None

        for b_row, b_column in empty_balconies:
            dist = distance(a_row, a_column, b_row, b_column)
            if min_dist > dist:
                min_dist = dist
                chosen_balcony = (b_row, b_column)

        return self._action_to_move_to(state, chosen_balcony)

    def _onion_required(self, state):
        """Returns True if there aren't enough onions in game to make a soup, that is, if the sum of the number of
        onions in the pan, the onions carried by the teammate and the onions in the balconies is smaller than 3. If the
        pan has a soup ready or there is no room in the balconies for more onions, we consider that no onions are
        required."""
        a0_row, a0_column, a1_row, a1_column, a0_heading, a1_heading, a0_hand, a1_hand, pan = state[:9]
        balcony_contents = self._balcony_contents(state)
        teammate_hand = a1_hand if self.index == 0 else a0_hand

        if pan == SOUP_READY:
            return False

        else:
            onions_in_pan = pan
            free_balconies = 0
            onions_on_balconies = 0
            for contents in balcony_contents:
                if contents == HOLDING_ONION:
                    onions_on_balconies += 1
                elif contents == HOLDING_NOTHING:
                    free_balconies += 1
            onions_on_teammate = int(teammate_hand == HOLDING_ONION)
            total_onions = onions_in_pan + onions_on_balconies + onions_on_teammate
            onion_required = total_onions < 3 and free_balconies > 0
            return onion_required

    def _dish_required(self, state):
        """Returns True if there is a soup ready in the pan, the agent is not holding a dish, there are no dishes in any
        balcony and there is room in the balcony for a dish."""
        a0_row, a0_column, a1_row, a1_column, a0_heading, a1_heading, a0_hand, a1_hand, pan = state[:9]
        balcony_contents = self._balcony_contents(state)
        teammate_hand = a1_hand if self.index == 0 else a0_hand

        #  if the teammate already has a dish
        if teammate_hand == HOLDING_DISH:
            return False

        onions_in_pan = min(pan, 3)
        onions_on_teammate = int(teammate_hand == HOLDING_ONION)
        onions_in_game = onions_in_pan + onions_on_teammate

        free_balconies = False
        for contents in balcony_contents:
            # if there is a dish in any of the balconies, no dish is required
            if contents == HOLDING_DISH:
                return False
            elif contents == HOLDING_NOTHING:
                free_balconies = True
            elif contents == HOLDING_ONION:
                onions_in_game += 1

        # if there is at least a free balcony, there is no dish on any balcony, and there are enough onions in game to
        # finish a soup, a dish is required
        return free_balconies and onions_in_game >= 3

    def _cell_faced(self, state):
        """Returns the row and column coordinates of the item the agent is facing. It could be an empty space, another
        agent, a balcony, the onion or the dish supply, etc."""
        a0_row, a0_column, a1_row, a1_column, a0_heading, a1_heading, a0_hand, a1_hand, pan = state[:9]
        row, col, direction = (a0_row, a0_column, a0_heading) if self.index == 0 else (a1_row, a1_column, a1_heading)
        if direction == NORTH:
            return row - 1, col
        elif direction == SOUTH:
            return row + 1, col
        elif direction == EAST:
            return row, col + 1
        else:
            return row, col - 1

    def _facing_teammate(self, state):
        a0_row, a0_column, a1_row, a1_column, a0_heading, a1_heading, a0_hand, a1_hand, pan = state[:9]
        row_faced, col_faced = self._cell_faced(state)
        row = a0_row if self.index == 1 else a1_row
        col = a0_column if self.index == 1 else a1_column
        return row_faced == row and col_faced == col

    def _facing_item(self, cell_faced_row, cell_faced_column, item_code):
        """Returns true if the agent is facing an item with code item_code."""
        return self.layout[cell_faced_row, cell_faced_column] == item_code

    def _facing_onion_supply(self, state):
        row, col = self._cell_faced(state)
        return self._facing_item(row, col, "O")

    def _facing_dish_supply(self, state):
        row, col = self._cell_faced(state)
        return self._facing_item(row, col, "D")

    def _facing_empty_balcony(self, state):
        row, col = self._cell_faced(state)
        balcony_contents = self._balcony_contents(state)
        if self._facing_item(row, col, "B"):
            index = self.balconies.index((row, col))
            return balcony_contents[index] == HOLDING_NOTHING
        else:
            return False

    def _facing_empty_space(self, state):
        row, col = self._cell_faced(state)
        return self._facing_item(row, col, " ")

    def _facing_wall(self, state):
        row, col = self._cell_faced(state)
        return self._facing_item(row, col, "X")

    def _facing_balcony_with_onion(self, state):
        row, col = self._cell_faced(state)
        balcony_contents = self._balcony_contents(state)
        if self._facing_item(row, col, "B"):
            index = self.balconies.index((row, col))
            return balcony_contents[index] == HOLDING_ONION
        else:
            return False

    def _facing_balcony_with_dish(self, state):
        row, col = self._cell_faced(state)
        balcony_contents = self._balcony_contents(state)
        if self._facing_item(row, col, "B"):
            index = self.balconies.index((row, col))
            return balcony_contents[index] == HOLDING_DISH
        else:
            return False

    def _facing_balcony_with_soup(self, state):
        row, col = self._cell_faced(state)
        balcony_contents = self._balcony_contents(state)
        if self._facing_item(row, col, "B"):
            index = self.balconies.index((row, col))
            return balcony_contents[index] == HOLDING_SOUP
        else:
            return False

    def _facing_delivery_window(self, state):
        row, col = self._cell_faced(state)
        return self._facing_item(row, col, "S")

    def _facing_pan(self, state):
        row, col = self._cell_faced(state)
        return self._facing_item(row, col, "P")

    def _free_balconies(self, state):
        """Returns True if and only if there are any empty balconies"""
        balcony_contents = self._balcony_contents(state)

        for contents in balcony_contents:
            if contents == HOLDING_NOTHING:
                return True
        return False

    def _action_to_move_to_balcony_with_onion(self, state):
        """ Returns the index of the action that allows the agent to get closer to the closest balcony holding an onion.
        If there are no such balconies, it returns None!"""

        a0_row, a0_column, a1_row, a1_column, a0_heading, a1_heading, a0_hand, a1_hand, pan = state[:9]
        a_row, a_column = (a0_row, a0_column) if self.index == 0 else (a1_row, a1_column)

        balcony_contents = self._balcony_contents(state)
        balconies_with_onions = []
        for index, contents in enumerate(balcony_contents):
            if contents == HOLDING_ONION:
                balconies_with_onions.append(self.balconies[index])

        if not balconies_with_onions:
            return None

        min_dist = float('inf')
        chosen_balcony = None

        for b_row, b_column in balconies_with_onions:
            dist = distance(a_row, a_column, b_row, b_column)
            if min_dist > dist:
                min_dist = dist
                chosen_balcony = (b_row, b_column)

        return self._action_to_move_to(state, chosen_balcony)

    def _action_to_move_to_balcony_with_dish(self, state):
        """ Returns the index of the action that allows the agent to get closer to the closest balcony holding an onion.
        If there are no such balconies, it returns None!"""

        a0_row, a0_column, a1_row, a1_column, a0_heading, a1_heading, a0_hand, a1_hand, pan = state[:9]
        a_row, a_column = (a0_row, a0_column) if self.index == 0 else (a1_row, a1_column)

        balcony_contents = self._balcony_contents(state)
        balconies_with_dish = []
        for index, contents in enumerate(balcony_contents):
            if contents == HOLDING_DISH:
                balconies_with_dish.append(self.balconies[index])

        if not balconies_with_dish:
            return None

        min_dist = float('inf')
        chosen_balcony = None

        for b_row, b_column in balconies_with_dish:
            dist = distance(a_row, a_column, b_row, b_column)
            if min_dist > dist:
                min_dist = dist
                chosen_balcony = (b_row, b_column)

        return self._action_to_move_to(state, chosen_balcony)

    def _balcony_contents(self, state):
        return Overcooked.balcony_contents(state)