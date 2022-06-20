from heapq import *

from overcooked import Overcooked, LAYOUTS

STAY = (0, 0)
RIGHT = (1, 0)
LEFT = (-1, 0)
DOWN = (0, 1)
UP = (0, -1)

MAP = {
    UP: "up",
    DOWN: "down",
    LEFT: "left",
    RIGHT: "right",
    STAY: "stay"
}

def agent_directions():
    return RIGHT, LEFT, DOWN, UP

def direction(source, target, w, h):

    dx_forward = (target[0] - source[0]) % w
    dx_backward = (source[0] - target[0]) % w

    dy_forward = (target[1] - source[1]) % h
    dy_backward = (source[1] - target[1]) % h

    if dx_forward < dx_backward: return 1, 0
    elif dx_backward < dx_forward: return -1, 0
    elif dy_forward < dy_backward: return 0, 1
    elif dy_backward < dy_forward: return 0, -1
    else: return 0, 0


def neighbors(position, world_size):
    directions = agent_directions()
    result = []
    for direction in directions: result.append(move(position, direction, world_size))
    return result


def distance(source, target, w, h):
    dx = min((source[0] - target[0]) % w, (target[0] - source[0]) % w)
    dy = min((source[1] - target[1]) % h, (target[1] - source[1]) % h)
    return dx, dy

def move(entity_position, direction, world_size):

    w = world_size[0]
    h = world_size[1]

    x = entity_position[0]
    y = entity_position[1]

    dx = direction[0]
    dy = direction[1]

    new_position = (x + dx) % w, (y + dy) % h

    return new_position


class Node(object):
    def __init__(self, position, parent, cost, heuristic):
        self.position = position
        self.parent = parent
        self.cost = cost
        self.heuristic = heuristic

    def __lt__(self, other):
        return self.cost + self.heuristic < other.cost + other.heuristic

    def __hash__(self):
        return self.position.__hash__()

    def __eq__(self, other):
        return self.position == other.position


def A_star_search(source, obstacles, target, world_size):

    """A* Search for Pursuit"""

    if source == target:
        return (0, 0), 0

    w, h = world_size
    obstacles = obstacles - {target}

    def heuristic(position):
        return sum(distance(source, position, w, h))

    # each item in the queue contains (heuristic+cost, cost, position, parent)
    initial_node = Node(source, None, 0, heuristic(source))
    queue = [Node(n, initial_node, 1, sum(distance(n, target, w, h)))
             for n in neighbors(source, world_size) if n not in obstacles]

    #heapify(queue)
    visited = set()
    visited.add(source)
    current = initial_node

    while len(queue) > 0:
        #current = heappop(queue)
        current = queue.pop(0)

        if current.position in visited:
            continue

        visited.add(current.position)

        if current.position == target:
            break

        for position in neighbors(current.position, world_size):
            if position not in obstacles:
                new_node = Node(position, current, current.cost + 1, heuristic(position))
                #heappush(queue, new_node)
                queue.append(new_node)

    if target not in visited:
        #return None, w * h
        return (0, 0), 0

    i = 1
    while current.parent != initial_node:
        current = current.parent
        i += 1

    return direction(source, current.position, h, w), i

def non_toroidal_A_star_search(source, target, obstacles, world_size):
    new_world_size = (world_size[0] + 1, world_size[1] + 1)
    vertical_barrier = [(world_size[0], y) for y in range(new_world_size[1])]
    horizontal_barrier = [(x, world_size[1]) for x in range(new_world_size[0])]
    new_obstacles = list(obstacles) + vertical_barrier + horizontal_barrier
    new_obstacles = set(new_obstacles)
    return A_star_search(source, new_obstacles, target, new_world_size)

def A_star_search_overcooked_wrapper(source, target, obstacles, num_rows, num_columns):
    world_size = (num_columns, num_rows)
    source = source[1], source[0]
    target = target[1], target[0]
    new_obstacles = set([(obs[1], obs[0]) for obs in obstacles])
    direction, _ = non_toroidal_A_star_search(source, target, new_obstacles, world_size)
    action_meaning = MAP[direction]
    return action_meaning


def example():

    "Example for overcooked"

    player = (3, 3)
    onion = (4, 3)
    layout = LAYOUTS["kitchen"]
    walls = Overcooked.get_cells_for("X", layout)
    teammate = (1, 1)
    pan = Overcooked.get_cells_for("P", layout)
    balconies = Overcooked.get_cells_for("B", layout)
    window = Overcooked.get_cells_for("S", layout)
    onion_supplies = Overcooked.get_cells_for("O", layout)
    dish_supplies = Overcooked.get_cells_for("D", layout)
    obstacles = walls + [teammate] + balconies + pan + window + onion_supplies + dish_supplies
    print(A_star_search_overcooked_wrapper(player, onion, set(obstacles), layout.shape[0], layout.shape[1]))

