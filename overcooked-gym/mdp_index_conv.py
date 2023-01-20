from numpy import ndarray
import numpy as np
import pickle

MAP = "Lab1"

with open("/home/anavc/Overcooked_Gym/overcooked-gym/mmdp.pickle", "rb") as f:
    mdp = pickle.load(f)

print("Loaded MDP")
np.save("/home/anavc/Overcooked_Gym/overcooked-gym/policy_{}".format(MAP), mdp.policy)

def generate_states():

    def valid_state(state):

        p2, ball_status = state[1], state[2:]

        if (ball_status == 1).sum() > 1:
            # Human can't hold two balls
            return False
        else:
            return True

    num_nodes = 8
    ball_statuses = 3   # Ground, Held, Disposed
    states = [
        np.array([p1, p2, b1, b2, b3, b4])
        for p1 in range(num_nodes)
        for p2 in range(num_nodes)
        for b1 in range(ball_statuses)
        for b2 in range(ball_statuses)
        for b3 in range(ball_statuses)
        for b4 in range(ball_statuses)
    ]

    return [state for state in states if valid_state(state)]

states = generate_states()
mdp_ind = [None]*len(states)

for s_mdp in states:
    print("Registering state: ", s_mdp)
    mdp_ind[mdp.state_index(s_mdp)] = s_mdp

print(len(mdp_ind))

np.save("/home/anavc/Overcooked_Gym/overcooked-gym/mdp_ind_{}".format(MAP), mdp_ind)
