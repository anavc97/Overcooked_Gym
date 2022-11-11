from numpy import ndarray
import numpy as np
import pickle
import itertools
import time, random



ACTION_SPACE = tuple(range(6))
layout_name = "Lab2"
mdp_ind = np.load('/home/anavc/Overcooked_Gym/overcooked-gym/mdp_ind_{}.npy'.format(layout_name), allow_pickle=True)
p_joint =  np.load('/home/anavc/Overcooked_Gym/overcooked-gym/policy_{}.npy'.format(layout_name), allow_pickle=True)
JOINT_ACTION_SPACE = list(itertools.product(ACTION_SPACE, repeat=2))

def check_index(state):
    for ind, val in enumerate(mdp_ind):
        if np.array_equal(np.array(val),np.array(state)):
            return ind

print(p_joint[check_index([3, 3, 0, 0, 0, 0])])
#a_joint = JOINT_ACTION_SPACE[np.random.choice(range(len(JOINT_ACTION_SPACE)), p=self.p_joint[self.check_index([0,0,0,0,0,0])])]
'''
print("Best action: ", JOINT_ACTION_SPACE[np.argmax(policy[mmdp.state_index([0,0,0,0,0,0])])])
print("Prob: ", policy[mmdp.state_index([0,0,0,0,0,0])])
print("Best action: ", JOINT_ACTION_SPACE[np.argmax(policy[mmdp.state_index([5,5,0,0,0,0])])])
print("Prob: ", policy[mmdp.state_index([5,5,0,0,0,0])])
print("Best action: ", JOINT_ACTION_SPACE[np.argmax(policy[mmdp.state_index([5,5,0,0,0,1])])])
print("Prob: ", policy[mmdp.state_index([5,5,0,0,0,1])])
print("Best action: ", JOINT_ACTION_SPACE[np.argmax(policy[mmdp.state_index([5,5,0,0,0,2])])])
print("Prob: ", policy[mmdp.state_index([5,5,0,0,0,2])])

'''