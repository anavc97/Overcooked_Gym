from numpy import ndarray
import numpy as np
import pickle
import itertools
import time, random


with open("/home/anavc/Overcooked_Gym/overcooked-gym/mmdp.pickle", "rb") as f:
    mmdp = pickle.load(f)

print("INIT POLICY")
#with open("/home/anavc/overcooked-gym/policy.pickle", "rb") as f:
#    policy = pickle.load(f)

f = open("/home/anavc/Overcooked_Gym/overcooked-gym/policy.pickle","wb")
pickle.dump(mmdp.policy, f)

ACTION_SPACE = tuple(range(6))

JOINT_ACTION_SPACE = list(itertools.product(ACTION_SPACE, repeat=2))
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