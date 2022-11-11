from numpy import ndarray
import numpy as np
import pickle

with open("/home/anavc/Overcooked_Gym/overcooked-gym/mmdp.pickle", "rb") as f:
    mmdp = pickle.load(f)

np.save("/home/anavc/Overcooked_Gym/overcooked-gym/policy", mmdp.policy)
np.save("/home/anavc/Overcooked_Gym/overcooked-gym/state_index", mmdp.state_index)

p = np.load('/home/anavc/Overcooked_Gym/overcooked-gym/policy.npy')
print(p)

