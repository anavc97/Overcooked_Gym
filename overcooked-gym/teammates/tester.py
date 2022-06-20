from numpy import ndarray
import numpy as np
import pickle
import time, random


with open("/home/anavc/overcooked-gym/mmdp.pickle", "rb") as f:
    mmdp = pickle.load(f)

print(mmdp.P)
