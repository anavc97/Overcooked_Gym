# import pygame module in this program
from teammates.Astro import ACTION_MEANINGS_MDP 
import itertools
import pickle
from collections import Counter


log = []
LVL=1
LOG_NR = 301
log_file = f"/home/anavc/Overcooked_Gym/overcooked-gym/logfiles/logfile_{LOG_NR}_lvl{LVL}.pickle"
ACTION_SPACE = tuple(range(len(ACTION_MEANINGS_MDP[LVL-1])))
JOINT_ACTION_SPACE = list(itertools.product(ACTION_SPACE, repeat=2))
print("READING FROM: ", log_file)

class LogFrame:
  def __init__(self, timestep:int , state_env, state_mdp:list, action_env:tuple, action_mdp:tuple, onion_time:float, game_time:float):
    self.timestep = timestep
    self.state_env = state_env
    self.state_mdp = state_mdp
    self.action_env = action_env
    self.action_mdp = action_mdp
    self.onion_time = onion_time
    self.game_time = game_time

def count_occurrences(arrays):
    return Counter([tuple(array) for array in arrays])

# infinite loop

with open(log_file, "rb") as f:
    log = pickle.load(f)

s_env = []
s_mdp = []

for logframe in log:
   print(logframe.state_env)
   s_env.append(logframe.state_env)
   s_mdp.append(logframe.state_mdp)

print("STATE ENV: ", count_occurrences(s_env))
print("STATE MDP: ", count_occurrences(s_mdp))

#for l in log:

