from yaaf.agents import HumanAgent, RandomAgent

from finite_overcooked import Overcooked, LAYOUTS, SingleAgentWrapper
from teammates.Fetcher import Fetcher
from teammates.CookNServer import CookNServer
import numpy as np

if __name__ == '__main__':
    layout = "simple_kitchen"
    #layout = "small_r"
    env = Overcooked(layout=layout, max_timesteps=1500, rewards=(0,1,2,10))
    agent = CookNServer(LAYOUTS[layout], index=0)
    #agent = RandomAgent(5)
    teammate = Fetcher(LAYOUTS[layout], index=1)
    env = SingleAgentWrapper(env, teammate)

    rewards = []
    n = 1
    for i in range(n):
        state = env.reset()
        terminal = False
        total_reward = 0
        #env.render(mode="plt")
        while not terminal:
            action = agent.action(state)
            next_state, reward, terminal, info = env.step(action)
            state = next_state
            #env.render(mode="plt")
            total_reward += reward
            #print(reward)
            #print(next_state)
        rewards.append(total_reward)
    rewards = np.asarray(rewards)
    print(f"avg: {np.average(rewards)}")
    print(f"std: {np.std(rewards)}")
    print(f"max: {np.max(rewards)}")

