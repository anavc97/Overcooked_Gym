from yaaf.agents import HumanAgent

from overcooked import Overcooked, LAYOUTS, SingleAgentWrapper
from teammates.Fetcher import Fetcher
from teammates.CookNServer import CookNServer
from teammates.RandomlyMovingTeammate import RandomyMovingTeammate

if __name__ == '__main__':

    env = Overcooked(layout="kitchen")
    agent = Fetcher(LAYOUTS["kitchen"], index=0)
    teammate = RandomyMovingTeammate(LAYOUTS["kitchen"], index=1)
    env = SingleAgentWrapper(env, teammate)

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

    print(f"FINAL REWARD: {total_reward}")

