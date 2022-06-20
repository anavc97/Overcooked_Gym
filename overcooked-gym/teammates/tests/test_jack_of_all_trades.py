from yaaf.agents import HumanAgent

from overcooked import Overcooked, LAYOUTS, SingleAgentWrapper
from teammates.Fetcher import Fetcher
from teammates.JackOfAllTrades import JackOfAllTrades
from teammates.CookNServer import CookNServer

if __name__ == '__main__':

    env = Overcooked(layout="kitchen")
    # agent = HumanAgent(action_meanings=env.action_meanings)
    agent = JackOfAllTrades(LAYOUTS["kitchen"], index=0)
    teammate = JackOfAllTrades(LAYOUTS["kitchen"], index=1)
    env = SingleAgentWrapper(env, teammate)

    state = env.reset()
    terminal = False
    #env.render(mode="plt")
    tot_reward = 0
    while not terminal:
        action = agent.action(state)
        next_state, reward, terminal, info = env.step(action)
        state = next_state
        #env.render(mode="plt")
        tot_reward += reward
        print(reward)
    print(f"Final reward: {tot_reward}")
