from yaaf.agents import HumanAgent

from overcooked import Overcooked, LAYOUTS, SingleAgentWrapper
from teammates.Fetcher import Fetcher

if __name__ == '__main__':

    env = Overcooked(layout="kitchen")
    agent = HumanAgent(action_meanings=env.action_meanings)
    teammate = Fetcher(LAYOUTS["kitchen"], index=1)
    env = SingleAgentWrapper(env, teammate)

    state = env.reset()
    terminal = False
    env.render(mode="plt")
    while not terminal:
        action = agent.action(state)
        next_state, reward, terminal, info = env.step(action)
        state = next_state
        env.render(mode="plt")
        print(reward)
        print(next_state)
