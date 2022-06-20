from yaaf.agents import HumanAgent, RandomAgent

from finite_overcooked import Overcooked, LAYOUTS, SingleAgentWrapper
from teammates.Fetcher import Fetcher

if __name__ == '__main__':

    env = Overcooked(layout="small_r", max_timesteps=500)
    #agent = HumanAgent(action_meanings=env.action_meanings)
    agent = RandomAgent(env.num_actions)

    teammate = Fetcher(LAYOUTS["small_r"], index=1)
    env = SingleAgentWrapper(env, teammate)

    state = env.reset()
    terminal = False
    env.render(mode="plt")
    total_reward = 0
    while not terminal:
        action = agent.action(state)
        next_state, reward, terminal, info = env.step(action)
        state = next_state
        total_reward += reward
        if reward > 0:
            print(reward)
            env.render(mode="plt")
        #env.render(mode="plt")
        #print(reward)
        #print(next_state)
        #print(f"lenght of state: {len(next_state)}")

    print(f"Final reward: {total_reward}")
