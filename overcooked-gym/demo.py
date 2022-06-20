import numpy as np

from yaaf.agents import RandomAgent
from overcooked import Overcooked, SingleAgentWrapper

def run(agent, env, steps):

    state = env.reset()
    env.render(mode="plt")
    for step in range(steps):
        action = agent.action(state)
        next_state, reward, terminal, info = env.step(action)
        env.render(mode="plt")
        state = next_state if not terminal else env.reset()

if __name__ == '__main__':

    small = np.array([
            ["X", "X", "X", "P", "X"],
            ["O", "1", "B", "2", "X"],
            ["D", "1", "B", "2", "X"],
            ["X", "X", "X", "S", "X"],
    ])

    multi_agent = True
    layout = small

    if multi_agent:

        # Multi-Agent Interface
        env = Overcooked(layout)
        agent = RandomAgent(env.num_joint_actions)
        run(agent, env, steps=1000)

    else:

        # "Single-Agent" Interface
        env = Overcooked(layout)
        agent = RandomAgent(env.num_actions)
        teammate = RandomAgent(env.num_actions)
        env = SingleAgentWrapper(env, teammate)
        run(agent, env, steps=1000)
