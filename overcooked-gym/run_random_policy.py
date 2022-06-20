from yaaf.agents import RandomAgent

from overcooked import Overcooked, SingleAgentWrapper

if __name__ == '__main__':

    single_agent = True
    render = True
    render_mode = "plt"  # Available: window (pop-up) and matplotlib (plt.imshow). Video rendering planned for the future.

    if single_agent:
        # Option 1
        env = Overcooked()
        agent = RandomAgent(env.num_actions)
        teammate = RandomAgent(env.num_actions)
        env = SingleAgentWrapper(env, teammate)
        state = env.reset()
        env.render(render_mode)
        terminal = False
        while not terminal:
            action = agent.action(state)
            next_state, reward, terminal, info = env.step(action)
            env.render(render_mode)

    else:
        # Option 2
        env = Overcooked()
        agent = RandomAgent(env.num_joint_actions)
        state = env.reset()
        env.render(render_mode)
        terminal = False
        while not terminal:
            joint_action = agent.action(state)
            next_state, reward, terminal, info = env.step(joint_action)
            env.render(render_mode)


