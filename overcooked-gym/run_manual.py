from yaaf.agents import HumanAgent
from overcooked import Overcooked, SingleAgentWrapper

if __name__ == '__main__':

    single_agent = False
    render = True
    render_mode = "window"  # Available: window (pop-up) and matplotlib (plt.imshow). Video rendering planned for the future.

    env = Overcooked(layout="kitchen")
    agent = HumanAgent(action_meanings=env.action_meanings, name="Player 1")
    teammate = HumanAgent(action_meanings=env.action_meanings, name="Player 2")
    env = SingleAgentWrapper(env, teammate)
    state = env.reset()
    env.render(render_mode)
    terminal = False
    while not terminal:
        print("Blue hat goes first")
        print(f"State: {state}")
        action = agent.action(state)
        next_state, reward, terminal, info = env.step(action)
        env.render(render_mode)
        print(f"State: {next_state}")
        print(f"Reward: {reward}")
