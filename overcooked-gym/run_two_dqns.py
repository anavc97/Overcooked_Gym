from PIL.Image import fromarray
from yaaf import Timestep
from yaaf.agents.dqn import MLPDQNAgent

from make_video import make_video
from overcooked import Overcooked, SingleAgentWrapper
import yaaf

if __name__ == '__main__':

    timesteps = 50000
    save_episodes_as_video = True

    env = Overcooked()
    
    # Make sure you have the latest yaaf pip install git+https://github.com/jmribeiro/yaaf.git

    agent = MLPDQNAgent(env.num_features, env.num_actions)
    teammate = MLPDQNAgent(env.num_features, env.num_actions)

    env = SingleAgentWrapper(env, teammate)
    state = env.reset()
    episodes = 0

    frames = [env.render(mode="silent")]

    for T in range(timesteps):

        action = agent.action(state)
        next_state, reward, terminal, info = env.step(action)
        timestep = Timestep(state, action, reward, next_state, terminal, info)
        info = agent.reinforcement(timestep)

        frames.append(env.render(mode="silent"))

        if terminal:

            # Save video
            episodes += 1
            print(f"Episode {episodes} ended ({len(frames)} steps)")
            directory = f"resources/episodes/episode_{episodes}_frames"
            for f, frame in enumerate(frames):
                image = fromarray(frame)
                yaaf.mkdir(directory)
                filename = f"{directory}/step_{f}.png"
                image.save(filename)

            if save_episodes_as_video:
                make_video(directory, f"resources/episodes/episode_{episodes}.mp4")

            state = env.reset()

            frames = [env.render(mode="silent")]

        else:
            state = next_state
