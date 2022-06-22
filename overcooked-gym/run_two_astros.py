from PIL.Image import fromarray
from yaaf import Timestep
from yaaf.agents.dqn import MLPDQNAgent

from make_video import make_video
from overcooked2 import Overcooked, SingleAgentWrapper, LAYOUTS
from teammates.Astro import AstroHandcoded, AstroSmart, JOINT_ACTION_SPACE
import yaaf

if __name__ == '__main__':

    timesteps = 170
    save_episodes_as_video = True

    layout = "Lab"
    env = Overcooked(layout=layout)
    
    # Make sure you have the latest yaaf pip install git+https://github.com/jmribeiro/yaaf.git

    agent = AstroSmart(LAYOUTS[layout], 0, env=env)  # 1 - selects robot; 0 - selects human
    teammate = AstroSmart(LAYOUTS[layout], 1, env=env)
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

        if terminal or T == timesteps-1:

            print("EPISODE: ", episodes)    
            # Save video
            episodes += 1
            print(f"Episode {episodes} ended ({len(frames)} steps)")
            directory = f"resources_astro/episodes/episode_{episodes}_frames"
            for f, frame in enumerate(frames):
                image = fromarray(frame)
                yaaf.mkdir(directory)
                filename = f"{directory}/step_{f}.png"
                image.save(filename)

            if save_episodes_as_video:
                make_video(directory, f"resources_astro/episodes/episode_{episodes}.mp4")

            frames = [env.render(mode="silent")]
            
            break

        else:
            state = next_state
