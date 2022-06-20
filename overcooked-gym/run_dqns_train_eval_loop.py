import yaaf
from PIL.Image import fromarray
from gym.wrappers import TimeLimit
from yaaf import Timestep
from yaaf.agents.dqn import MLPDQNAgent

from make_video import make_video
from overcooked import Overcooked, SingleAgentWrapper


def train(agent, teammate, timesteps):
    agent.train()
    teammate.train()
    env = Overcooked()
    env = SingleAgentWrapper(env, teammate)
    state = env.reset()
    for t in range(timesteps):
        action = agent.action(state)
        next_state, reward, terminal, info = env.step(action)
        timestep = Timestep(state, action, reward, next_state, terminal, info)
        agent.reinforcement(timestep)
        state = env.reset() if terminal else next_state

def evaluate(agent, teammate, episodes, evaluation_directory):

    agent.eval()
    teammate.eval()
    env = Overcooked()
    env = SingleAgentWrapper(env, teammate)
    env = TimeLimit(env, max_episode_steps=1000)

    for episode in range(episodes):

        # Run episode
        state = env.reset()
        terminal = False
        accumulated_reward = 0.0
        t = 0
        frames = [env.render(mode="silent")]
        while not terminal:
            action = agent.action(state)
            next_state, reward, terminal, info = env.step(action)
            accumulated_reward += reward
            timestep = Timestep(state, action, reward, next_state, terminal, info)
            agent.reinforcement(timestep)
            frames.append(env.render(mode="silent"))
            state = next_state
            t += 1

        # Save episode
        frames_directory = f"{evaluation_directory}/frames/episode_{episode}"
        yaaf.mkdir(frames_directory)
        print(f"Episode {episode} ended ({t} steps)", flush=True)
        print(f"Accumulated Reward: {accumulated_reward}", flush=True)
        for f, frame in enumerate(frames):
            image = fromarray(frame)
            filename = f"{frames_directory}/step_{f}.png"
            image.save(filename)
        print(f"Making video", flush=True)
        make_video(f"{frames_directory}", f"{evaluation_directory}/episode_{episode}.mp4")

if __name__ == '__main__':

    total_training_timesteps = 150000
    timestep_evaluation_frequency = 25000
    evaluation_episodes = 3

    env = Overcooked()
    # Make sure you have the latest yaaf pip install git+https://github.com/jmribeiro/yaaf.git
    agent = MLPDQNAgent(env.num_features, env.num_actions)
    teammate = MLPDQNAgent(env.num_features, env.num_actions)
    del env

    num_phases = int(total_training_timesteps / timestep_evaluation_frequency)

    print(f"{total_training_timesteps} training timesteps, evaluating every {timestep_evaluation_frequency} ({num_phases} total train eval phases)", flush=True)
    for phase_id in range(num_phases):

        print(f"Phase {phase_id+1}/{num_phases}", flush=True)
        print(f"Training for {timestep_evaluation_frequency} timesteps", flush=True)
        train(agent, teammate, timestep_evaluation_frequency)

        print(f"Evaluating for {evaluation_episodes} episodes", flush=True)
        directory = f"resources/2DQN Evaluation/phase_{phase_id + 1}"
        evaluate(agent, teammate, evaluation_episodes, directory)
