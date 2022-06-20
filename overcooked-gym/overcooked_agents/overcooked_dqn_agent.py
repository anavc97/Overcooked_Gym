from dqn.agents.dqn_agent import DQNAgent
from dqn.policies.train_eval_policy import TrainEvalPolicy
from make_video import make_video
from PIL.Image import fromarray
import numpy as np
import os
import torch
import torch.nn.functional as F


class OvercookedDQNAgent(DQNAgent):
    def __init__(self, env, replay, n_actions, net_type, net_parameters, minibatch_size=32,
                 optimizer=torch.optim.RMSprop, C=10_000, update_frequency=1, gamma=0.99, loss=F.mse_loss,
                 policy=TrainEvalPolicy(), populate_policy=None, seed=0,
                 device="cuda" if torch.cuda.is_available() else "cpu", optimizer_parameters=None):
        super().__init__(env, replay, n_actions, net_type, net_parameters, minibatch_size=minibatch_size,
                         optimizer=optimizer, C=C, update_frequency=update_frequency, gamma=gamma, loss=loss,
                         policy=policy, populate_policy=populate_policy, seed=seed, device=device,
                         optimizer_parameters=optimizer_parameters)

    # ================================================================================================================
    # Overcooked Specific Methods
    # ================================================================================================================
    def play_with_video(self, render=True, reset_at_the_end=True, frames_directory=None, videos_directory=None,
                        episode_number=0):
        self.eval()
        frames = []
        observation = self.env.reset()
        done = False
        total_reward = 0

        while not done:
            if render:
                self.env.render()
            if frames_directory is not None:
                frames.append(self.env.render(mode="silent"))
            at = self.action(observation)
            observation, rt, done, _ = self.env.step(at)
            total_reward += rt
        if render:
            self.env.render()
        if reset_at_the_end:
            self.env.reset()

        if frames_directory is not None:
            for f, frame in enumerate(frames):
                image = fromarray(frame)
                filename = f"{frames_directory}/step_{f}.png"
                image.save(filename)
            if videos_directory is not None:
                make_video(f"{frames_directory}", f"{videos_directory}/episode_{episode_number}.mp4")
        return total_reward

    # ================================================================================================================
    # DQN Specific Methods
    # ================================================================================================================

    def eval_round(self, save_dir, eval_episodes):
        """Used to evaluate the agent. Makes the agent play eval_episodes games in eval mode and stores its
        statistics."""
        agent_dir = save_dir if (save_dir[-1] == '/' or save_dir[-1] == '\\') else save_dir + '/'
        eval_stats_path = agent_dir + f"eval_stats_{eval_episodes}_eps.csv"
        intermediate_agent_dir = agent_dir + str(self.n_frames) + "/"
        videos_dir = intermediate_agent_dir+f"videos/"

        os.makedirs(intermediate_agent_dir)
        os.makedirs(videos_dir)

        eval_results = []
        for ep in range(eval_episodes):
            frames_dir = intermediate_agent_dir+f"frames_{ep}/"
            os.makedirs(frames_dir)
            eval_results.append(self.play_with_video(render=False, frames_directory=frames_dir,
                                                     videos_directory=videos_dir, episode_number=ep))

        std = np.std(eval_results)
        avg = np.average(eval_results)

        if not os.path.exists(eval_stats_path):
            with open(eval_stats_path, "w") as stats_file:
                label_line = "frames,avg,std"+"".join([f",ep{str(i)}" for i in range(1, eval_episodes+1)])
                stats_file.write(label_line)

        with open(eval_stats_path, "a") as stats_file:
            line = f"\n{self.n_frames},{avg},{std}"+"".join([f",{str(eval_result)}" for eval_result in eval_results])
            stats_file.write(line)

        self.best_eval_average = max(avg, self.best_eval_average)
        self.save(intermediate_agent_dir, save_replay=False, save_policy=False, save_train_stats=False)



