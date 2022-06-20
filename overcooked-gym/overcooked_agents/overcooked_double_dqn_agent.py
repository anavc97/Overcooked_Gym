from dqn.policies.train_eval_policy import TrainEvalPolicy
from overcooked_agents.overcooked_dqn_agent import OvercookedDQNAgent
from make_video import make_video
from PIL.Image import fromarray
import numpy as np
import os
import torch
import torch.nn.functional as F


class OvercookedDoubleDQNAgent(OvercookedDQNAgent):
    def __init__(self, env, replay, n_actions, net_type, net_parameters, minibatch_size=32,
                 optimizer=torch.optim.RMSprop, C=10_000, update_frequency=1, gamma=0.99, loss=F.mse_loss,
                 policy=TrainEvalPolicy(), populate_policy=None, seed=0,
                 device="cuda" if torch.cuda.is_available() else "cpu", optimizer_parameters=None):
        super().__init__(env, replay, n_actions, net_type, net_parameters, minibatch_size=minibatch_size,
                         optimizer=optimizer, C=C, update_frequency=update_frequency, gamma=gamma, loss=loss,
                         policy=policy, populate_policy=populate_policy, seed=seed, device=device,
                         optimizer_parameters=optimizer_parameters)

    def update_net(self, r, not_done, next_phi):
        with torch.no_grad():
            actions = self.Q_target(next_phi).max(axis=1, keepdim=True).indices.view(self.minibatch_size)
            q_estimate = torch.zeros(self.minibatch_size).to(self.device)
            q_phi = self.Q(next_phi)
            for i in range(self.minibatch_size):
                q_estimate[i] = q_phi[i, actions[i]]
            y = (r + self.gamma * not_done * q_estimate)
        return y

