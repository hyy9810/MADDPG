import numpy as np
import torch
import os
from maddpg.maddpg import MADDPG


class Agent:
    def __init__(self, agent_id, args):
        self.args = args
        self.agent_id = agent_id
        self.policy = MADDPG(args, agent_id)

    def select_action(self, o, noise_rate, epsilon):
        # o = o.transpose(2,0,1) # HWC->CHW
        if np.random.uniform() < epsilon:
            u = np.random.uniform(self.args.low_action, self.args.high_action, self.args.action_shape[self.agent_id]).astype('float32')
        else:
            inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
            pi = self.policy.actor_network(inputs).squeeze(0)
            # print('{} : {}'.format(self.name, pi))
            u = pi.cpu().numpy().astype('float32')
            noise = noise_rate * self.args.high_action * np.random.randn(*u.shape)  # gaussian noise
            u += noise
            u = np.clip(u, self.args.low_action, self.args.high_action).astype("float32")
        return u.copy()
    def select_actions(self, all_o, noise_rate, epsilon):
        # o = o.transpose(2,0,1) # HWC->CHW

        inputs = torch.tensor(all_o, dtype=torch.float32).to(self.args.device)
        pi = self.policy.actor_network(inputs)
        # print('{} : {}'.format(self.name, pi))
        all_u = pi.cpu().numpy().astype('float32')
        noise = noise_rate * self.args.high_action * np.random.randn(*all_u.shape)  # gaussian noise
        all_u += noise
        all_u = np.clip(all_u, self.args.low_action, self.args.high_action).astype("float32")
        for i in range(all_u.shape[0]):
            if np.random.uniform() < epsilon:
                all_u[i] = np.random.uniform(self.args.low_action, self.args.high_action, self.args.action_shape[self.agent_id]).astype('float32')
                
        return all_u.copy()

    def learn(self, transitions):
        self.policy.train(transitions)

