import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class dQN(nn.Module):

    def __init__(self, env):
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        super(dQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(num_states, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, state):
        """map the state input to the actions through the network"""
        #state_flat = torch.flatten(state, start_dim=1)
        out = self.network(state)

        return out


    
    