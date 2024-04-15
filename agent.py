import numpy as np
import random
import torch
import torch.nn as nn
from dQN import dQN
import torch.nn.functional as F
from ReplayBuffer import ReplayBuffer

class Agent:

    def __init__(self, env):

        self.env = env
        self.q_network = dQN(env).to("cuda")
        self.target_network = dQN(env).to("cuda")
        self.optimizer = torch.optim.Adam(self.q_network.parameters(),lr = 1e-3)
        self.count = 0
        self.batch_size = 64

        self.Buffer = ReplayBuffer(self.env.action_space.n, self.env.observation_space.shape[0], self.batch_size, buffer_size = 1000000)


    def get_action(self, state, eps):
        state = torch.tensor(state, device="cuda")
        self.q_network.eval()
        with torch.no_grad():
            best_action = torch.argmax(self.q_network(state)).cpu().item()
        self.q_network.train()

        # eps greedy algorithm 
        if random.random() <= eps:
            action = random.randrange(0, self.env.action_space.n, 1)
        else:
            action = best_action
        
        return action
        

    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def calc_loss(self, q_values, target_q_values, actions, rewards, dones):
        num_actions = self.env.action_space.n
        gamma = 0.99

        Q_samp = rewards + torch.bitwise_not(dones).double().to("cuda")*gamma*torch.max(target_q_values, 1)[0]
        Q_value = torch.sum(q_values*F.one_hot(actions, num_actions), 1).double() # torch.tensor(actions, device="cuda")

        return F.mse_loss(Q_samp, Q_value)
    
    def step (self, state, action, reward, next_state, done):
        state = torch.tensor(state, device="cuda")
        next_state = torch.tensor(next_state, device="cuda")
        self.Buffer.add(state, action, reward, next_state, done)
        self.count += 1

        if (self.count % 3) == 0: # train every _ steps
            if self.count > self.batch_size:
                states, actions, rewards, next_states, dones = self.Buffer.sample()
                self.train(states, actions, rewards, next_states, dones)
                
    def train(self, states, actions, rewards, next_states, dones):
        target_q_values = self.target_network(next_states)
        q_values = self.q_network(states)
        loss = self.calc_loss(q_values, target_q_values, actions, rewards, dones)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        
        if self.count % 50 == 0:
            self.update_target()
