from collections import namedtuple
import random
import torch
import numpy as np

data = namedtuple("data", "state, action, reward, next_state, done")

class ReplayBuffer:

    def __init__(self, num_actions, num_states, batch_size, buffer_size):
        self.num_actions = num_actions
        self.num_states = num_states
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.location = 0
        self.buffer = []
        
    def add(self, state, action, reward, next_state, done):
        """ Add a (s a r s' done) tuple to the buffer"""
        
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(data(state, action, reward, next_state, done))

        else:
            self.buffer[self.location] = (data(state, action, reward, next_state, done))

        self.location = (self.location + 1) % self.buffer_size


    def sample(self):
        samples = random.sample(self.buffer, self.batch_size)
        batch = data(*zip(*samples))
        states = torch.reshape(torch.cat(batch.state), (self.batch_size, self.num_states))
        actions = torch.tensor(batch.action, device="cuda")
        next_states = torch.reshape(torch.cat(batch.next_state), (self.batch_size, self.num_states))
        rewards = torch.tensor(batch.reward, device="cuda")
        dones = torch.tensor(batch.done, device="cuda")
        return states, actions, rewards, next_states, dones

