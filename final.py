import numpy as np
import torch
import gymnasium as gym
import dQN 
import agent


env = gym.make("LunarLander-v2", render_mode="human", enable_wind=False)

agent = agent.Agent(env)

agent.q_network.load_state_dict(torch.load(r"q_weights_128.pz"))
agent.target_network.load_state_dict(torch.load(r"q_weights_128.pz"))

score = 0
state = env.reset()[0]
while True:
    env.render()
    action = agent.get_action(state,eps=0)
    next_state, reward, done, _ , _ = env.step(action)
    score += reward
    state = next_state
    if done:
        break

print("Score reached:", score)