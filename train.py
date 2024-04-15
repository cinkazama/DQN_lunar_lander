import dQN
import gymnasium as gym
import torch
import agent
import numpy as np

# trains the agent and saves trained weights

env = gym.make("LunarLander-v2")#, render_mode="human")
a = env.observation_space.shape
num_actions = env.action_space.n

eps = 1
eps_end = 0.05
eps_decay = 1e-3

agent = agent.Agent(env)
scores = []

for i in range(3000):
    state = env.reset()[0]
    score = 0
    for j in range(1000):
        #env.render()
        action = agent.get_action(state, eps)
        next_state, reward, done, _ , _ = env.step(action)
        agent.step(state, action, reward, next_state, done)
        score += reward
        state = next_state
        if done: 
            break
    eps = max(eps_end, eps-eps_decay)    
    scores.append(score)

    if (i+1)%50 == 0:
        print("Epoch Average Score:", i, np.mean(scores[-50:]), j, eps)
    
torch.save(agent.q_network.state_dict(),"q_weights_128.pz")








