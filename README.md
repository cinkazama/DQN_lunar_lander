# DQN_lunar_lander
DQN to land the Lunar-Lander from openAI's Gymnasium

The DQN was built with Pytorch with usage of Nvidia's CUDA.
A simple ReplayBuffer is also used.
Unfortunately I've hardcoded the cuda usage such that one has to probably rewrite some lines of code io to make it work on a CPU.
Some already trained weights for different NN sizes are also uploaded. 
_train.py_ trains the DQN while _final.py_ just visualizes the trained agent from loaded weights previously saved in _train.py_.

The trained agent in action:
![](https://github.com/cinkazama/DQN_lunar_lander/blob/main/output.gif)
