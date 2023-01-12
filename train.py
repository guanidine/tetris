import numpy as np
import torch

# from tetris.src.dqn import DQNAgent
from tetris.rl.rainbow_no_c51 import DQNAgent
from tetris.src.tetris import Tetris

if __name__ == "__main__":
    env = Tetris()

    seed = 19260817
    np.random.seed(seed)
    torch.manual_seed(seed)

    """
    # DQN算法

    # hyper-parameters
    num_episodes = 5000
    memory_size = 30000
    batch_size = 512
    target_update = 200
    epsilon_decay = 1 / 10000

    # train
    agent = DQNAgent(env, memory_size, batch_size, target_update, epsilon_decay)
    """

    # Rainbow算法

    # hyper-parameters
    num_episodes = 2000
    memory_size = 30000
    batch_size = 512
    target_update = 200

    # train
    agent = DQNAgent(env, memory_size, batch_size, target_update)

    agent.train(num_episodes, 100, False)
