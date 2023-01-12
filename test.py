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

    # test
    agent = DQNAgent(env)
    agent.dqn = torch.load("dqn_10.pkl")
    agent.dqn_target = torch.load("dqn_target_10.pkl")
    
    agent.test("dqn.mp4")
    """

    # Rainbow算法

    # test
    agent = DQNAgent(env)
    if torch.cuda.is_available():
        agent.dqn = torch.load("models/dqn.pkl")
        agent.dqn_target = torch.load("models/dqn_target.pkl")
    else:
        agent.dqn = torch.load(
            "models/dqn.pkl",
            map_location=lambda storage, loc: storage)
        agent.dqn_target = torch.load(
            "models/dqn_target.pkl",
            map_location=lambda storage, loc: storage)

    agent.test("rainbow.mp4", max_scores=10000)
