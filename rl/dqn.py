from typing import Dict, List, Tuple

import cv2
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output
from numpy import random
from torch import Tensor


class ReplayBuffer:
    """使用numpy.ndarray的经验回放数组。"""

    def __init__(self, obs_dim: int, size: int, batch_size: int = 32):
        """初始化。

        Args:
            obs_dim: 状态空间维度
            size: 经验回放数组大小
            batch_size: 批量经验回放取样数
        """
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

    def store(
            self,
            obs: np.ndarray,
            act: np.ndarray,
            next_obs: np.ndarray,
            rew: float,
            done: bool,
    ):
        """存入一条记录。

        Args:
            obs: 当前状态s
            act: 当前动作a
            next_obs: 下一状态s'
            rew: 当前奖励r
            done: 回合是否结束
        """
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        """批量取出batch_size条记录做经验回放。

        Returns:
            返回batch_size条记录，默认 32 条
        """
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def __len__(self) -> int:
        return self.size


class Network(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """初始化。三个全连接层使用线性函数y=Wx+b，激活函数使用ReLU。

        Args:
            in_dim: 输入维度
            out_dim: 输出维度
        """
        super(Network, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入的Tensor

        Returns:
            经过神经网络处理后的Tensor
        """
        return self.layers(x)


class DQNAgent:
    """DQN智能体。

    Attributes:
        env (gym.Env): OpenAI Gym 环境
        memory (ReplayBuffer): 经验回放数组大小
        batch_size (int): 经验回放取样数
        target_update (float): 做一次目标网络硬更新间隔的迭代次数
        epsilon (float): ε-greedy算法的ε
        epsilon_decay (float): ε衰减率
        max_epsilon (float): ε最大值
        min_epsilon (int): ε最小值
        gamma (float): 折扣率
        dqn (Network): DQN神经网络
        dqn_target (Network): 目标网络
        optimizer (torch.optim): 神经网络优化器
        transition (list): 用于经验回放的记录，为五元组(state, action, reward, next_state, done)
    """

    def __init__(
            self,
            env,
            memory_size: int = 10000,
            batch_size: int = 128,
            target_update: int = 100,
            epsilon_decay: float = 1 / 2000,
            max_epsilon: float = 1.0,
            min_epsilon: float = 0.1,
            gamma: float = 0.99,
    ):
        """初始化。

        Args:
            env: OpenAI Gym 环境
            memory_size: 经验回放数组大小
            batch_size: 经验回放取样数
            target_update: 做一次目标网络硬更新间隔的迭代次数
            epsilon_decay: ε衰减率
            max_epsilon: ε最大值
            min_epsilon: ε最小值
            gamma: 折扣率
        """
        # 有效观察空间维度
        obs_dim = 4
        # 动作空间维度
        action_dim = 1

        self.env = env
        self.memory = ReplayBuffer(obs_dim, memory_size, batch_size)
        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.target_update = target_update
        self.gamma = gamma

        # 判断当前训练使用的是CPU还是GPU
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)

        # 用TD算法训练DQN
        self.dqn = Network(obs_dim, action_dim).to(self.device)
        self.dqn_target = Network(obs_dim, action_dim).to(self.device)
        # 将dqn网络的参数权重加载到dqn_target网络中
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        # model.eval()将模型调整为评估模式（对应于训练模式）
        # 在评估模式下，batchNorm层，dropout层等用于优化训练而添加的网络层会被关闭，从而使得评估时不会发生偏移
        self.dqn_target.eval()

        # 神经网络优化器，使用Adam算法
        # 对于Adam这类自适应学习率的方法，实验证明再使用学习率衰减依旧是有用的，不过这里没有设置
        # https://www.cnblogs.com/wuliytTaotao/p/11101652.html
        self.optimizer = optim.Adam(self.dqn.parameters())

        # 一条记录，(s, a)和(r, s', done)分两步存入transition
        self.transition = list()

        # 模式：训练/测试
        self.is_test = False

    def select_action(self, state: np.ndarray, next_steps: Dict, next_states: Tensor) -> int:
        """根据当前状态选取动作。

        Args:
            state: 当前状态
            next_steps: 下一帧可选状态
            next_states: 下一帧可选状态

        Returns:
            应选动作
        """
        # ε-greedy策略
        if self.epsilon > np.random.random():
            # ε 概率在动作空间中均匀抽取一个动作
            selected_action = random.randint(0, len(next_steps) - 1)
        else:
            # 1-ε 概率选取动作 argmax Q(s, a; w)
            selected_action = torch.argmax(self.dqn(next_states.to(self.device))).detach().cpu()

        if not self.is_test:
            # 如果是在训练，则记录当前状态与动作
            self.transition = [state, selected_action, next_states[selected_action, :]]

        return selected_action

    def step(
            self, action: np.ndarray,
            render: bool = True,
            video: cv2.VideoWriter = None
    ) -> Tuple[np.float64, bool]:
        """采取动作并更新环境。

        Args:
            action: 采取的动作
            render: 是否生成视频
            video: cv2视频输出

        Returns:
            (下一状态，当前奖励，是否结束)
        """
        reward, done = self.env.step(action, render=render, video=video)

        if not self.is_test:
            self.transition += [reward, done]
            # 不能直接存self.transition，因为它是会变的
            self.memory.store(*self.transition)

        return reward, done

    def update_model(self) -> float:
        """用梯度下降算法更新网络。

        Returns:
            本次迭代的Loss（数值形式）
        """
        samples = self.memory.sample_batch()

        loss = self._compute_dqn_loss(samples)

        # 用Optimizer更新神经网络，依旧需要清零梯度，随后只要step()就可以自动梯度下降更新参数了
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, num_episodes: int, plotting_interval: int = 1000):
        """训练agent。

        Args:
            num_episodes: 迭代次数
            plotting_interval: 多少轮迭代输出一次训练图像
        """
        self.is_test = False

        state = self.env.reset()
        update_cnt = 0
        epsilons = []
        losses = []
        scores = []
        episode = 0

        while episode < num_episodes:
            next_steps = self.env.get_next_states()
            next_actions, next_states = zip(*next_steps.items())
            next_states = torch.stack(next_states)

            action = self.select_action(state, next_steps, next_states)
            next_state = next_states[action, :]
            action = next_actions[action]

            reward, done = self.step(action)

            # 回合结束，重置环境，记录得分
            if done:
                score = self.env.score
                final_tetrominoes = self.env.tetrominoes
                final_cleared_lines = self.env.cleared_lines
                state = self.env.reset()
                scores.append(score)
            else:
                state = next_state
                continue

            episode += 1

            # 等回放数组中有足够多的四元组时，才开始做经验回放更新 DQN
            if len(self.memory) >= self.batch_size:
                loss = self.update_model()
                losses.append(loss)
                update_cnt += 1

                # 采用最简单的线性函数更新ε-greedy策略的ε，随着ε减小，策略变得更加稳定
                self.epsilon = max(
                    self.min_epsilon, self.epsilon - (
                            self.max_epsilon - self.min_epsilon
                    ) * self.epsilon_decay
                )
                epsilons.append(self.epsilon)

                # 目标网络硬更新
                if update_cnt % self.target_update == 0:
                    self._target_hard_update()

            print("Epoch: {}/{}, Score: {}, Tetrominoes {}, Cleared lines: {}".format(
                episode,
                num_episodes,
                score,
                final_tetrominoes,
                final_cleared_lines))

            if episode % plotting_interval == 0 and episode < num_episodes:
                torch.save(self.dqn, "dqn_{}.pkl".format(episode))
                torch.save(self.dqn_target, "dqn_target_{}.pkl".format(episode))
                self._plot(episode, scores, losses, epsilons)

        torch.save(self.dqn, "dqn.pkl")
        torch.save(self.dqn_target, "dqn_target.pkl")
        self._plot(episode, scores, losses, epsilons)

    def test(self, video_folder: str) -> None:
        """测试agent。

        Args:
            video_folder: 视频存放路径
        """
        self.is_test = True

        # 生成测试视频
        # for recording a video
        out = cv2.VideoWriter(
            video_folder,
            cv2.VideoWriter_fourcc(*'mp4v'),
            300,
            (450, 600)
        )

        state = self.env.reset()
        done = False

        while not done:
            next_steps = self.env.get_next_states()
            next_actions, next_states = zip(*next_steps.items())
            next_states = torch.stack(next_states)

            action = self.select_action(state, next_steps, next_states)
            next_state = next_states[action, :]
            action = next_actions[action]

            reward, done = self.step(action, render=True, video=out)

            state = next_state

        score = self.env.score
        final_tetrominoes = self.env.tetrominoes
        final_cleared_lines = self.env.cleared_lines

        print("Score: {}, Tetrominoes {}, Cleared lines: {}".format(
            score,
            final_tetrominoes,
            final_cleared_lines))

        out.release()

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        """计算DQN的Loss。

        Args:
            samples: 取出的batch_size条记录

        Returns:
            返回Loss（Tensor形式）
        """
        device = self.device
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # torch.gather函数：https://zhuanlan.zhihu.com/p/352877584
        # A = B.gather(dim=0, index=torch.tensor([[2, 1, 2]]))按列索引，替换行idx
        # 依次取出B[0->2][0], B[0->1][1], B[0->2][2]
        # A = B.gather(dim=1, index=torch.tensor([[2, 1, 2]]))按行索引，替换列idx
        # 依次取出B[0][0->2], B[0][1->1], B[0][2->2]
        # action是(0, 1)张量，通过gather获取批量状态对应批量动作的Q(S, a)
        curr_q_value = self.dqn(state)
        # dqn_target网络的输出要用于梯度下降，但梯度下降不能更改dqn_target的参数，故用detach分离
        next_q_value = self.dqn_target(next_state).max(dim=1, keepdim=True)[0].detach()

        # TD算法
        # yt = rt + γ · Q(s(t+1), a*)   if 还有下一状态
        # yt = rt                       if 回合终止
        mask = 1 - done
        target = (reward + self.gamma * next_q_value * mask).to(self.device)
        # δt = qt - yt
        # Smooth L1 Loss函数
        # 相比L1 Loss改进了零点不平滑的问题：L(x) = 0.5x²  if |x| < 1
        # 相比于L2 Loss在x较大时不会对异常值过于敏感，变化缓慢：L(x) = |x| - 0.5  if |x| > 1
        loss = F.smooth_l1_loss(curr_q_value, target)

        return loss

    def _target_hard_update(self):
        """硬更新。将dqn_target网络更新为最新的dqn网络。

        对DQN做正向传播得到qt，
        对DQN Target网络做正向传播得到q(t+1)并由此计算yt和δt，
        再用梯度下降更新DQN的参数，
        可降低自举造成的危害，缓解高估问题。
        """
        # 将dqn网络的参数权重加载至dqn_target网络中，完成更新
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    def _plot(
            self,
            episode: int,
            scores: List[float],
            losses: List[float],
            epsilons: List[float],
    ):
        """绘出训练过程。

        其实在DQN训练过程中，唯一需要关注的指标就是Average Return，其他的一切Loss都可能是伪指标。

        Args:
            episode: 当前帧（迭代次数）
            scores: 从开始训练起各回合得分
            losses: 从开始训练起各轮迭代Loss
            epsilons: 从开始训练起的各ε
        """
        clear_output(True)
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('frame %s. score: %s' % (episode, np.mean(scores[-10:])))
        plt.plot(scores)
        plt.subplot(132)
        plt.title('loss')
        plt.plot(losses)
        plt.subplot(133)
        plt.title('epsilons')
        plt.plot(epsilons)
        plt.show()
