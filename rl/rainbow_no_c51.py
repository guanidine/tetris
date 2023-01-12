import math
import random
from collections import deque
from typing import Deque, Dict, List, Tuple, Union

import cv2
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from tetris.src.segment_tree import SumSegmentTree, MinSegmentTree


class ReplayBuffer:
    """使用numpy.ndarray的经验回放数组。"""

    def __init__(
            self,
            obs_dim: int,
            size: int,
            batch_size: int = 32,
            n_step: int = 1,
            gamma: float = 0.99
    ):
        """初始化。

        Args:
            obs_dim: 状态空间维度
            size: 经验回放数组大小
            batch_size: 批量经验回放取样数
            n_step: 经验回放步数，默认1，不做设定时就相当于普通的单步经验回放数组
            gamma: 折扣率
        """
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def store(
            self,
            obs: np.ndarray,
            act: np.ndarray,
            next_obs: np.ndarray,
            rew: float,
            done: bool,
    ) -> Union[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool], None]:
        """存入一条经验回放记录。

        N-step中有两个经验回放数组。其中memory的n_step是1，这意味着store()中“做一次N步的转移”实际上不会发生，buffer存放的还是单步的记录。而对于memory_n而言，它的n_step_buffer存放的是最近N次的记录（调用store的时候，n_step_buffer就直接把记录存进去了，是单步的），而buffer中存放的则是多步记录。

        Args:
            obs: 当前状态s
            act: 当前动作a
            next_obs: 下一状态s'
            rew: 当前奖励r
            done: 回合是否结束

        Returns:
            一条单步经验回放记录。这条记录来自于n_step_buffer[0]。这意味着，这条单步经验回放记录的obs和本次调用store()生成的多步经验回放记录的obs是相同的。
        """
        transition = (obs, act, next_obs, rew, done)
        self.n_step_buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return None

        # make an n-step transition
        next_obs, rew, done = self._get_n_step_info(
            self.n_step_buffer, self.gamma
        )
        obs, act = self.n_step_buffer[0][:2]

        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        return self.n_step_buffer[0]

    def sample_batch(self) -> Dict[str, np.ndarray]:
        """批量取出batch_size条记录做经验回放。

        这些记录是单步还是多步取决于调用这一函数的对象。如果是memory数组调用，取出的都是单步记录，如果是memory_n，取出的则是多步记录。

        Returns:
            返回batch_size条记录，默认32条
        """
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)

        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
            indices=idxs,
        )

    def sample_batch_from_idxs(
            self, idxs: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """根据索引取出N-step的记录。

        具体用法是先调用memory.sample_batch()取出一些单步记录，由于这些记录与memory_n中同obs的记录有着相同的索引，所以可以利用它们的索引idxs，调用memory_n.simple_batch_from_idxs(idxs)取出这些状态对应的多步记录。

        Args:
            idxs: 记录索引

        Returns:
            根据索引返回记录
        """
        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
        )

    @staticmethod
    def _get_n_step_info(
            n_step_buffer: Deque, gamma: float
    ) -> Tuple[np.int64, np.ndarray, bool]:
        """做一次N步的转移。

        这一次转移的obs来自n_step_buffer[0]，转移至的状态next_obs来自n_step_buffer[-1]。

        Args:
            n_step_buffer: 多步经验回放数组
            gamma: 折扣率

        Returns:
            返回N步后的状态：(N步奖励, 下(N)一状态，是否结束)
        """
        # info of the last transition
        next_obs, rew, done = n_step_buffer[-1][-3:]

        for transition in reversed(list(n_step_buffer)[:-1]):
            n_o, r, d = transition[-3:]

            rew = r + gamma * rew * (1 - d)
            next_obs, done = (n_o, d) if d else (next_obs, done)

        return next_obs, rew, done

    def __len__(self) -> int:
        return self.size


class PrioritizedReplayBuffer(ReplayBuffer):
    """优先经验回放数组。

    Attributes:
        max_priority (float): 最大优先级（实际是max_priorityᵅ）
        tree_ptr (int): 线段树指针
        alpha (float): 各样本被抽中的概率是Pi=pᵅ/∑pᵅ，线段树中存的优先级都是pᵅ
        sum_tree (SumSegmentTree): 抽样所用的线段树
        min_tree (MinSegmentTree): 小根堆维护min_priority
    """

    def __init__(
            self,
            obs_dim: int,
            size: int,
            batch_size: int = 32,
            alpha: float = 0.6,
            n_step: int = 1,
            gamma: float = 0.99,
    ):
        """初始化。

        Args:
            obs_dim: 状态空间维度
            size: 经验回放数组大小
            batch_size: 批量经验回放取样数
            alpha: 各样本被抽中的概率是Pᵢ=pᵅ/∑pᵅ，线段树中叶子结点存放的数值都是pᵅ
        """
        assert alpha >= 0

        super(PrioritizedReplayBuffer, self).__init__(
            obs_dim, size, batch_size, n_step, gamma
        )
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def store(
            self,
            obs: np.ndarray,
            act: np.ndarray,
            next_obs: np.ndarray,
            rew: float,
            done: bool,
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        """存入一条记录。

        Args:
            obs: 当前状态s
            act: 当前动作a
            next_obs: 下一状态s'
            rew: 当前奖励r
            done: 回合是否结束

        Returns:
            一条单步经验回放记录。rainbow中只有memory是优先经验回放数组，返回的就是刚存入的transition
        """
        transition = super().store(obs, act, next_obs, rew, done)

        if transition:
            self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.tree_ptr = (self.tree_ptr + 1) % self.max_size

        return transition

    def sample_batch(self, beta: float = 0.4) -> Dict[str, np.ndarray]:
        """批量取出batch_size条记录做经验回放。

        优先经验回放利用线段树维护各记录的权重，权重大的更有几率被抽中。

        Args:
            beta: 学习率αᵢ=α/(N·Pᵢ)ᵝ中的β，是一个在(0, 1)之间的超参数，论文建议一开始让β比较小，最终增长到1

        Returns:
            返回batch_size条记录，默认 32 条
        """
        assert len(self) >= self.batch_size
        assert beta > 0

        indices = self._sample_proportional()

        obs = self.obs_buf[indices]
        next_obs = self.next_obs_buf[indices]
        acts = self.acts_buf[indices]
        rews = self.rews_buf[indices]
        done = self.done_buf[indices]
        weights = np.array([self._calculate_weight(i, beta) for i in indices])

        return dict(
            obs=obs,
            next_obs=next_obs,
            acts=acts,
            rews=rews,
            done=done,
            weights=weights,
            indices=indices,
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """更新各记录优先级。

        Args:
            indices: 各记录在经验回放数组中的索引
            priorities: 对应优先级，由loss+eps得到，利用Pᵢ=pᵃ/∑pᵃ折算成被抽中的概率Pᵢ
        """
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self) -> List[int]:
        """根据优先级采样。

        多次采样的时候，既希望采样到“更重要”的记录，也要兼顾均匀性，因此要在batch_size个小区间内分别抽样。

        Returns:
            被采样记录的下标
        """
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)

        return indices

    def _calculate_weight(self, idx: int, beta: float):
        """计算权重。

        Args:
            idx: 在经验回放数组中的索引
            beta: 学习率αᵢ=α/(N·Pᵢ)ᵝ中的β，是一个在(0, 1)之间的超参数，论文建议一开始让β比较小，最终增长到1
        Returns:
            权重（乘在各样本的loss上，实际作用是使各样本的学习率不同）
        """
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight

        return weight


class NoisyLinear(nn.Module):
    """噪声DQN的噪声层。

    噪声DQN用噪声层代替原来的Linear层。原先y=Wx+b中权重W(weight)和偏移量b(bias)都是固定的，DQN在训练好的W和b参数下表现尚可，而一旦参数发生了微小的变化，则会变得很差。添加噪声使得W和b分别在μw和μb邻域内浮动，可以增强网络的鲁棒性。

    Attributes:
        in_features (int): 噪声层输入维度
        out_features (int): 噪声层输出维度
        std_init (float): 初始σ
        weight_mu (nn.Parameter): μ_weight
        weight_sigma (nn.Parameter): σ_weight
        bias_mu (nn.Parameter): μ_bias
        bias_sigma (nn.Parameter): σ_bias
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            std_init: float = 0.5,
    ):
        """噪声层的初始化。

        Args:
            in_features: 噪声层输入维度
            out_features: 噪声层输出维度
            std_init: 初始σ
        """
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        self.register_buffer(
            "weight_epsilon", torch.Tensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """分解高斯参数的初始化。

        μ₁, μ₂ ~ U(-1/√n₁, 1/√n₁)

        σ₁ = σ/√n₁

        σ₂ = σ/√n₂
        """
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

    def reset_noise(self):
        """生成新的噪声ξ"""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Q(s, a, ξ; µ, σ) ≜ Q(s, a; µ + σ ◦ ξ)

        y = (µ₁ + σ₁ ◦ ξ₁) x + (µ₂ + σ₂ ◦ ξ₂)

        理论上，训练好以后用DQN做决策时，不再需要噪声，可以把参数σ设置为全0，只保留参数μ，这样噪声DQN就变成了标准的DQN。
        实际上我们不用这么做，是否有噪声对DQN的表现并没有多少影响。

        Args:
            x: 输入的Tensor

        Returns:
            经过噪声层处理的Tensor
        """
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )

    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """f(x) = sgn(x)√|x|

        Args:
            size: tensor的大小

        Returns:
            通过torch.randn()随机生成x，随后返回f(x)
        """
        x = torch.randn(size)

        return x.sign().mul(x.abs().sqrt())


class Network(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """初始化。

        结合了Noisy Net、Dueling DQN两种优化策略。第一个共享的特征层使用线性函数y=Wx+b，维度为128，随后分为优势头和状态价值头。优势头输出的是每个动作的优势值，状态价值头输出的是状态价值。两者都使用两层噪声网络，组合起来得到每个动作的动作价值函数。激活函数使用ReLU。

        Args:
            in_dim: 输入维度
            out_dim: 输出维度
        """
        super(Network, self).__init__()

        # set common feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
        )

        # set advantage layer
        self.advantage_hidden_layer = NoisyLinear(128, 128)
        self.advantage_layer = NoisyLinear(128, out_dim)

        # set value layer
        self.value_hidden_layer = NoisyLinear(128, 128)
        self.value_layer = NoisyLinear(128, 1)

        self._create_weights()

    def _create_weights(self):
        """随机初始化。"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        在值分布中，q=zᵀp。

        Args:
            x: 输入的Tensor

        Returns:
            经过神经网络处理后的Tensor
        """
        feature = self.feature_layer(x)

        adv_hid = F.relu(self.advantage_hidden_layer(feature))
        val_hid = F.relu(self.value_hidden_layer(feature))

        value = self.value_layer(val_hid)
        advantage = self.advantage_layer(adv_hid)

        q = value + advantage - advantage.mean(dim=-1, keepdim=True)

        return q

    def reset_noise(self):
        """生成新的噪声ξ"""
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()


class DQNAgent:
    """DQN Agent interacting with environment.

    Attributes:
        env (gym.Env): OpenAI Gym 环境
        memory (ReplayBuffer): 经验回放数组大小
        batch_size (int): 经验回放取样数
        target_update (float): 做一次目标网络硬更新间隔的迭代次数
        gamma (float): 折扣率
        dqn (Network): DQN神经网络
        dqn_target (Network): 目标网络
        optimizer (torch.optim): 神经网络优化器
        transition (list): 用于经验回放的记录，为五元组(state, action, reward, next_state, done)
        use_n_step (bool): 是否使用多步经验回放
        n_step (int): 经验回放步数
        memory_n (ReplayBuffer): 多步经验回放数组
    """

    def __init__(
            self,
            env,
            memory_size: int = 10000,
            batch_size: int = 128,
            target_update: int = 100,
            gamma: float = 0.99,
            alpha: float = 0.2,
            beta: float = 0.6,
            prior_eps: float = 1e-6,
            n_step: int = 10,
    ):
        """初始化。

        Args:
            env: OpenAI Gym 环境
            memory_size: 经验回放数组大小
            batch_size: 经验回放取样数
            target_update: 做一次目标网络硬更新间隔的迭代次数
            gamma: 折扣率
            alpha: 各样本被抽中的概率是Pᵢ=pᵅ/∑pᵅ，线段树中叶子结点存放的数值都是pᵅ
            beta: 学习率αᵢ=α/(N·Pᵢ)ᵝ中的β，是一个在(0, 1)之间的超参数，论文建议一开始让β比较小，最终增长到1
            prior_eps: eps是很小的正数，以确保一个样本的优先级不至于过小而无法被选中
            n_step: 经验回放步数
        """
        obs_dim = 4
        action_dim = 1

        self.env = env
        self.batch_size = batch_size
        self.target_update = target_update
        self.gamma = gamma
        # NoisyNet: All attributes related to epsilon are removed

        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)

        # PER
        # memory for 1-step Learning
        self.beta = beta
        self.prior_eps = prior_eps
        self.memory = PrioritizedReplayBuffer(
            obs_dim, memory_size, batch_size, alpha=alpha
        )

        # memory for N-step Learning
        self.use_n_step = True if n_step > 1 else False
        if self.use_n_step:
            self.n_step = n_step
            self.memory_n = ReplayBuffer(
                obs_dim, memory_size, batch_size, n_step=n_step, gamma=gamma
            )

        # networks: dqn, dqn_target
        self.dqn = Network(obs_dim, action_dim).to(self.device)
        self.dqn_target = Network(obs_dim, action_dim).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters())

        # transition to store in memory
        self.transition = list()

        # mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray, next_states: Tensor) -> np.ndarray:
        """根据当前状态选取动作。

        噪声网络本身就带有随机性，可以鼓励探索，不再需要ε-greedy策略。

        Args:
            state: 当前状态
            next_states: 下一帧可选状态

        Returns:
            应选动作
        """
        # NoisyNet: no epsilon greedy action selection
        selected_action = torch.argmax(
            self.dqn(next_states.to(self.device))
        ).detach().cpu()

        if not self.is_test:
            self.transition = [
                state,
                selected_action,
                next_states[selected_action, :]
            ]

        return selected_action

    def step(
            self, action: np.ndarray,
            render: bool = True,
            video: cv2.VideoWriter = None
    ) -> Tuple[np.float64, bool]:
        """采取动作并更新环境。

        开启多步经验回放后，调用一次step会同时存入一条单步记录和一条多步记录。

        Args:
            action: 采取的动作
            render: 是否生成视频
            video: cv2视频输出

        Returns:
            (下一状态，当前奖励，是否结束)——这里的“下一状态”就是单步转移后的下一状态
        """
        reward, done = self.env.step(action, render=render, video=video)

        if not self.is_test:
            self.transition += [reward, done]

            # N-step transition
            if self.use_n_step:
                one_step_transition = self.memory_n.store(*self.transition)
            # 1-step transition
            else:
                one_step_transition = self.transition

            # add a single step transition
            if one_step_transition:
                self.memory.store(*one_step_transition)

        return reward, done

    def update_model(self) -> float:
        """用梯度下降算法更新网络。

        rainbow的梯度下降，从优先经验回放数组中取单步经验计算单步loss，从多步经验回放数组中取多步经验计算多步loss。具体loss计算采用Double DQN优化并根据C51算法计算KL散度，返回element-wise形式的loss。对总梯度设置一个梯度剪切的阈值，防止梯度爆炸。完成梯度下降后，更新优先级并重新初始化噪声，开始下一次迭代。

        Returns:
            本次迭代的Loss（数值形式）
        """
        # PER needs beta to calculate weights
        samples = self.memory.sample_batch(self.beta)
        weights = torch.FloatTensor(
            samples["weights"].reshape(-1, 1)
        ).to(self.device)
        indices = samples["indices"]

        # 1-step Learning loss
        elementwise_loss = self._compute_dqn_loss(samples, self.gamma)

        # PER: importance sampling before average
        loss = torch.mean(elementwise_loss * weights)

        # N-step Learning loss
        # we are gonna combine 1-step loss and n-step loss so as to
        # prevent high-variance. The original rainbow employs n-step loss only.
        if self.use_n_step:
            gamma = self.gamma ** self.n_step
            samples = self.memory_n.sample_batch_from_idxs(indices)
            elementwise_loss_n_loss = self._compute_dqn_loss(samples, gamma)
            elementwise_loss += elementwise_loss_n_loss

            # PER: importance sampling before average
            loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.optimizer.step()

        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)

        # NoisyNet: reset noise
        self.dqn.reset_noise()
        self.dqn_target.reset_noise()

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
        losses = []
        scores = []
        episode = 0

        while episode < num_episodes:
            next_steps = self.env.get_next_states()
            next_actions, next_states = zip(*next_steps.items())
            next_states = torch.stack(next_states)

            action = self.select_action(state, next_states)
            next_state = next_states[action, :]
            action = next_actions[action]

            reward, done = self.step(action)

            # NoisyNet: removed decrease of epsilon

            # PER: increase beta
            fraction = min(episode / num_episodes, 1.0)
            self.beta = self.beta + fraction * (1.0 - self.beta)

            # if training is ready
            if len(self.memory) >= self.batch_size:
                loss = self.update_model()
                losses.append(loss)
                update_cnt += 1

                # if hard update is needed
                if update_cnt % self.target_update == 0:
                    self._target_hard_update()

            # if episode ends
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

            print("Epoch: {}/{}, Score: {}, Tetrominoes {}, Cleared lines: {}".format(
                episode,
                num_episodes,
                score,
                final_tetrominoes,
                final_cleared_lines))

            if episode % plotting_interval == 0 and episode < num_episodes:
                torch.save(self.dqn, "models/dqn_{}.pkl".format(episode))
                torch.save(self.dqn_target, "models/dqn_target_{}.pkl".format(episode))
                self._plot(episode, scores, losses)

        torch.save(self.dqn, "models/dqn.pkl")
        torch.save(self.dqn_target, "models/dqn_target.pkl")
        self._plot(episode, scores, losses)

    def test(self, video_folder: str, max_scores: int = -1) -> None:
        """测试agent。

        Args:
            video_folder: 视频存放路径
            max_scores: 最大得分，-1为不设限制
        """
        self.is_test = True

        # for recording a video
        out = cv2.VideoWriter(
            video_folder,
            cv2.VideoWriter_fourcc(*'mp4v'),
            300,
            (450, 600)
        )

        state = self.env.reset()
        done = False
        score = 0

        while not done:
            next_steps = self.env.get_next_states()
            next_actions, next_states = zip(*next_steps.items())
            next_states = torch.stack(next_states)

            action = self.select_action(state, next_states)
            next_state = next_states[action, :]
            action = next_actions[action]

            reward, done = self.step(action, render=True, video=out)
            score = self.env.score

            state = next_state

            if 0 < max_scores <= score:
                break

        final_tetrominoes = self.env.tetrominoes
        final_cleared_lines = self.env.cleared_lines

        print("Score: {}, Tetrominoes {}, Cleared lines: {}".format(
            score,
            final_tetrominoes,
            final_cleared_lines))

        out.release()

    def _compute_dqn_loss(
            self,
            samples: Dict[str, np.ndarray],
            gamma: float
    ) -> torch.Tensor:
        """计算DQN的Loss。

        使用TD算法时，运用Double DQN策略，从DQN网络中选择动作a，在目标网络中计算Q，以缓解高估问题。最终以element-wise的形式返回loss，用于更新优先经验回放数组中的优先级。

        Args:
            samples: 取出的batch_size条记录
            gamma: 折扣率（一步/多步）

        Returns:
            返回Loss（element-wise）
        """
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        curr_q_value = self.dqn(state)
        next_q_value = self.dqn_target(next_state).gather(
            1, self.dqn(next_state).argmax(dim=1, keepdim=True)
        ).detach()

        # TD算法
        # yt = rt + γ · Q(s(t+1), a*)   if 还有下一状态
        #    = rt + γⁿ · Q(s(t+n), a*)
        # yt = rt                       if 回合终止
        mask = 1 - done
        target = (reward + gamma * next_q_value * mask).to(self.device)

        elementwise_loss = F.smooth_l1_loss(curr_q_value, target, reduction="none")

        return elementwise_loss

    def _target_hard_update(self):
        """硬更新。将dqn_target网络更新为最新的dqn网络。

        对DQN做正向传播得到qt，对DQN Target网络做正向传播得到q(t+1)并由此计算yt和δt，再用梯度下降更新DQN的参数，可降低自举造成的危害，缓解高估问题。
        """
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    @staticmethod
    def _plot(
            episode: int,
            scores: List[float],
            losses: List[float],
    ):
        """绘出训练过程。

        其实在DQN训练过程中，唯一需要关注的指标就是Average Return，其他的一切Loss都可能是伪指标。

        Args:
            episode: 当前迭代次数
            scores: 从开始训练起各回合得分
            losses: 从开始训练起各轮迭代Loss
        """
        clear_output(True)
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('frame %s. score: %s' % (episode, np.mean(scores[-10:])))
        plt.plot(scores)
        plt.subplot(132)
        plt.title('loss')
        plt.plot(losses)
        plt.show()
