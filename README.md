# Tetris

使用DQN训练智能体玩俄罗斯方块。

## 算法

rl文件夹中包含两种算法：

DQN为最原始的Deep Q Network算法，仅使用了经验回放和目标网络两个小优化。

Rainbow为当前基于价值的强化学习算法的SOTA（state of the art），将Double DQN，对决网络（Dueling DQN），优先经验回放（PER），噪声网络（Noisy Net），值分布强化学习（Categorical DQN），多步经验回放（N-Step Learning）结合起来，使得网络更快收敛。本项目中的Rainbow使用C51值分布算法的时候，尚有bug没能解决，故目前只有rainbow_no_c51可用。

## 训练

训练使用train.py，核心代码为：

```python
agent.train(num_episodes=2000, plotting_interval=100, render=True)
```

`train` 接受三个参数，参数 `num_episodes` 规定agent训练多少局，参数 `plotting_interval` 规定agent每多少局输出一次score、loss等指标的图像，并保存当前阶段的神经网络，默认1000，参数 `render` 表示是否要输出可见的游戏画面，为了减少低配显卡的卡顿感，训练的时候可以关闭。

## 测试

测试使用test.py，核心代码为：

```python
agent.test(video_folder="rainbow.mp4", max_scores=10000)
```

`test` 接受两个参数，参数 `video_folder` 指定测试视频输出路径，只支持mp4格式，参数 `max_scores` 指定测试最高分，达到最高分后不会继续测试，默认-1，表示没有限制，可以直接终止程序以获取视频输出。

## 参考

* [https://github.com/uvipen/Tetris-deep-Q-learning-pytorch](https://github.com/uvipen/Tetris-deep-Q-learning-pytorch)
* [https://github.com/Curt-Park/rainbow-is-all-you-need](https://github.com/Curt-Park/rainbow-is-all-you-need)