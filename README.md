<h2 align="center"> MARLToolkit: The  Multi Agent Rainforcement Learning Toolkit</h2>

MARLToolkit 是一个专门为多智能体强化学习（MARL）设计的工具包，基于 PyTorch 构建，旨在为研究者提供一个统一的平台，便于在各种多智能体环境中开发和评估新的算法和理论。下面是对该工具包的详细介绍：

1. 统一框架
MARLToolkit 的一大特色是将现有的多智能体强化学习算法统一到一个框架下。传统上，不同的算法可能使用不同的代码结构和接口，导致研究者在尝试不同算法时需要花费大量时间去适应每个算法的实现细节。MARLToolkit 通过将这些算法标准化，极大地简化了研究和实验的过程。

该工具包涵盖了多种广为人知的 MARL 算法，按其学习模式可分为以下几类：

a. 独立学习（Independent Learning）
独立学习算法指的是在多智能体环境中，每个智能体独立地进行学习和决策，而不依赖于其他智能体的策略或信息。这类算法将多智能体问题视为单智能体问题的集合，每个智能体只关注自己的行动和奖励。这些算法包括：

IQL（Independent Q-Learning）：IQL 是独立 Q 学习的多智能体版本。每个智能体独立地学习 Q 函数，即在给定状态下选择某一动作的预期回报。IQL 不考虑其他智能体的动作，因此在环境相互作用较强时可能会出现不稳定。

A2C（Advantage Actor-Critic）：A2C 是一种基于策略梯度的方法，结合了策略网络（Actor）和价值网络（Critic）。Actor 负责决定智能体的动作，而 Critic 则评估当前策略的好坏。A2C 可以处理离散和连续的动作空间。

DDPG（Deep Deterministic Policy Gradient）：DDPG 是一种用于连续控制任务的深度强化学习算法，它结合了 Q 学习和策略梯度方法。DDPG 使用 Actor-Critic 架构，其中 Actor 生成连续的动作，Critic 评估动作的价值。

TRPO（Trust Region Policy Optimization）：TRPO 是一种用于策略优化的算法，通过限制策略更新的步长，确保每次更新后的策略不会偏离原始策略太远，从而提高训练的稳定性。TRPO 在复杂的环境中表现出色，尤其是在多智能体环境中。

PPO（Proximal Policy Optimization）：PPO 是 TRPO 的改进版，采用了一种更加简单的策略更新方法，通过限制策略更新时的变化幅度，保证了算法的稳定性和效率。PPO 适用于多种任务，包括多智能体环境。

b. 集中评估学习（Centralized Critic Learning）
集中评估学习算法通过一个中央评估器（Critic）来帮助每个智能体进行学习，这个评估器通常可以访问全局信息，并为每个智能体提供策略改进的指导。这类算法适合合作性较强的任务，包括：

COMA（Counterfactual Multi-Agent Policy Gradient）：COMA 是一种多智能体策略梯度算法，使用集中式的 Critic 来评估每个智能体的行动价值，同时引入反事实基线，减少其他智能体行为对策略更新的影响，从而提高策略梯度的效率。

MADDPG（Multi-Agent Deep Deterministic Policy Gradient）：MADDPG 是 DDPG 的多智能体扩展版本，使用集中式 Critic 评估所有智能体的联合动作，并且每个智能体拥有自己的 Actor 网络，独立选择动作。MADDPG 适用于具有连续动作空间的多智能体任务。

MAPPO（Multi-Agent Proximal Policy Optimization）：MAPPO 是 PPO 的多智能体扩展，结合了集中式 Critic 和独立的 Actor。MAPPO 在合作性任务中表现良好，尤其是在需要多个智能体协同工作的场景中。

HATRPO（Hierarchical Actor-Trained Region Policy Optimization）：HATRPO 是一种分层的多智能体算法，通过分层的策略结构，让高层次策略负责宏观决策，低层次策略负责具体操作。集中式 Critic 在此结构中起到协调作用，确保不同层次的策略有效协同。

c. 价值分解（Value Decomposition）
价值分解算法通过将全局价值函数分解为多个智能体的局部价值函数来实现多智能体学习，适用于合作性任务。这类算法包括：

QMIX：QMIX 是一种基于混合网络的价值分解算法，它将所有智能体的局部 Q 值通过一个非线性混合网络组合成全局 Q 值。这个混合网络的结构保证了各个智能体的独立性，同时也能有效地促进合作。

VDN（Value-Decomposition Networks）：VDN 是一种将全局 Q 值简单地表示为各智能体局部 Q 值的和的算法，适合于完全合作的任务。VDN 提供了一种简单而有效的合作方式，但在需要复杂合作的场景中可能受到限制。

FACMAC（Factorized Multi-Agent Centralized Actor-Critic）：FACMAC 是一种结合了集中评估器和价值分解的算法，它使用一个因式分解的架构来分解全局价值函数，并通过集中式 Critic 来帮助各智能体优化策略，适用于复杂合作任务。

VDA2C（Value Decomposition Actor-Critic）：VDA2C 结合了 A2C 和价值分解思想，将集中评估与价值分解相结合，在提升算法稳定性的同时，提高了多智能体的合作效率。
这些算法被统一实现，并共享同一套接口，使得研究者可以方便地在不同算法之间切换，进行对比实验。

2. 跨环境接口
在多智能体强化学习的研究中，不同的实验环境可能具有不同的接口和数据格式。MARLToolkit 通过提供一个统一的接口，使得研究者可以在不同的环境中轻松切换，而无需修改算法部分的代码。

支持的环境包括：

SMAC（StarCraft Multi-Agent Challenge）：一个基于《星际争霸》的多智能体强化学习环境，常用于测试智能体的协作能力。
MaMujoco：一个用于物理仿真的多智能体环境，支持复杂的连续控制任务。
Google Research Football：一个足球模拟环境，支持复杂的策略和合作研究。
Pommerman：一个基于经典游戏《炸弹人》的多智能体环境，测试竞争与合作策略。
MetaDrive：一个自动驾驶模拟环境，支持部分可观测和连续控制任务。
这些环境的统一接口意味着，无论是在模拟对抗、合作还是混合任务中，研究者都能方便地部署和评估不同的算法。

3. 高效训练与采样
MARLToolkit 在设计时高度重视效率，无论是训练过程还是数据采样过程，都进行了优化，以确保算法能在合理的时间范围内完成学习。高效的训练流程不仅加快了实验速度，还减少了计算资源的消耗，使得研究者能够在有限的资源下进行更多的实验。

4. 结果可信度
为保证研究结果的可信度，MARLToolkit 提供了经过细致调优的训练结果，包括：

学习曲线：展示了算法在不同任务中的学习进展。
预训练模型：针对每个任务和算法的组合提供了经过微调的预训练模型，研究者可以基于这些模型进行进一步的实验，而不必从头开始训练。
超参数调优：每个算法都经过了仔细的超参数调优，以确保其在特定任务中的表现最佳，这增强了结果的可重复性和可靠性。


超参数调优是指对算法中那些不能通过数据直接学习到的参数进行优化，以提高算法的性能。常见的超参数包括学习率、折扣因子、网络结构（如隐藏层数和节点数）、探索策略参数（如 epsilon 的初始值和衰减率）等。

超参数的选择对算法的表现有很大的影响。通常，研究者会通过以下方式进行超参数调优：

网格搜索：在一个预定义的参数范围内，穷举所有可能的超参数组合，找出最优的组合。这种方法保证了全局最优，但计算成本较高。

随机搜索：在参数空间中随机选择一组超参数进行评估，效率比网格搜索高，但可能错过最优解。

贝叶斯优化：使用贝叶斯优化方法，在超参数空间中找到表现最佳的参数组合。相比于网格搜索和随机搜索，贝叶斯优化更高效，能够在较少的试验中找到接近最优的超参数组合。

通过超参数调优，研究者可以显著提高算法在特定任务中的表现，确保模型在实际应用中的有效性和稳定性。


5. 基准测试与社区支持
MARLToolkit 提供了一系列基准测试，用于评估算法在不同任务中的表现。工具包内置了多种经典 MARL 算法的实现，研究者可以直接使用这些实现进行对比实验。此外，MARLToolkit 还得到了社区的广泛支持，用户可以通过 GitHub 等平台分享自己的实验结果和代码贡献，使得整个研究社区能够共享进步。

总结
MARLToolkit 是一个强大的工具包，通过统一的框架和接口，以及高效的训练和评估功能，为多智能体强化学习的研究提供了极大的便利。无论是初学者还是资深研究者，都可以利用这个工具包快速开发、测试和优化自己的多智能体算法，在各种复杂的环境中探索前沿的研究问题。


**MARLToolkit** is a *Multi-Agent Reinforcement Learning Toolkit* based on **Pytorch**.
It provides MARL research community a unified platform for developing and evaluating the new ideas in various multi-agent environments.
There are four core features of **MARLToolkit**.

- it collects most of the existing MARL algorithms widely acknowledged by the community and unifies them under one framework.
- it gives a solution that enables different multi-agent environments using the same interface to interact with the agents.
- it guarantees excellent efficiency in both the training and sampling process.
- it provides trained results, including learning curves and pretrained models specific to each task and algorithm's combination, with finetuned hyper-parameters to guarantee credibility.

## Overview

We collected most of the existing multi-agent environment and multi-agent reinforcement learning algorithms and unified them under one framework based on \[**Pytorch**\] to boost the MARL research.

The MARL baselines include **independence learning (IQL, A2C, DDPG, TRPO, PPO)**, **centralized critic learning (COMA, MADDPG, MAPPO, HATRPO)**, and **value decomposition (QMIX, VDN, FACMAC, VDA2C)** are all implemented.

Popular environments like **SMAC, MaMujoco, and Google Research Football** are provided with a unified interface.

The algorithm code and environment code are fully separated. Changing the environment needs no modification on the algorithm side and vice versa.

|                              Benchmark                               |                                                                    Github Stars                                                                     | Learning Mode | Available Env | Algorithm Type | Algorithm Number | Continues Control  | Asynchronous Interact | Distributed Training |                                       Framework                                        |                                                     Last Update                                                     |
| :------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------: | :-----------: | :------------: | :--------------: | :----------------: | :-------------------: | :------------------: | :------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------: |
|             [PyMARL](https://github.com/oxwhirl/pymarl)              |                 [![GitHub stars](https://img.shields.io/github/stars/oxwhirl/pymarl)](https://github.com/oxwhirl/pymarl/stargazers)                 |      CP       |       1       |       VD       |        5         |                    |                       |                      |                                           \*                                           |         ![GitHub last commit](https://img.shields.io/github/last-commit/oxwhirl/pymarl?label=last%20update)         |
|            [PyMARL2](https://github.com/hijkzzz/pymarl2)             |                [![GitHub stars](https://img.shields.io/github/stars/hijkzzz/pymarl2)](https://github.com/hijkzzz/pymarl2/stargazers)                |      CP       |       1       |       VD       |        12        |                    |                       |                      |                      [PyMARL](https://github.com/oxwhirl/pymarl)                       |        ![GitHub last commit](https://img.shields.io/github/last-commit/hijkzzz/pymarl2?label=last%20update)         |
|      [off-policy](https://github.com/marlbenchmark/off-policy)       |                                    ![GitHub stars](https://img.shields.io/github/stars/marlbenchmark/off-policy)                                    |      CP       |       4       |    IL+VD+CC    |        4         |                    |                       |                      |               [off-policy](https://github.com/marlbenchmark/off-policy)                |    ![GitHub last commit](https://img.shields.io/github/last-commit/marlbenchmark/off-policy?label=last%20update)    |
|       [on-policy](https://github.com/marlbenchmark/on-policy)        |                                    ![GitHub stars](https://img.shields.io/github/stars/marlbenchmark/on-policy)                                     |      CP       |       4       |    IL+VD+CC    |        1         |                    |                       |                      |                [on-policy](https://github.com/marlbenchmark/on-policy)                 |    ![GitHub last commit](https://img.shields.io/github/last-commit/marlbenchmark/on-policy?label=last%20update)     |
| [MARL-Algorithms](https://github.com/starry-sky6688/MARL-Algorithms) | [![GitHub stars](https://img.shields.io/github/stars/starry-sky6688/MARL-Algorithms)](https://github.com/starry-sky6688/MARL-Algorithms/stargazers) |      CP       |       1       |    VD+Comm     |        9         |                    |                       |                      |                                           \*                                           | ![GitHub last commit](https://img.shields.io/github/last-commit/starry-sky6688/MARL-Algorithms?label=last%20update) |
|           [EPyMARL](https://github.com/uoe-agents/epymarl)           |         [![GitHub stars](https://img.shields.io/github/stars/uoe-agents/epymarl)](https://github.com/hijkzzz/uoe-agents/epymarl/stargazers)         |      CP       |       4       |    IL+VD+CC    |        10        |                    |                       |                      |                      [PyMARL](https://github.com/oxwhirl/pymarl)                       |       ![GitHub last commit](https://img.shields.io/github/last-commit/uoe-agents/epymarl?label=last%20update)       |
|     [Marlbenchmark](https://github.com/marlbenchmark/on-policy)      |        [![GitHub stars](https://img.shields.io/github/stars/marlbenchmark/on-policy)](https://github.com/marlbenchmark/on-policy/stargazers)        |     CP+CL     |       4       |     VD+CC      |        5         | :heavy_check_mark: |                       |                      | [pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) |    ![GitHub last commit](https://img.shields.io/github/last-commit/marlbenchmark/on-policy?label=last%20update)     |
|             [MAlib](https://github.com/sjtu-marl/malib)              |            [![GitHub stars](https://img.shields.io/github/stars/sjtu-marl/malib)](https://github.com/hijkzzz/sjtu-marl/malib/stargazers)            |      SP       |       8       |       SP       |        9         | :heavy_check_mark: |                       |                      |                                           \*                                           |        ![GitHub last commit](https://img.shields.io/github/last-commit/sjtu-marl/malib?label=last%20update)         |
|        [MARLlib](https://github.com/Replicable-MARL/MARLlib)         |        [![GitHub stars](https://img.shields.io/github/stars/Replicable-MARL/MARLlib)](https://github.com/Replicable-MARL/MARLlib/stargazers)        |  CP+CL+CM+MI  |      10       |    IL+VD+CC    |        18        | :heavy_check_mark: |  :heavy_check_mark:   |  :heavy_check_mark:  |                  [Ray/RLlib](https://docs.ray.io/en/releases-1.8.0/)                   |    ![GitHub last commit](https://img.shields.io/github/last-commit/Replicable-MARL/MARLlib?label=last%20update)     |
|                                                                      |                                                                                                                                                     |               |               |                |                  |                    |                       |                      |                                                                                        |                                                                                                                     |

CP, CL, CM, and MI represent cooperative, collaborative, competitive, and mixed task learning modes.
IL, VD, and CC represent independent learning, value decomposition, and centralized critic categorization. SP represents self-play.
Comm represents communication-based learning.
Asterisk denotes that the benchmark uses its framework.

## Environment

#### Supported Multi-agent Environments / Tasks

Most of the popular environment in MARL research has been incorporated in this benchmark:

| Env Name                                                          | Learning Mode | Observability | Action Space | Observations |
| ----------------------------------------------------------------- | ------------- | ------------- | ------------ | ------------ |
| [LBF](https://github.com/semitable/lb-foraging)                   | Mixed         | Both          | Discrete     | Discrete     |
| [RWARE](https://github.com/semitable/robotic-warehouse)           | Collaborative | Partial       | Discrete     | Discrete     |
| [MPE](https://github.com/openai/multiagent-particle-envs)         | Mixed         | Both          | Both         | Continuous   |
| [SMAC](https://github.com/oxwhirl/smac)                           | Cooperative   | Partial       | Discrete     | Continuous   |
| [MetaDrive](https://github.com/decisionforce/metadrive)           | Collaborative | Partial       | Continuous   | Continuous   |
| [MAgent](https://www.pettingzoo.ml/magent)                        | Mixed         | Partial       | Discrete     | Discrete     |
| [Pommerman](https://github.com/MultiAgentLearning/playground)     | Mixed         | Both          | Discrete     | Discrete     |
| [MaMujoco](https://github.com/schroederdewitt/multiagent_mujoco)  | Cooperative   | Partial       | Continuous   | Continuous   |
| [GRF](https://github.com/google-research/football)                | Collaborative | Full          | Discrete     | Continuous   |
| [Hanabi](https://github.com/deepmind/hanabi-learning-environment) | Cooperative   | Partial       | Discrete     | Discrete     |

Each environment has a readme file, standing as the instruction for this task, talking about env settings, installation, and some important notes.

## Algorithm

We provide three types of MARL algorithms as our baselines including:

**Independent Learning:**
IQL
DDPG
PG
A2C
TRPO
PPO

**Centralized Critic:**
COMA
MADDPG
MAAC
MAPPO
MATRPO
HATRPO
HAPPO

**Value Decomposition:**
VDN
QMIX
FACMAC
VDAC
VDPPO

Here is a chart describing the characteristics of each algorithm:

| Algorithm                                                                                                                  | Support Task Mode | Need Global State | Action     | Learning Mode        | Type       |
| -------------------------------------------------------------------------------------------------------------------------- | ----------------- | ----------------- | ---------- | -------------------- | ---------- |
| IQL                                                                                                                        | Mixed             | No                | Discrete   | Independent Learning | Off Policy |
| [PG](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf) | Mixed             | No                | Both       | Independent Learning | On Policy  |
| [A2C](https://arxiv.org/abs/1602.01783)                                                                                    | Mixed             | No                | Both       | Independent Learning | On Policy  |
| [DDPG](https://arxiv.org/abs/1509.02971)                                                                                   | Mixed             | No                | Continuous | Independent Learning | Off Policy |
| [TRPO](http://proceedings.mlr.press/v37/schulman15.pdf)                                                                    | Mixed             | No                | Both       | Independent Learning | On Policy  |
| [PPO](https://arxiv.org/abs/1707.06347)                                                                                    | Mixed             | No                | Both       | Independent Learning | On Policy  |
| [COMA](https://ojs.aaai.org/index.php/AAAI/article/download/11794/11653)                                                   | Mixed             | Yes               | Both       | Centralized Critic   | On Policy  |
| [MADDPG](https://arxiv.org/abs/1706.02275)                                                                                 | Mixed             | Yes               | Continuous | Centralized Critic   | Off Policy |
| MAA2C                                                                                                                      | Mixed             | Yes               | Both       | Centralized Critic   | On Policy  |
| MATRPO                                                                                                                     | Mixed             | Yes               | Both       | Centralized Critic   | On Policy  |
| [MAPPO](https://arxiv.org/abs/2103.01955)                                                                                  | Mixed             | Yes               | Both       | Centralized Critic   | On Policy  |
| [HATRPO](https://arxiv.org/abs/2109.11251)                                                                                 | Cooperative       | Yes               | Both       | Centralized Critic   | On Policy  |
| [HAPPO](https://arxiv.org/abs/2109.11251)                                                                                  | Cooperative       | Yes               | Both       | Centralized Critic   | On Policy  |
| [VDN](https://arxiv.org/abs/1706.05296)                                                                                    | Cooperative       | No                | Discrete   | Value Decomposition  | Off Policy |
| [QMIX](https://arxiv.org/abs/1803.11485)                                                                                   | Cooperative       | Yes               | Discrete   | Value Decomposition  | Off Policy |
| [FACMAC](https://arxiv.org/abs/2003.06709)                                                                                 | Cooperative       | Yes               | Continuous | Value Decomposition  | Off Policy |
| [VDAC](https://arxiv.org/abs/2007.12306)                                                                                   | Cooperative       | Yes               | Both       | Value Decomposition  | On Policy  |
| VDPPO\*                                                                                                                    | Cooperative       | Yes               | Both       | Value Decomposition  | On Policy  |

*IQL* is the multi-agent version of Q learning.
*MAA2C* and *MATRPO* are the centralized version of A2C and TRPO.
*VDPPO* is the value decomposition version of PPO.
