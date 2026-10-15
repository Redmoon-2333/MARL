# Day1–Day10 阅读索引

十天的正文、关键代码和配图按原学习顺序保留。每篇均可直接阅读，文末附完整练习入口；Notebook 保存逐单元说明与输出，Python 文件提供相同代码的分段版本。

| 天数 | 主题 | 阅读重点 |
|---|---|---|
| [Day1](Day1.md) | 奖励、回报、Q-learning 与 SARSA | 一步奖励怎样连成长期回报，TD target 怎样更新 Q 表 |
| [Day2](Day2.md) | DQN 的一次更新 | 实际动作的 gather、目标网络、损失、梯度与同步 |
| [Day3](Day3.md) | Actor–Critic | 动作概率、状态价值、单步 TD 优势与完整轨迹 |
| [Day4](Day4.md) | REINFORCE、baseline 与 GAE | 整局回报、策略损失、相对评分、终止与截断、固定目标 |
| [Day5](Day5.md) | PPO 训练闭环 | 固定旧概率、正负优势裁剪、有限多轮更新与冻结评估 |
| [Day6](Day6.md) | 合作训练与通信 | CTDE、逐角色 PPO、共同评分、反事实思路与消息协议 |
| [Day7](Day7.md) | VDN 与 QMIX | 局部 utility、状态条件混合、联合回放、Double Q 与分散执行 |
| [Day8](Day8.md) | MADDPG 风格连续协调 | 联合动作价值、本人的动作梯度与目标软同步 |
| [Day9](Day9.md) | 循环 Actor 与完整序列 | hidden 的边界、时间顺序、GAE/PPO 与记忆清零干预 |
| [Day10](Day10.md) | 记忆、消息与团队评分 | 双方私有线索、延迟离散通信、两份策略损失及信息干预 |

[Day8 前的一个小总结](before-day8-summary.md)放在 Day7 后复习，集中澄清合作与必要通信、预测与回报、共同奖励与个体贡献、模型价值与真实表现。

需要运行时，参阅[项目首页](../README.md)和[实验指南](experiments.md)。[核验数据](validation.json)记录默认 CPU 配置下两种入口的实际输出。
