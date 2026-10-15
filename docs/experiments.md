# 实验指南

每份练习独立初始化环境与网络，互不读取其它练习的变量。Day 编号延续强化学习课程，因此从 Day6 开始。完整代码既可逐单元阅读，也可通过统一入口执行。

## 运行方式

在项目根目录运行：

```bash
python scripts/marl.py list
python scripts/marl.py run qmix
python scripts/marl.py run maddpg --engine notebook --output runs/maddpg-notebook
```

脚本引擎为每份练习启动独立 Python 进程，保存 `stdout.txt`、`stderr.txt` 和 `figure-01.png` 等图。Notebook 引擎启动独立内核，在结果目录保存 `executed.ipynb` 和终端输出。两种入口都保存 `report.json`，验证内嵌检查数并确认原代码未被执行过程修改。

`--timeout 900` 是默认等待上限；脚本引擎按整份练习计时，Notebook 引擎按代码单元计时。失败时报告记为 `failed`，保留已产生的输出并返回非零退出码。再次运行时使用新的输出目录。

Windows 练习在导入 PyTorch 和绘图库前设置本进程的 `MKL_THREADING_LAYER=SEQUENTIAL`，用于避开混用库时的 OpenMP 冲突。若已在 Notebook 中导入这些库，先重启内核再从头执行。统一运行器使用无界面的绘图后端；直接执行 `.py` 时可以看到本机图形窗口。

## Day6

三份练习分别使用三个任务，动作与奖励规则要分开理解：

| 入口 | 决策与数据 | 训练方法 | 检查数 |
|---|---|---|---:|
| `ctde` | 三条手工联合经历；两人选 A/B，相同奖励 2 | 两个局部 Actor、一个集中式 V，一次更新 | 32 |
| `mappo` | 每人只看自己的灯；保持或翻转；最终相同奖励 2，一步终止 | 联合采样、固定目标、逐角色 PPO | 56 |
| `communication` | 发送者看随机方向提示，接收者只看实际消息；方向正确奖励 2 | 两阶段离散 REINFORCE，固定 baseline=1 | 41 |

一步终止的合作任务中，GAE 退化为 `reward-old_value`，Critic 目标就是奖励。默认 PPO 训练为 120 批、每批 64 局、每批最多 4 轮优化。执行仅调用局部 Actor。

通信任务为 300 批、每批 64 局。采样期间保存实际消息，训练重算双方各自选择的 `log_prob`；接收者损失不跨普通整数消息反传给发送者。消息清零、只翻转通道、双方一起重命名符号是三种不同干预。消息没有预先规定的含义，需结合编码和解码一起解读。

## Day7

`qmix-update` 用一条手工经验和单层状态条件线性 Mixer，观察共同 TD 怎样改变两个局部 Q 网络及 Mixer；环境步为 0。`qmix` 才接上完整训练：两层非线性单调 Mixer、两轮灯控、联合回放、Double Q 和周期硬同步。

全局状态为 `[灯0,灯1,阶段]`，本人观测为 `[自己的灯,阶段]`；第一轮共同变亮，第二轮共同变灭。每轮共同成功奖励 1，两轮后真正终止。局部观测足以决定正确动作，因此这里检验价值分解和分散执行，未设置通信需求。

默认训练 2000 局、4000 个共同环境步；第 32 局开始每局更新一次，总计 1969 次更新，每 100 次更新同时同步局部目标网络和目标 Mixer，共 19 次。

```python
next_actions = collect_agent_qs(online_agents, next_obs).argmax(
    dim=-1, keepdim=True,
)
next_utilities = collect_agent_qs(target_agents, next_obs).gather(
    2, next_actions,
).squeeze(-1)
```

在线网络选择动作，目标网络评价这些动作。执行阶段只用局部在线 Q 网络，无需 Mixer。完整训练有 23 项检查，一次更新练习有 36 项。

## Day8

`maddpg` 用两个连续 Actor 和各自的集中式 Q Critic。每轮动作直接设置 `[-1,1]` 数值，团队奖励为双方差距平方的负数；两轮后终止。两人达成任意共同值即可，无隐藏目标或消息。

回放中一条记录包含同一次共同交互的双方观测与动作。Critic 拟合实际联合动作；更新 Actor i 时，只把本人的回放动作替换为当前可微动作，队友继续使用回放值：

```python
own_action = actors[i](batch.obs[:, i])
action_parts = [batch.actions[:, j] for j in range(2)]
action_parts[i] = own_action
actions_for_actor = torch.stack(action_parts, dim=1)
loss = -critic(batch.obs, actions_for_actor).mean()
```

此时冻结 Critic 参数，仍保留动作到评分的梯度；不能把整个分支包进 `no_grad()`。每轮两次 Critic 更新、两次 Actor 更新，再软同步两个目标 Actor 和两个目标 Critic。

默认 2500 局、5000 环境步；第 256 步保存后开始更新，共 4745 轮联合更新、18980 次优化器 step。冻结评估关闭噪声并走完两轮，“成功”指两轮差距都不超过 0.1。练习包含 24 项检查。

## Day9

`memory` 每局四步。开头两人分别看见自己的私有提示，此后三步提示消失；前三步只能等待，第四步各自选门。两人都选中自己的门才得奖励 1。

两份独立 GRU 的 hidden 在新局开始时清零，同局逐步传递。采样用 `no_grad()`，训练从回合起点重建每个人的完整历史。Actor 只在末步计算 PPO 损失，Critic 用四个真实时刻的 GAE 价值目标训练。

默认 160 批×64 局，共 10240 局、40960 条共同转移；每批最多优化四轮，KL 停止可能使实际次数少于 640。正常记忆、选门前重置 hidden、无记忆 MLP 是信息对照。这里没有消息通道；MLP 对照未做完整训练。练习包含 39 项检查。

## Day10

`recurrent-communication` 将双方记忆和真实通信放在同一条四步轨迹里。发送者知道 a，接收者知道 b，二者独立均匀随机；接收者最后要选择 `a XOR b` 对应的门。

| 时间 | 发送者可见内容 | 接收者可见内容 | 自由决策 |
|---|---|---|---|
| t=0 | 自己的 a | 自己的 b | 无，双方等待 |
| t=1 | 阶段与此前记忆 | 阶段与此前记忆 | 发送者选择消息 0/1 |
| t=2 | 阶段与此前记忆 | 实际送达的消息 | 无，双方等待 |
| t=3 | 阶段与此前记忆 | 阶段与此前记忆 | 接收者选择左/右，真正终止 |

两份 GRU 都输入 `[B,T,5]` 的本人历史。五个观测字段是自己的提示是否可见、提示编码、消息是否可见、消息编码、公共阶段。0/1 线索编码为 -1/+1；字段不可见时为 0。

训练按原来的观测历史重算实际消息及选门的概率：

```python
advantage = (batch["rewards"][:, -1] - baseline).detach()
sender_logp = distributions[0].log_prob(batch["actions"][:, 1, 0])
receiver_logp = distributions[1].log_prob(batch["actions"][:, 3, 1])
sender_loss = -(sender_logp * advantage).mean()
receiver_loss = -(receiver_logp * advantage).mean()
```

`baseline=0.5`，两份损失再减去熵奖励。每批重新采样、更新一次；当前实现为 REINFORCE。前面的反事实 Q 数值解释没有替换成训练 Critic，精确枚举也只用于评估。

普通整数消息切断接收端损失到发送端参数的可微路径。发送端通过自己的 `sender_loss` 学习任务奖励；总损失中的发送端熵项也能调整其参数，因此删除 `sender_loss` 后仍观察到参数改变，不能证明接收端任务梯度穿过了整数消息。

默认 800 批×256 局，共 204800 个共同回合、819200 条环境转移和 800 次优化器更新。精确成功率按四组等概率提示、每组两个消息分支求和计算；训练可能先在 50% 附近停留较久。冻结消息、发送前重置发送者、收消息前或选门前重置接收者分别检验信息链路。练习包含 40 项检查。

## 结果与图

笔记中的数值标明其保存输出所用环境；新运行结果以 `runs/.../report.json` 为准。PyTorch 版本、CPU/GPU 与随机数实现可能影响小数值和所学符号约定。有限采样全部成功与精确策略概率等于 1 是不同结论。

训练时有探索噪声或 epsilon，冻结评估关闭它们。训练曲线中的均值与评估成绩应在相同信息、噪声、起点及终止条件下比较。loss 衡量目标拟合；轨迹、精确枚举和冻结干预检验行为及信息使用。

配套笔记使用已有实际数据图；重新运行训练会在结果目录产生该次曲线。笔记引用的原本地绘图脚本依赖个人工作区的历史记录，在公开文档中统一指向本指南，复现入口为当前仓库的训练代码。
