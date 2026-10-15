# Generated from the adjacent notebook by scripts/marl.py export.

# %% [markdown]
# # Day7：QMIX 风格单步更新
#
# 这一份练习只回答一个问题：
#
# > 一个团队 TD 目标，怎样通过一次 `optimizer.step()` 同时更新多个局部网络和 Mixer？
#
# 教师提供的分段练习，供 RedMoon 阅读、修改与复跑。以下观测、奖励与动作是手工构造的一条教学经历，网络是随机初始化的；输出不代表已学会合作。
#
# 本练习会把一条联合经历完整走一遍：
#
# ```text
# 当前局部网络
#     -> 当前实际动作的局部 utility
#     -> online Mixer
#     -> 当前团队价值 Q_total
#
# 下一状态的目标局部网络
#     -> 每个智能体各自取最大动作价值
#     -> target Mixer
#     -> 下一步团队价值
#
# 奖励 + 折扣后的下一步团队价值
#     -> 一个共同 TD 目标
#     -> 一个共同损失
#     -> 更新在线局部网络与在线 Mixer
# ```
#
# 这里使用的是一个**简化的单层单调 Mixer**：全局状态通过超网络生成非负混合权重；固定全局状态和其他输入时，某个局部 utility 增大，团队价值不会下降。它是状态条件线性加和，不是此前介绍的两层非线性 QMIX Mixer；本节先沿用简单结构观察一次参数更新，没有环境、回放池、批量训练、RNN 历史状态、Double Q 或多回合评估。

# %% [markdown]
# ## 1. 先固定本节的变量含义
#
# 为了避免一上来被张量形状绕晕，先把每个变量翻译成中文：
#
# | 变量 | 含义 |
# |---|---|
# | `n_agents` | 智能体数量，本例为 2 |
# | `n_actions` | 每个智能体可选动作数，本例为 2 |
# | `obs` | 当前时刻每个智能体自己的局部观测 |
# | `next_obs` | 下一时刻每个智能体自己的局部观测 |
# | `state` | 训练阶段可用的全局状态 |
# | `actions` | 这条经历中真实执行过的联合动作 |
# | `reward` | 团队共享奖励，只记录一次 |
# | `terminated` | 任务是否真正结束 |
# | `agent_qs` | 局部网络对各动作给出的 utility |
# | `q_total` | Mixer 合成的团队价值 |
#
# 本例张量形状统一写在注释里。看到形状时，先问自己：
#
# > 这一个维度是在区分批次、智能体、动作，还是状态特征？

# %%
# 导入依赖，并固定随机种子，同一执行环境下初始化可复现。
import copy
import json

import torch
from torch import nn

torch.manual_seed(7)
torch.set_num_threads(1)
torch.set_printoptions(precision=4, sci_mode=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checks_passed = 0


def check(condition, name):
    # 只有真正通过断言才计数，不手写通过总数。
    global checks_passed
    assert bool(condition), name
    checks_passed += 1
    print("PASS:", name)


n_agents = 2
n_actions = 2
obs_dim = 3
state_dim = 4
hidden_dim = 8

print("device:", device)
print("n_agents:", n_agents, "n_actions:", n_actions)

# %% [markdown]
# ## 2. 构造一条联合经历
#
# 这次只放一条经历，因此批次大小 `batch_size=1`。
#
# 注意 `actions = [0, 1]` 的意思是：
#
# - 智能体 0 实际执行动作 0；
# - 智能体 1 实际执行动作 1。
#
# 训练当前价值时，必须取出这两个**真实执行动作**对应的 utility，不能偷偷改成当前网络此刻认为最好的动作。

# %%
# 每个智能体只看到自己的局部观测。
# obs / next_obs: [batch_size, n_agents, obs_dim]
obs = torch.tensor(
    [[[1.0, 0.0, 0.5],
      [0.0, 1.0, 0.2]]],
    dtype=torch.float32,
    device=device,
)
next_obs = torch.tensor(
    [[[0.8, 0.1, 0.4],
      [0.1, 0.9, 0.3]]],
    dtype=torch.float32,
    device=device,
)

# 训练阶段使用的全局状态；执行阶段不把它交给局部 Q 网络。
# state / next_state: [batch_size, state_dim]
state = torch.tensor([[0.6, 0.2, 0.4, 0.8]], dtype=torch.float32, device=device)
next_state = torch.tensor([[0.7, 0.1, 0.5, 0.6]], dtype=torch.float32, device=device)

# 这条经历真实执行的联合动作：[agent0_action, agent1_action]
# actions: [batch_size, n_agents]
actions = torch.tensor([[0, 1]], dtype=torch.long, device=device)

# 团队奖励只放一个数，不为每个智能体重复复制一份。
reward = torch.tensor([2.0], dtype=torch.float32, device=device)
terminated = torch.tensor([False], dtype=torch.bool, device=device)
gamma = 0.9

print("obs shape:", tuple(obs.shape))
print("state shape:", tuple(state.shape))
print("actions:", actions.tolist())
print("reward:", reward.tolist(), "terminated:", terminated.tolist())

# %% [markdown]
# ## 3. 两个局部 Q 网络
#
# 每个局部网络只接收自己的观测，输出自己对两个动作的评分：
#
# ```text
# 自己的观测 -> 自己的局部网络 -> [动作0评分, 动作1评分]
# ```
#
# 它不会看到队友观测，也不会直接看到全局状态。

# %%
class LocalQNetwork(nn.Module):
    """一个教学用局部 Q 网络：局部观测 -> 每个动作的 utility。"""

    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, local_obs: torch.Tensor) -> torch.Tensor:
        # local_obs: [batch_size, obs_dim]
        # 返回: [batch_size, n_actions]
        return self.net(local_obs)


online_agents = nn.ModuleList(
    [LocalQNetwork(obs_dim, hidden_dim, n_actions).to(device) for _ in range(n_agents)]
)
target_agents = copy.deepcopy(online_agents)
online_agents.train()
target_agents.eval()

for network in target_agents:
    for parameter in network.parameters():
        parameter.requires_grad_(False)


def collect_agent_qs(
    networks: nn.ModuleList,
    observations: torch.Tensor,
) -> torch.Tensor:
    """逐个运行局部网络，整理成 [batch, n_agents, n_actions]。"""
    per_agent = []
    for agent_id, network in enumerate(networks):
        # observations[:, agent_id, :] -> [batch_size, obs_dim]
        q_values = network(observations[:, agent_id, :])
        per_agent.append(q_values)
    return torch.stack(per_agent, dim=1)


online_q_all = collect_agent_qs(online_agents, obs)
print("online_q_all shape:", tuple(online_q_all.shape))
check(online_q_all.shape == (1, n_agents, n_actions), "local Q output shape")

# %% [markdown]
# ## 4. 一个简化的状态条件 Mixer
#
# 完整 QMIX 的超网络会根据全局状态生成 Mixer 的参数。本节保留最关键的结构：
#
# ```text
# 全局状态 -> 超网络 -> 非负混合权重和偏置
# 局部 utility + 混合权重 -> 团队 Q_total
# ```
#
# 这里用 `abs()` 保证权重非负。因为 Mixer 对每个 utility 只是正权重加和，所以固定状态下，某个 utility 增大不会让团队价值下降。
#
# 偏置不强制非负，因此团队价值本身完全可以是负数。

# %%
class TinyMonotonicMixer(nn.Module):
    """保留 QMIX 核心约束的单层教学版 Mixer。"""

    def __init__(self, state_dim: int, n_agents: int):
        super().__init__()
        # 超网络：读取全局状态，生成每个 agent 的混合权重。
        self.hyper_w = nn.Linear(state_dim, n_agents)
        # 超网络：生成一个团队层面的偏置。
        self.hyper_b = nn.Linear(state_dim, 1)

    def forward(
        self,
        agent_qs: torch.Tensor,
        global_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # agent_qs: [batch_size, n_agents]
        # global_state: [batch_size, state_dim]
        raw_weights = self.hyper_w(global_state)
        weights = raw_weights.abs()
        bias = self.hyper_b(global_state).squeeze(-1)

        # 非负权重加和：输出 [batch_size] 的团队价值。
        q_total = (agent_qs * weights).sum(dim=-1) + bias
        return q_total, weights


online_mixer = TinyMonotonicMixer(state_dim, n_agents).to(device)
target_mixer = copy.deepcopy(online_mixer)
online_mixer.train()
target_mixer.eval()
for parameter in target_mixer.parameters():
    parameter.requires_grad_(False)

print("online mixer parameters:", sum(p.numel() for p in online_mixer.parameters()))

# %% [markdown]
# ## 5. 当前团队价值：只取真实执行动作
#
# 局部网络输出的是所有动作的评分。当前这条经历实际执行的是 `[0, 1]`，因此要按动作索引取出：
#
# ```text
# agent0: Q0(action0)
# agent1: Q1(action1)
# ```
#
# 不能在这里使用 `argmax()`，因为 `argmax()` 得到的是当前网络此刻最想执行的动作，而不是这条经历真实执行过的动作。

# %%
# 从所有动作评分中，取出这条经历真实执行的动作。
# online_q_all: [batch, n_agents, n_actions]
# actions.unsqueeze(-1): [batch, n_agents, 1]
chosen_qs = online_q_all.gather(dim=2, index=actions.unsqueeze(-1)).squeeze(-1)
# chosen_qs: [batch, n_agents]

q_total_current, current_weights = online_mixer(chosen_qs, state)

print("chosen local utility:", chosen_qs.detach().cpu())
print("current mixer weights:", current_weights.detach().cpu())
print("current team Q:", q_total_current.detach().cpu())

check(chosen_qs.shape == (1, n_agents), "selected utility shape")
check(q_total_current.shape == (1,), "team Q shape")
check(torch.all(current_weights >= 0), "nonnegative mixer weights")
check(torch.equal(chosen_qs, torch.stack([
    online_q_all[:, 0, 0], online_q_all[:, 1, 1]
], dim=1)), "gather selects each actual action")

# %% [markdown]
# ### 保存输出解读：当前团队 Q 是怎样组成的
#
# 以下解析于 **2026-10-10** 根据当前已保存输出补充，没有替换为教师重新运行的结果。本次输出显示设备为 `cuda`；以下数字是本次快照，重新运行后应以新输出为准。
#
# `obs shape = (1, 2, 3)` 表示 1 条经历、2 名智能体、每人 3 个局部特征；`online_q_all shape = (1, 2, 2)` 的最后一个 2 是每人两个动作的评分。经历中实际执行 `[0, 1]`，所以 `gather` 取的是智能体 0 的动作 0、智能体 1 的动作 1，得到 `[0.2346437, -0.0079036]`，不是当前各自的最大值。
#
# 当前状态生成的非负混合权重约为 `[0.6984, 0.1253]`。本例团队预测按下面的形式计算：
#
# ```text
# Q_team = w0(state) × utility0 + w1(state) × utility1 + bias(state)
#        ≈ 0.3083004
# ```
#
# 还有状态生成的偏置，不能直接把两个局部 utility 相加来复核这个数。局部 utility 和团队 Q 都不是概率，负值合法，也没有“两个值必须加起来为 1”的要求。
#
# 权重非负保证固定状态、固定 Mixer 参数时，提高一个 utility 不会降低团队预测；它不要求 utility 为正，也不说明两个 utility 是奖励份额。Mixer 的 15 个参数来自权重超网络 `4 × 2 + 2 = 10` 和偏置超网络 `4 × 1 + 1 = 5`，不是 15 个状态或 15 次训练。

# %% [markdown]
# ## 6. 下一状态的目标团队价值：各自取最大，再交给目标 Mixer
#
# 下一状态不是当前经历的实际动作，所以这里才使用目标网络对每个智能体取最大动作价值：
#
# ```text
# agent0: max_a Q_target_0(next_obs_0, a)
# agent1: max_a Q_target_1(next_obs_1, a)
# ```
#
# 然后把两个最大局部 utility 交给目标 Mixer。整个目标分支都放在 `torch.no_grad()` 中，因为本次更新只训练在线网络，不能让目标网络被这条损失直接更新。

# %%
with torch.no_grad():
    target_q_all = collect_agent_qs(target_agents, next_obs)
    target_max_qs = target_q_all.max(dim=2).values
    q_total_next, target_weights = target_mixer(target_max_qs, next_state)

    # 真正终止时不再读取未来团队价值；未终止时保留 bootstrap。
    not_terminated = (~terminated).float()
    td_target = reward + gamma * not_terminated * q_total_next

print("target local max utility:", target_max_qs.cpu())
print("next team Q:", q_total_next.cpu())
print("TD target:", td_target.cpu())

check(target_max_qs.shape == (1, n_agents), "target local max shape")
check(q_total_next.shape == (1,), "target team Q shape")
check(td_target.shape == (1,), "TD target shape")
check(not td_target.requires_grad, "TD target has no gradient graph")
check(torch.allclose(td_target, reward + gamma * q_total_next),
      "one reward plus discounted next team Q")

# 另设真正终止标记，只检查公式，不执行第二次参数更新。
terminal_target = reward + gamma * torch.zeros_like(reward) * q_total_next
check(torch.equal(terminal_target, reward), "true terminal target is current reward")

# %% [markdown]
# ## 7. 一个共同团队损失
#
# 现在只有一个比较：
#
# ```text
# 当前团队预测 q_total_current
# 目标团队价值 td_target
# ```
#
# 局部 utility 不分别拟合 `td_target`。它们通过 Mixer 连接到同一个团队损失：

# %%
# 只对团队总价值计算一个 TD 损失。
loss_before = torch.mean((q_total_current - td_target.detach()) ** 2)

print("current team Q:", q_total_current.detach().cpu())
print("fixed TD target:", td_target.detach().cpu())
print("loss before step:", float(loss_before.detach().cpu()))

check(loss_before.ndim == 0, "loss is a scalar")

# %% [markdown]
# ### 保存输出解读：2.2752 的共同目标和 3.8685 的损失
#
# 目标局部网络在**下一观测**上的各自最大 utility 约为 `[0.2383, 0.1027]`；把它们与**下一全局状态**交给目标 Mixer，得到下一团队 Q 约为 0.3057。它和当前经历实际动作 `[0, 1]` 的当前评分不是同一组量。
#
# 这条经历 `terminated=False`，所以保留未来预测：
#
# ```text
# 共同 TD 目标 = 当前团队奖励 + gamma × 下一团队 Q
#             ≈ 2 + 0.9 × 0.30572785
#             ≈ 2.27515507
# 共同 TD 损失 = (当前团队 Q - 固定 TD 目标)^2
#             ≈ (0.30830038 - 2.27515507)^2
#             ≈ 3.86851740
# ```
#
# 输出张量只显示四位小数，但计算使用完整精度；用显示的 `0.3057` 手算会有微小差异。当前奖励 2 只加一次，没有按两名智能体重复加成 4。如果真正终止，未来项变为 0，目标就等于 2；本单元的终止断言只是另验公式，没有额外执行一次参数更新。
#
# 这里仅有一个团队损失，优化的是 Mixer 输出。两个局部 utility **不是分别以 2.275155 为目标做 MSE**，而是通过混合关系共同影响团队误差。目标分支在 `no_grad` 下计算，随后保持为固定训练参照。

# %% [markdown]
# ## 8. 反向传播：谁会收到梯度？
#
# `loss.backward()` 只产生梯度，还没有真正改参数。
#
# 理论上的梯度路径是：
#
# ```text
# team loss
#   -> online mixer
#   -> chosen local utility
#   -> 两个 online local Q 网络
# ```
#
# 目标局部网络和目标 Mixer 在目标分支中已经被 `no_grad()` 保护，所以不会收到本次更新的梯度。

# %%
# 保存参数数值，分别比较 backward 和 step 之后是否改变。
def clone_parameters(module: nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in module.parameters()]


def parameters_changed(before: list[torch.Tensor], module: nn.Module) -> bool:
    return any(
        not torch.equal(old, new.detach())
        for old, new in zip(before, module.parameters())
    )


online_agent_before = [clone_parameters(network) for network in online_agents]
online_mixer_before = clone_parameters(online_mixer)
target_agent_before = [clone_parameters(network) for network in target_agents]
target_mixer_before = clone_parameters(target_mixer)

# 优化器只接收在线局部网络和在线 Mixer。
online_parameters = list(online_mixer.parameters())
for network in online_agents:
    online_parameters.extend(network.parameters())

optimizer = torch.optim.SGD(online_parameters, lr=0.05)
optimizer.zero_grad(set_to_none=True)
loss_before.backward()

online_grad_norm = 0.0
for parameter in online_parameters:
    if parameter.grad is not None:
        online_grad_norm += float(parameter.grad.detach().norm().cpu())

target_grad_count = sum(
    parameter.grad is not None
    for network in target_agents
    for parameter in network.parameters()
) + sum(parameter.grad is not None for parameter in target_mixer.parameters())

print("sum of online gradient norms:", online_grad_norm)
print("number of target tensors with gradients:", target_grad_count)

check(online_grad_norm > 0.0, "online parameters receive gradients")
check(target_grad_count == 0, "target parameters have no gradients")
for agent_id, network in enumerate(online_agents):
    check(any(p.grad is not None and p.grad.abs().sum().item() > 0
              for p in network.parameters()),
          f"online agent {agent_id} receives a nonzero gradient")
check(any(p.grad is not None and p.grad.abs().sum().item() > 0
          for p in online_mixer.hyper_w.parameters()),
      "weight hypernetwork receives a gradient")
check(any(p.grad is not None and p.grad.abs().sum().item() > 0
          for p in online_mixer.hyper_b.parameters()),
      "bias hypernetwork receives a gradient")
check(all(not parameters_changed(old, net) for old, net in zip(
    online_agent_before, online_agents
)) and not parameters_changed(online_mixer_before, online_mixer),
    "backward alone does not modify online parameters")

# %% [markdown]
# ## 9. 真正执行一次 `optimizer.step()`
#
# 这一步才是真正的参数更新：
#
# ```python
# optimizer.step()
# ```
#
# 预期现象：
#
# - 在线局部网络参数发生变化；
# - 在线 Mixer / 超网络参数发生变化；
# - 目标局部网络参数不变；
# - 目标 Mixer 参数不变；
# - 同一条经历、固定目标下，重新计算损失检查本例是否降低。一般非线性网络用有限学习率更新一次，不保证所有输出都升高或损失必定降低。

# %%
# 真正修改参数的是 step，而不是前面的 backward。
optimizer.step()


online_agents_changed = [
    parameters_changed(before, network)
    for before, network in zip(online_agent_before, online_agents)
]
online_mixer_changed = parameters_changed(online_mixer_before, online_mixer)
target_agents_changed = [
    parameters_changed(before, network)
    for before, network in zip(target_agent_before, target_agents)
]
target_mixer_changed = parameters_changed(target_mixer_before, target_mixer)

print("online agent parameters changed:", online_agents_changed)
print("online mixer parameters changed:", online_mixer_changed)
print("target agent parameters changed:", target_agents_changed)
print("target mixer parameters changed:", target_mixer_changed)

check(all(online_agents_changed), "both online agents changed after step")
check(online_mixer_changed, "online hypernetwork parameters changed")
check(not any(target_agents_changed), "target local parameters unchanged")
check(not target_mixer_changed, "target mixer parameters unchanged")

# %% [markdown]
# ### 保存输出解读：backward、step、目标副本分别发生了什么
#
# 反传输出 `sum of online gradient norms = 20.6891082`。代码对每个参数张量的梯度分别求范数，再把范数相加；**它不是把所有梯度拼接后得到的全局 L2 范数**，不能直接拿这个数和常见梯度裁剪阈值比较，更不能把它当成功率。
#
# `number of target tensors with gradients = 0` 表示目标网络参数没有收到反传梯度，不是目标网络的预测为零。检查同时确认两个在线局部网络、权重超网络和偏置超网络均收到非零梯度。
#
# | 阶段 | 在线局部网络 | 在线超网络 / Mixer | 目标局部网络和目标 Mixer |
# |---|---|---|---|
# | 仅执行 `backward()` | 产生梯度，参数未变 | 产生梯度，参数未变 | 无梯度，参数未变 |
# | 随后一次 `optimizer.step()` | 两个网络参数均改变 | 参数改变 | 参数仍不变 |
#
# 梯度沿共同团队损失经过 Mixer 传到两人的局部网络。同一团队误差不代表两人收到相同梯度：各自的混合权重和局部网络导数都参与链式计算。
#
# 本次优化器只持有在线参数，学习率为 0.05；真正改参数的是 `step()`，目标副本需要后面的显式同步才会改。当前这些检查说明更新路径接通，不说明任何环境任务已完成。

# %% [markdown]
# ## 10. 更新后重新计算当前预测
#
# 目标值 `td_target` 是本次更新时固定下来的参照。更新在线参数后，重新用同一条当前经历计算 `q_total_current`，就能观察在线网络是否改变了自己的预测。
#
# 注意：一次 `optimizer.step()` 不保证预测恰好等于目标；学习率过大也可能让损失上升。这里检查当前默认数值下的真实结果，不把它当成普遍定律。

# %%
online_q_all_after = collect_agent_qs(online_agents, obs)
chosen_qs_after = online_q_all_after.gather(
    dim=2,
    index=actions.unsqueeze(-1),
).squeeze(-1)
q_total_after, weights_after = online_mixer(chosen_qs_after, state)
loss_after = torch.mean((q_total_after - td_target.detach()) ** 2)

print("current team Q before:", q_total_current.detach().cpu())
print("current team Q after :", q_total_after.detach().cpu())
print("fixed TD target      :", td_target.detach().cpu())
print("loss before step     :", float(loss_before.detach().cpu()))
print("loss after step      :", float(loss_after.detach().cpu()))
print("local utility before :", chosen_qs.detach().cpu())
print("local utility after  :", chosen_qs_after.detach().cpu())

check(q_total_after.shape == (1,), "updated team Q shape")
check(loss_after.item() < loss_before.item(), "loss decreases in this configured example")
check(torch.isfinite(loss_after), "updated loss is finite")
with torch.no_grad():
    next_q_again, _ = target_mixer(
        collect_agent_qs(target_agents, next_obs).max(dim=2).values, next_state
    )
    target_again = reward + gamma * (~terminated).float() * next_q_again
check(torch.equal(target_again, td_target), "target copies recompute unchanged target")

# 固定 state，只提高一个局部 utility，检查更新后的 Mixer 仍单调。
with torch.no_grad():
    increased_qs = chosen_qs_after.detach().clone()
    increased_qs[:, 0] += 0.1
    increased_team_q, _ = online_mixer(increased_qs, state)
check(torch.all(increased_team_q >= q_total_after.detach()),
      "updated mixer remains monotonic at fixed state")

# %% [markdown]
# ### 保存输出解读：同一条经历的一次更新确实更接近固定目标
#
# | 数量 | 更新前 | 一次更新后 |
# |---|---:|---:|
# | 当前团队 Q | 0.3083004 | 1.1381497 |
# | 本次固定 TD 目标 | 2.2751551 | 2.2751551 |
# | 同一经历的团队 MSE | 3.8685174 | 1.2927811 |
# | 智能体 0 实际动作的 utility | 0.2346437 | 0.6928017 |
# | 智能体 1 实际动作的 utility | -0.0079036 | 0.0478739 |
#
# 更新前团队预测低于目标，本次 SGD 后预测提高，离固定目标更近，因此平方误差下降。这里固定的是**同一个目标**，否则用更新前后的两个不同目标比较损失，就不能单独归因于当前预测改善。
#
# 两个 utility 本次也都提高，但在线混合权重和偏置同样发生了更新，不能把团队 Q 的变化完全归因于其中一个局部数值，更不能把提高的幅度解释成各自贡献比例。
#
# 重新调用尚未同步的目标副本，得到的目标仍与原来的 `2.2751551` 相同；这验证了在线更新没有偷偷修改目标网络。固定状态、只人为提高一个 utility 的检查，也确认更新后的 Mixer 仍保持单调。
#
# **这份输出证明的是一条教学经历上的 TD 拟合改善。**没有采集环境回合、没有验证真实奖励或成功率，而且下一状态价值仍是网络估计，不能把本次 loss 下降直接写成“两个智能体已经学会合作”。

# %% [markdown]
# ## 11. 用一句话解释这次更新
#
# 这次更新不是：
#
# ```text
# 智能体 0 各自拟合一个奖励
# 智能体 1 各自拟合一个奖励
# ```
#
# 而是：
#
# ```text
# 两个局部网络共同产生一个联合动作的团队预测
# 团队预测与团队目标比较
# 共同误差沿 Mixer 反向传给在线网络
# ```
#
# 还要记住目标网络的职责：
#
# ```text
# 在线网络：本次被梯度更新
# 目标网络：本次只提供相对稳定的目标参照
# ```

# %%
# 分散执行函数的参数列表里没有 global_state 或 mixer。
@torch.no_grad()
def decentralized_actions(networks, local_observations):
    execution_actions = []
    for agent_id, network in enumerate(networks):
        local_scores = network(local_observations[:, agent_id, :])
        execution_actions.append(local_scores.argmax(dim=-1))
    return torch.stack(execution_actions, dim=1)


online_agents.eval()
execution_actions = decentralized_actions(online_agents, obs)
print("decentralized execution actions:", execution_actions.cpu().tolist())
check(execution_actions.shape == (1, n_agents), "decentralized action shape")
check(all(0 <= a < n_actions for a in execution_actions.flatten().tolist()),
      "decentralized actions are legal")
# 这里只跑局部网络前向，不采环境、不计算合作成功率。

# %% [markdown]
# ## 12. 目标同步：局部目标网络和目标 Mixer 一起复制
#
# 在线参数训练一段时间后，可以做硬同步：
#
# ```text
# online 局部网络 -> target 局部网络
# online Mixer   -> target Mixer
# ```
#
# 只同步局部网络而遗漏 Mixer，会把不同时间的参数拼在一起。这里单独手动同步一次，让你看到 `step` 和 `load_state_dict` 的不同职责，没有实现周期调度。
#
# 同步改变的是目标副本数值，不会把它们加入在线优化器，也不会修改已经算好的 `td_target` 张量。下次重新构造目标时，才会用到同步后的参数。

# %%
fixed_target_copy = td_target.clone()
online_params_before_sync = [clone_parameters(net) for net in online_agents]
online_mixer_before_sync = clone_parameters(online_mixer)

target_agents.load_state_dict(online_agents.state_dict())
target_mixer.load_state_dict(online_mixer.state_dict())

check(all(torch.equal(p, t) for p, t in zip(
    online_agents.parameters(), target_agents.parameters()
)), "local target copies equal online after sync")
check(all(torch.equal(p, t) for p, t in zip(
    online_mixer.parameters(), target_mixer.parameters()
)), "target mixer equals online after sync")
check(all(not parameters_changed(old, net) for old, net in zip(
    online_params_before_sync, online_agents
)) and not parameters_changed(online_mixer_before_sync, online_mixer),
    "sync does not modify online parameters")
check(torch.equal(td_target, fixed_target_copy), "sync does not rewrite existing target tensor")
check(all(not p.requires_grad for p in target_agents.parameters())
      and all(not p.requires_grad for p in target_mixer.parameters()),
      "target copies remain excluded from gradient training")

with torch.no_grad():
    synced_next_qs = collect_agent_qs(target_agents, next_obs).max(dim=2).values
    synced_next_team_q, _ = target_mixer(synced_next_qs, next_state)
    new_td_target = reward + gamma * (~terminated).float() * synced_next_team_q

print("old fixed target:", td_target.cpu())
print("newly computed target after sync:", new_td_target.cpu())
check(torch.isfinite(new_td_target).all(), "recomputed target is finite")

summary = {
    "checks_passed": checks_passed,
    "parameter_updates": 1,
    "target_syncs": 1,
    "environment_steps": 0,
    "device": str(device),
    "team_q_before": q_total_current.detach().cpu().item(),
    "team_q_after": q_total_after.detach().cpu().item(),
    "fixed_td_target": td_target.cpu().item(),
    "loss_before": loss_before.detach().cpu().item(),
    "loss_after": loss_after.detach().cpu().item(),
    "chosen_utility_before": chosen_qs.detach().cpu().tolist(),
    "chosen_utility_after": chosen_qs_after.detach().cpu().tolist(),
    "online_agents_changed": online_agents_changed,
    "online_mixer_changed": online_mixer_changed,
    "target_agents_changed_by_step": target_agents_changed,
    "target_mixer_changed_by_step": target_mixer_changed,
    "execution_actions": execution_actions.cpu().tolist(),
    "target_after_sync": new_td_target.cpu().item(),
}
print("SUMMARY_JSON=" + json.dumps(summary, ensure_ascii=True))

# %% [markdown]
# ### 保存输出解读：执行动作与同步后新目标不是训练成功指标
#
# 局部贪心执行输出 `[0, 0]`，与教学经历保存的实际动作 `[0, 1]` 不同。前者是更新后对当前观测重新取 `argmax`，后者是用于训练的既有动作；重新执行时不要求照抄旧动作。仅凭动作变了，也不能说变得更好，因为这里没有交给环境得到新奖励。某动作的 utility 提高，也不保证它高过同状态另一个动作。
#
# 硬同步分别复制在线局部网络到目标局部网络，以及在线 Mixer 到目标 Mixer。同步本身不改在线参数、不执行优化器更新，也不把已经保存的 `td_target` 张量自动改写。
#
# | 比较项 | 当前保存值 |
# |---|---:|
# | 更新时使用的旧目标 | 2.2751551 |
# | 同步后原目标张量 | 仍为 2.2751551 |
# | 同步后重新前向计算的新目标 | 2.9790525 |
#
# 新目标升高，是因为目标网络复制了已更新的在线参数，对下一观测和下一状态的估计改变了；奖励仍为 2、gamma 仍为 0.9、终止标记仍为假。新算出的目标可以用于之后的更新，但本例没有再用它执行第二次 `step()`，也没有把它混入上面的更新前后损失比较。
#
# 末尾摘要记录 **36 项检查、1 次参数更新、1 次目标同步、0 个环境步，设备 cuda**。它是单层状态条件线性单调 Mixer 的数据流教学，不是标准两层非线性 QMIX 完整训练，也没有合作成功率可报告。

# %% [markdown]
# ## 本练习的边界
#
# - 这是一个 QMIX 风格单步更新教学实验，不是完整 QMIX 训练工程。
# - 只使用一条联合经历，没有环境交互、回放池、批量采样或多回合曲线。
# - Mixer 是单层状态条件非负加权和，目的是展示超网络、Mixer、共同 TD 损失和目标网络之间的数据流。
# - `optimizer.step()` 只执行一次，不能据此声称策略已经学会合作。
# - 执行阶段只运行局部网络，Notebook 中的全局状态、超网络和 Mixer 只服务于集中训练。
