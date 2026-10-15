# Generated from the adjacent notebook by scripts/marl.py export.

# %% [markdown]
# # Day4：走完一局，再做一次 REINFORCE 更新
#
# 学习者：RedMoon。
#
# 前面分别练过动作分布、回报和损失求和。这次把它们接起来：**策略采样动作 → 环境走到终点 → 计算各步回报 → 构造损失 → 反向传播 → 更新策略**。
#
# 本练习只更新一局，不是训练到收敛。它使用 Day3 的两路线环境，没有 Critic、baseline、经验回放或 PPO。
#
# 按单元从上到下运行。想再跑一次完整实验时，重新启动内核并运行所有单元；不要只重复执行最后的 `backward()` 单元，因为该次计算图已经使用过。
#
# 代码和讲解由互动教学提供；这里的执行结果用于核对链路，不自动表示已经独立掌握算法。

# %% [markdown]
# ## 1. 导入工具，确定两个超参数
#
# - `gamma`：折扣因子，用于比较现在与未来的奖励。
# - `learning_rate`：SGD 每次沿梯度改变参数的幅度。
#
# 整个练习只需要 PyTorch，不需要安装新的环境库。

# %%
import torch
from torch import nn
from torch.distributions import Categorical

gamma = 0.9
learning_rate = 0.1
torch.manual_seed(1)

print("PyTorch：", torch.__version__)

# %% [markdown]
# ## 2. 环境：最多两步就到终点
#
# 状态 0 是起点，状态 1 是中间状态，`None` 表示回合已结束。
#
# ```text
# 长路线：状态0 --动作0，奖励0--> 状态1 --任意动作，奖励2--> 终点
# 短路线：状态0 --动作1，奖励1----------------------------> 终点
# ```
#
# 长路线起点的折扣回报是 $0+0.9\times2=1.8$；短路线是 $1$。
#
# `terminated=False` 时必须继续行动；`terminated=True` 才结束本局。这两条路线都合法，采样到短路线也不是代码出错。

# %%
class TinyChainEnv:
    """两个非终止状态、两个动作的教学环境。"""

    def __init__(self):
        self.n_states = 2
        self.n_actions = 2
        self.current_state = 0

    def reset(self):
        self.current_state = 0
        return self.current_state

    def step(self, action):
        if action not in (0, 1):
            raise ValueError("action 必须是 0 或 1")

        if self.current_state == 0:
            if action == 0:
                next_state, reward, terminated = 1, 0.0, False
            else:
                next_state, reward, terminated = None, 1.0, True
        elif self.current_state == 1:
            next_state, reward, terminated = None, 2.0, True
        else:
            raise RuntimeError("回合已结束，请先 reset()")

        self.current_state = next_state
        return next_state, reward, terminated, {}


env = TinyChainEnv()

# %% [markdown]
# ## 3. 策略：状态进来，两个动作的得分出去
#
# 使用 one-hot 表示状态：状态 0 是 `[[1, 0]]`，状态 1 是 `[[0, 1]]`。形状 `[1, 2]` 中，第一个 `1` 表示这次只输入一个状态。
#
# 为了观察一次更新，用一个无偏置的线性层作为策略，不增加隐藏层。它仍然是可以训练的策略网络。
#
# 将初始权重设为 0，两个动作初始概率都为 `0.5`，方便核对。这里不是训练得到的结果。
#
# `get_action_probs()` 只查看概率，不参与训练，所以允许使用 `no_grad()`。采样并保存训练用 `log_prob` 的部分不能这样关闭梯度。

# %%
def state_to_tensor(state, n_states):
    state_tensor = torch.zeros((1, n_states), dtype=torch.float32)
    state_tensor[0, state] = 1.0
    return state_tensor


# 输入两个状态特征，输出两个动作得分；暂时不需要 Critic。
policy = nn.Linear(env.n_states, env.n_actions, bias=False)
nn.init.zeros_(policy.weight)
optimizer = torch.optim.SGD(policy.parameters(), lr=learning_rate)


def get_action_probs(state):
    with torch.no_grad():
        logits = policy(state_to_tensor(state, env.n_states))
        return Categorical(logits=logits).probs.squeeze(0)


# 复制更新前的数值，供后面比较，不让它们继续跟着参数变化。
before_probs = torch.stack([get_action_probs(0), get_action_probs(1)])
before_weights = policy.weight.detach().clone()
print("更新前，各状态的动作概率：", before_probs.tolist())

# %% [markdown]
# ## 4. 采样并走到终点，先不更新参数
#
# 一局里先保存三个列表：
#
# | 名称 | 保存什么 | 用途 |
# |---|---|---|
# | `log_probs` | 每一步实际动作的对数概率张量 | 构造损失，保留通向策略参数的梯度 |
# | `rewards` | 每一步的即时奖励数字 | 结束后计算各步回报 |
# | `trajectory` | 状态、动作、奖励、下一状态、是否终止 | 打印，检查是否真的走到终点 |
#
# 固定采样种子方便复现。在本机环境下，这个种子会采样到长路线；代码仍调用 `sample()`，没有把动作写死。不同版本或种子可能选择短路线，后面的计算同样适用。
#
# `action_tensor.item()` 是给环境看的整数，动作采样本身不求导；用于训练的 `log_prob` 必须保留张量。不要保存 `log_prob.item()` 作为损失输入。

# %%
torch.manual_seed(1)  # 在初始化之后固定本次动作采样的种子。
state = env.reset()
log_probs = []
rewards = []
trajectory = []

while True:
    state_tensor = state_to_tensor(state, env.n_states)
    logits = policy(state_tensor)  # [1, 2]：一个状态，两个动作得分。
    distribution = Categorical(logits=logits)
    action_tensor = distribution.sample()  # [1]：本次抽中的动作。
    action = int(action_tensor.item())

    # 去掉大小为 1 的批量维度，得到标量；这不会切断计算图。
    log_prob = distribution.log_prob(action_tensor).squeeze(0)
    next_state, reward, terminated, _ = env.step(action)

    log_probs.append(log_prob)  # 保留张量，不转换成普通数字。
    rewards.append(reward)
    trajectory.append((state, action, reward, next_state, terminated))

    if terminated:
        break
    state = next_state  # 本局没结束：到下一状态继续选动作。

for t, row in enumerate(trajectory):
    print(f"第 {t} 步：(状态, 动作, 奖励, 下一状态, 终止) = {row}")
print("rewards：", rewards)
print("log_prob 是否保留梯度：", [x.requires_grad for x in log_probs])

# %% [markdown]
# ## 5. 从终点倒着算，再恢复时间顺序
#
# $G_t$ 是从第 $t$ 步开始的未来回报，当前步的奖励不打折：
#
# $$
# G_t=r_t+\gamma G_{t+1}
# $$
#
# 本环境每次都真正终止，所以终点之后的回报为 0。长路线的奖励 `[0, 2]` 对应回报 `[1.8, 2.0]`；短路线的奖励 `[1]` 对应回报 `[1.0]`。
#
# `returns.reverse()` 让 `G0` 对应 `log_prob_0`，不是把回报的数值重新计算一遍。这里不做回报标准化，以免单局短轨迹的标准差和均值处理遮住原始信号。

# %%
G = 0.0
returns = []
for reward in reversed(rewards):
    G = reward + gamma * G
    returns.append(G)
returns.reverse()

# 回报来自环境数字，是本次更新中的固定权重，不需要求导。
returns_tensor = torch.tensor(returns, dtype=torch.float32)
print("按时间排列的 G_t：", returns)

# %% [markdown]
# ## 6. 补齐完整回合的时间折扣权重
#
# 前面用 `-log_prob * G_t` 解释了每一步的学习方向。若明确优化**从起点开始折扣的整局回报**，完整的 episodic REINFORCE 还给第 $t$ 步的项乘上 $\gamma^t$：
#
# $$
# J(\theta)=\mathbb{E}_{\pi_\theta}\left[\sum_t\gamma^t r_t\right],\qquad
# L=-\sum_t\gamma^t G_t\log\pi_\theta(a_t\mid s_t)
# $$
#
# 两种折扣参照点不同：`G_t` 内部的折扣以当前步为起点；外面的 `gamma ** t` 把该步信号换回整局起点的时间尺度。因为 $\gamma^tG_t=\sum_{k\ge t}\gamma^k r_k$，不是随意把同一奖励再打一次折。
#
# 长路线中：`returns=[1.8, 2.0]`，`time_discounts=[1.0, 0.9]`，最终损失使用的固定权重是 `[1.8, 1.8]`。`G1` 本身仍然是 `2.0`，没有改成 `1.8`。
#
# 如果 $\gamma=1$，外面的权重全是 1，就恢复前面讲过的求和式。一些实现省略外面的时间权重；本练习明确保留它，以对应上述从起点折扣的目标。

# %%
# t 是本局时间步编号，不是动作编号。
time_steps = torch.arange(len(returns), dtype=torch.float32)
time_discounts = gamma ** time_steps
training_weights = time_discounts * returns_tensor

print("各步时间折扣 gamma^t：", time_discounts.tolist())
print("用于损失的固定权重：", training_weights.tolist())

# %% [markdown]
# ## 7. 张量保留计算图，各步损失相加
#
# `torch.stack(log_probs)` 把标量张量拼成长度为回合步数的向量，同时保留每项到策略参数的计算关系。不要用 `torch.tensor([x.item() for x in log_probs])` 代替：那样会把训练输入变成断开计算图的常量。
#
# 本机固定种子的长路线中，两个动作被采样时的对数概率都是 $\log0.5$，总损失约为 $-1.8\log0.5-1.8\log0.5=2.4953$。
#
# 一个 `policy_loss` 包含本局全部时间步的信号；只需对它反向传播一次，不必逐项调用 `backward()`。

# %%
log_probs_tensor = torch.stack(log_probs)
loss_terms = -log_probs_tensor * training_weights
policy_loss = loss_terms.sum()

# detach/item 只用于这里的打印，训练仍使用原始 policy_loss。
print("log_probs：", log_probs_tensor.detach().tolist())
print("各步损失：", loss_terms.detach().tolist())
print("总损失：", round(policy_loss.item(), 4))
print("总损失能否反向传播：", policy_loss.requires_grad)

# %% [markdown]
# ## 8. 现在才更新一次策略
#
# - `zero_grad()`：清除旧的参数梯度，不会删掉刚才保存的计算图。
# - `backward()`：计算本局总损失对策略权重的梯度，此时参数数值还未改变。
# - `step()`：SGD 根据梯度修改参数，随后重新计算得到新概率。
#
# 一次完整采样期间没有修改策略参数；更新后下一局必须重新前向计算和采样，不能重复拿旧计算图训练。

# %%
optimizer.zero_grad()
policy_loss.backward()

# 把本次实际梯度复制下来，供打印和验证。
gradient = policy.weight.grad.detach().clone()
print("权重梯度（行=动作，列=状态）：", gradient.tolist())
optimizer.step()

after_probs = torch.stack([get_action_probs(0), get_action_probs(1)])
after_weights = policy.weight.detach().clone()
print("权重变化：", (after_weights - before_weights).tolist())
print("更新后，各状态的动作概率：", after_probs.tolist())

# %% [markdown]
# ## 9. 怎样理解这一次更新
#
# 本机种子为 1 时，轨迹为 `[(0,0,0.0,1,False), (1,1,2.0,None,True)]`。
#
# 起点动作 0 概率从 `0.5` 变为约 `0.5449`。这表示这次长路线样本推动了相应动作概率，不表示策略已经学会稳定走长路线。
#
# 中间状态两个动作的环境效果完全相同，奖励都是 2。本次采样到动作 1，因此这个单样本也提高了它的概率；不能据此得出中间状态动作 1 比动作 0 更优。多次采样的平均梯度才是比较长期期望的依据。
#
# 短路线同样有正回报，所以无 baseline 的单次短路线样本也会提高起点动作 1 的概率，即使它的期望回报低于长路线。这是 Day4 开始时讨论过的区别：**样本更新方向不等于已经比较所有动作后的最优结论**。
#
# 策略损失数值是用于求梯度的量，不是训练完成度或策略质量分数。更好的策略需要多局采样、更新并评估；这里先检查单局链路。

# %% [markdown]
# ## 10. 自动验证本局，不额外设置计算题
#
# 下面的断言检查真实轨迹、回报、计算图、梯度与参数变化。长短两条合法路线都能通过，不把特定随机动作当作唯一正确结果。全部通过说明这次更新链路成立，不等同于算法收敛。
#
# 后续练习再把“一局”放进多局循环；本次不自动标记学习完成。

# %%
checks = 0

assert terminated and env.current_state is None
checks += 1
assert len(trajectory) == len(log_probs) == len(rewards) == len(returns)
checks += 1
assert all(row[4] is False for row in trajectory[:-1])
checks += 1

if trajectory[0][1] == 0:
    assert len(trajectory) == 2 and trajectory[1][0] == 1
    expected_returns = torch.tensor([1.8, 2.0])
else:
    assert len(trajectory) == 1 and rewards == [1.0]
    expected_returns = torch.tensor([1.0])
checks += 1

assert torch.allclose(returns_tensor, expected_returns)
checks += 1
assert all(x.requires_grad and x.grad_fn is not None for x in log_probs)
checks += 1
assert log_probs_tensor.requires_grad and policy_loss.requires_grad
checks += 1
assert not returns_tensor.requires_grad and not training_weights.requires_grad
checks += 1
assert torch.isfinite(policy_loss).item() and torch.isfinite(gradient).all().item()
checks += 1
assert gradient.abs().sum().item() > 0
checks += 1
assert not torch.equal(before_weights, after_weights)
checks += 1
assert torch.allclose(after_weights, before_weights - learning_rate * gradient)
checks += 1
assert torch.allclose(after_probs.sum(dim=1), torch.ones(env.n_states))
checks += 1

# 这个线性 one-hot 策略中，各状态使用不同列的参数，可以单独核对方向。
assert all(after_probs[s, a] > before_probs[s, a] for s, a, _, _, _ in trajectory)
checks += 1
assert checks == 14
print(f"通过 {checks} 项检查：轨迹终止、回报配对、计算图、梯度、参数与概率。")
