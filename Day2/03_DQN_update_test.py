# Generated from the adjacent notebook by scripts/marl.py export.

# %% [markdown]
# # Day2 · 从 Q 表到 DQN 的一次更新
#
# 这个 Notebook 只做一件事：**把"一条经验怎样改动在线网络的一个参数"完整走一遍**，并且让每一步的数值都能手算核对。
#
# 它**不是**完整的 DQN：没有环境交互、没有回放池、没有训练循环、没有周期同步。它是一次更新的切片。
#
# ## 十个区块
#
# | 区块 | 内容 | 关键问题 |
# |---|---|---|
# | 1 | 创建 `online` 与 `target` 两个网络 | 两个网络分别负责什么？ |
# | 2 | 手动设置权重 | 为什么能用 `copy_` 手改参数？ |
# | 3 | 构造一条经验 $(s,a,r,s')$ | 一条经验里存了哪些量？ |
# | 4 | 计算在线网络**当前动作**的 Q 值 | `gather` 为什么按实际执行的动作取？ |
# | 5 | 计算目标网络**下一状态最大** Q 值 | `max` 为什么取最大而不是实际动作？ |
# | 6 | 计算 TD target | 终止时为什么没有未来项？ |
# | 7 | 计算 loss | 平方误差在比什么？ |
# | 8 | 反向传播并更新在线网络 | 谁被改了？谁没被改？ |
# | 9 | 检查更新前后参数变化 | 变化是否与手算一致？ |
# | 10 | 单独演示目标网络同步 | 同步之后两个网络谁变了？ |
#
# ## 形状检查清单
#
# 下面每一步都会打印张量形状，全部集中在一节里对照：
#
# ```text
# state.shape          q_values.shape       action.shape        current_q.shape
# next_q_values.shape  next_q.shape         target.shape        loss.shape
# ```

# %% [markdown]
# ## 为什么要手动设置这两组权重？
#
# 你看到的代码是：
#
# ```python
# with torch.no_grad():
#     online.weight.copy_(torch.tensor([[0.30, 0.0], [0.70, 0.0]]))
#     target.weight.copy_(torch.tensor([[0.0, 0.60], [0.0, 0.20]]))
# ```
#
# **先说结论：这些数字是为了教学而手工挑选的，不是 DQN 必须使用的权重，也不是智能体已经学出的正确答案。** 它们让这一次更新可以像算小学乘法一样逐项核对：当前估计是 `0.30`，下一状态的最大估计是 `0.60`，学习目标是 `0.50`。
#
# ## 图 1：一行权重怎样变成一个动作的 Q 值
#
# `torch.nn.Linear(2, 2, bias=False)` 接收两个状态特征 `x1, x2`，输出动作 0 和动作 1 的两个 Q 值。**权重矩阵的一行负责一个动作，一列负责一个输入特征**；这里关闭了偏置，所以没有额外的加数。
#
# ```text
#                         输入状态 [x1, x2]
#                               │
#                     ┌─────────┴─────────┐
#                     ▼                   ▼
#           动作 0 的一行 [w00, w01]   动作 1 的一行 [w10, w11]
#                     │                   │
#                     ▼                   ▼
#        Q(状态, 动作0)=x1*w00+x2*w01  Q(状态, 动作1)=x1*w10+x2*w11
# ```
#
# 这相当于 PyTorch 的 `输出 = 输入 @ weight.T`。两个方括号组成 `[[x1, x2]]` 时，最外层的括号表示“现在有 1 条状态”，不表示多出一个特征。
#
# ## 图 2：这两个矩阵分别让网络算出什么
#
# ```text
# 当前状态 s = [1, 0]                 下一状态 s_next = [0, 1]
#           │                                      │
#           ▼                                      ▼
# online 的权重                           target 的权重
# 动作0: [0.30, 0.00]                    动作0: [0.00, 0.60]
# 动作1: [0.70, 0.00]                    动作1: [0.00, 0.20]
#           │                                      │
#           ▼                                      ▼
# Q_online(s) = [0.30, 0.70]            Q_target(s_next) = [0.60, 0.20]
# ```
#
# 例如，在线网络的动作 0 输出是 $1\times0.30+0\times0=0.30$；目标网络的动作 0 输出是 $0\times0+1\times0.60=0.60$。输入只含一个 `1`，因此这些输出恰好等于对应位置的权重，方便我们心算。**`0` 的位置只是为了简化演示**；真实状态的两个特征不需要是 0 或 1，权重也不需要长这样。
#
# | 用哪个网络 | 输入哪个状态 | 两个动作的输出 | 这条经历用哪个值 |
# |---|---|---|---|
# | `online` | 当前状态 `s=[1,0]` | `[0.30, 0.70]` | 当时实际执行动作 0，因此取 `0.30` |
# | `target` | 下一状态 `s_next=[0,1]` | `[0.60, 0.20]` | 下一状态假设挑最优动作，因此取最大值 `0.60` |
#
# 虽然当前在线网络给动作 1 的估值 `0.70` 更高，但回放记录里**当时做的是动作 0**（可能来自探索或旧策略），所以本次学习要修正的是 `0.30`，不能把动作 0 的经历当成动作 1 的经历。两边分工如下：
#
# ```text
# 当前估计：Q_online(s, 实际动作0) = 0.30
#                                    │ 与它比较
#                                    ▼
# 学习目标：奖励 + 折扣 * max Q_target(s_next, ·)
#         = -0.04 + 0.9 * 0.60 = 0.50
# ```
#
# 随后代码计算误差 $(0.30-0.50)^2=0.04$，通过 `backward()` 和 `optimizer.step()` 调整在线网络。这个特意选定的线性层、输入和学习率使动作 0 的估值更新后恰好为 `0.50`；一般 DQN 的一次更新**不保证**恰好等于目标值。
#
# ## `copy_` 与 `no_grad()` 是在做什么？
#
# - `weight.copy_(...)` 是把右边的矩阵**原地复制进已有网络的权重**，不是训练，也没有额外创建一个新网络。
# - `with torch.no_grad():` 表示**只在这个代码块内**不记录求导过程。手工改参数属于初始化操作，不需要把“复制权重”当作一次待学习的计算；对需要梯度的叶子参数直接原地赋值也会被自动求导机制拦住。
# - 离开代码块后，`online` 仍可以正常求梯度并被优化器更新。这里 `target` 在本次训练中不变，是因为优化器只接收了 `online.parameters()`，且计算目标值也放在 `no_grad()` 内；不是因为前面这一次 `no_grad()` 永久冻结了它。
#
# **与真实 DQN 的区别：** 真实训练一般先把目标网络初始化为在线网络的一份拷贝，训练在线网络后每隔一段时间同步目标网络。本例一开始就给它们设置不同的矩阵，只是为了清楚地展示“在线网络负责当前估计、目标网络负责较稳定的目标”这两种角色；这段代码没有实现定期同步。第 10 个区块会补上独立的同步演示。

# %% [markdown]
# ## 区块 1 · 创建 online 网络和 target 网络
#
# 两个网络结构完全相同（2 个输入特征 → 2 个动作的 Q 值，无偏置），但**职责不同**：
#
# - `online`：给出"当前估计"，并且是唯一会被优化器修改的网络；
# - `target`：只负责算出 TD target 里的下一步价值，本身在这一步里不更新。

# %%
import torch

# 固定随机种子：两个网络的初始权重本来是随机的，固定后每次运行看到的初始值一致，
# 方便对照"手动设置权重"到底改了什么。
torch.manual_seed(7)

# torch 是 PyTorch 库：这里用它存数字（张量）、搭网络并自动计算梯度。
# 这是一次更新的微型演示，不包含环境交互、回放池或完整的 DQN 训练循环。
# online（在线网络）负责学习；target（目标网络）只帮我们计算学习目标。
# Linear(2, 2)：输入状态有 2 个数，输出 2 个动作各自的 Q 值；bias=False 表示不加偏置。
# 这两个输入数只是示意用的特征，不是 CartPole 真正的 4 维状态。
online = torch.nn.Linear(2, 2, bias=False)
target = torch.nn.Linear(2, 2, bias=False)

# 两个网络一开始的权重都是随机初始化的，先看一眼它们的差别。
print("online 初始权重：", online.weight.detach().tolist())
print("target 初始权重：", target.weight.detach().tolist())

# %% [markdown]
# ## 区块 2 · 手动设置权重
#
# 下面这一步只是**给参数赋值**，不是训练：
#
# - `weight.copy_(...)` 原地覆盖权重矩阵，不创建新对象；
# - 放在 `with torch.no_grad():` 里，是因为对需要梯度的叶子张量做原地赋值会被自动求导机制拦住，而且"设置初始值"本来也不该被记录成一次可求导的运算；
# - 离开这个代码块后，`online` 依然可以正常求梯度。
#
# 同时保存一份 `online` 更新前的权重快照，供区块 9 对比。

# %%
# 手动设置网络参数，让每一步的数值都能手算核对。
# weight 的两行分别对应动作 0、动作 1，两列分别对应状态的第 1、2 个数。
# no_grad()：设置初始权重时不要记录梯度；离开这里后，online 仍然能正常学习。
# 真实 DQN 通常先让两个网络参数相同，再定期同步；这里只为演示而设置不同数值。
with torch.no_grad():
    online.weight.copy_(torch.tensor([[0.30, 0.0], [0.70, 0.0]]))
    target.weight.copy_(torch.tensor([[0.0, 0.60], [0.0, 0.20]]))

# 记录更新前的权重，后面用来对比"哪些参数被改动了"。
online_weight_before = online.weight.detach().clone()
target_weight_before = target.weight.detach().clone()

print("online 权重：", online.weight.detach().tolist())
print("target 权重：", target.weight.detach().tolist())

# %% [markdown]
# ## 区块 3 · 构造一条经验
#
# 一条回放记录的核心内容是 $(s, a, r, s')$，即**动作发生前**的状态、**当时真正执行**的动作、这一步拿到的奖励、以及**动作发生后**到达的状态。
#
# 三个容易混的点：
#
# 1. `s` 和 `s_next` 是一前一后两个状态，属于同一条转移；
# 2. `action` 存的是动作**编号**（`0` 或 `1`），不是 Q 值；
# 3. `reward` 是环境当时给的数字，写进回放池之后不会再变——后面无论训练多少次，这条记录里的 $r$ 都是同一个值。

# %%
# 假装从回放池取出一条经历：(当前状态 s, 执行的动作, 奖励, 下一状态 s_next)。
# 最外层的 [...] 表示批量维度：本例批量大小是 1，即一次只学习这一条经历。
# s 是动作发生前的状态；把 [1, 0] 输入 online，会输出 [0.30, 0.70]。
s = torch.tensor([[1.0, 0.0]])
# action 是当时真正执行的动作编号 0，不是 Q 值；形状 [[0]] 方便稍后按动作编号取值。
action = torch.tensor([[0]])
# reward 是执行动作 0 后，环境当时给出的单步奖励；记录下来后它不会随训练改变。
reward = torch.tensor([-0.04])
# s_next 是这一步到达的下一状态；输入 target 会输出动作 0/1 的值 [0.60, 0.20]。
s_next = torch.tensor([[0.0, 1.0]])
# gamma 是未来价值的折扣系数：0.9 表示下一状态的估计价值乘以 0.9。
gamma = 0.9

print("s      =", s.tolist())
print("action =", action.tolist(), "（动作编号，不是 Q 值）")
print("reward =", [round(x, 4) for x in reward.tolist()], "（环境当时给的数字，之后不会变）")
print("s_next =", s_next.tolist())
print("gamma  =", gamma)

# %% [markdown]
# ## 区块 4 · 计算在线网络**当前动作**的 Q 值
#
# `online(s)` 一次性给出**两个动作**的估值 `[[0.30, 0.70]]`。但这条经历当时执行的是动作 `0`，所以要和目标值比较的是 $Q_{\mathrm{online}}(s,0)=0.30$。
#
# `gather(1, action)` 就是"按每条经历各自的动作编号，取对应列"：
#
# ```text
# q_values = [[0.30, 0.70]]
# action   = [[0]]              → 取第 0 列
# gather   = [[0.30]]
# ```
#
# 为什么不能取 `0.70`？因为 `0.70` 是**没有执行**的动作 1 的估值。用一条动作 0 的经验去修动作 1 的估值，等于把"这次走的是左边"记成"这次走的是右边"。

# %%
# online(s) 输出形状 [1, 2]：这一条经历的两个动作 Q 值是 [[0.30, 0.70]]。
q_values = online(s)

# gather(1, action)：沿第 1 维（动作列）取实际执行的动作 0，得到 [[0.30]]。
# squeeze(1)：去掉多余的动作列维度，变为 [0.30]，便于和目标值逐条比较。
# q_actual 是“当前估计”：不能拿动作 1 的 0.70 来训练这条动作 0 的经历。
current_q = q_values.gather(1, action).squeeze(1)

print("online(s) 全部动作的估值：", q_values.detach().tolist())
print("实际执行的动作编号：", action.tolist())
print("gather 取出的当前估计 current_q：", current_q.detach().tolist())

# %% [markdown]
# ## 区块 5 · 计算目标网络**下一状态最大**的 Q 值
#
# `target(s_next)` 给出下一状态在**两个动作**下的估值 `[[0.60, 0.20]]`。TD target 需要的是"假设下一步挑最好的动作"，所以取最大值 $0.60$。
#
# 一问一答式的区分：
#
# - **为什么这里是 `max` 而不是实际动作？** 因为这一步还没有发生。下一状态会选哪个动作由未来的策略决定，Q-learning 的约定是"假设未来取最优动作"，所以用 $\max_{a'}Q_{\mathrm{target}}(s',a')$。
# - **为什么用 `target` 而不是 `online`？** 如果用在线网络自己算目标，目标和被修正的量会一起动，训练目标就变成了"追自己"，容易震荡。用一份暂时固定的参数来算目标，相当于给这次更新一个稳定的落点。
#
# `max(dim=1).values` 返回的是**值**；`.indices` 才是取到最大值的动作编号。两者容易混：构造 target 用值，选动作才用编号。

# %%
# target(s_next) 为 [[0.60, 0.20]]，max(dim=1).values 在动作列中选 0.60。
next_q_values = target(s_next)
next_q, next_action_index = next_q_values.max(dim=1)

print("target(s_next) 两个动作的估值：", next_q_values.detach().tolist())
print("最大值 next_q：", next_q.detach().tolist())
print("取到最大值的动作编号：", next_action_index.tolist(),
      "（这是 argmax 的结果；构造 TD target 用的是值 next_q，不是这个编号）")

# %% [markdown]
# ## 区块 6 · 计算 TD target
#
# $$
# y = r + \gamma \max_{a'} Q_{\mathrm{target}}(s', a')
# $$
#
# 代入本例子：$-0.04 + 0.9\times0.60 = 0.50$。
#
# **终止转移要单独处理。** 如果这条经历是"到达终点/掉进悬崖"的最后一步，就不存在"下一状态的价值"——$s'$ 是终止状态，它的价值按定义是 0。两种写法等价：
#
# ```python
# y = reward                                    # 终止：直接等于当前奖励
# y = reward + gamma * next_q * not_done        # not_done = 0 时后一项归零
# ```
#
# 下面的代码把两种写法都算一遍。要注意：**`terminated` 是"任务真的结束了"，不是"被时间上限截断"。** 若是人为设的步数上限导致回合被截断，任务本身还可以继续，这时仍应保留未来项——把截断当成终止，会系统性低估接近上限的状态。

# %%
# td_target 是“希望 current_q 靠近的数”，不是回放池里预先存好的 Q 值。
# no_grad()：目标值只用于对照，不让这次损失沿着目标网络反向传播。
# 这条经历没有终止，因此目标值 = reward + gamma * 未来最大 Q 值 = -0.04 + 0.9*0.60 = 0.50。
with torch.no_grad():
    td_target = reward + gamma * next_q

# 若下一状态是真正的终止状态，就没有未来奖励，目标值应直接等于 reward。
terminated = False                      # 这条经验不是终止转移
not_done = 0.0 if terminated else 1.0
td_target_with_mask = reward + gamma * next_q * not_done

# 作为对照：如果这条经历是终止转移（例如到达终点），目标值只剩当前奖励。
td_target_if_terminated = reward + gamma * next_q * 0.0

print("非终止转移的 TD target：", round(td_target.item(), 4))
print("用 not_done 掩码算出的结果：", round(td_target_with_mask.item(), 4))
print("假设为终止转移时的 TD target：", round(td_target_if_terminated.item(), 4))

# %% [markdown]
# ## 区块 7 · 计算 loss
#
# 损失衡量的是"当前估计离目标有多远"，这里用平方误差：
#
# $$
# L = \big(Q_{\mathrm{online}}(s,a) - y\big)^2
#      = (0.30 - 0.50)^2 = 0.04
# $$
#
# `mean()` 是对一个批次求平均。批量大小为 1 时，结果仍是一条经历的误差；批量大于 1 时，它是**训练用的平均损失**，与测试时对多局回报取平均是两件完全不同的事：
#
# - 训练损失的平均：把多条经验的误差合成一个可反向传播的标量；
# - 测试回报的平均：对同一套策略的多次完整回合表现做统计。
#
# 平方误差的正负号没有意义（负误差平方后也是正数），它只关心距离。这也是它和 TD 误差的区别：TD 误差 $y - Q$ 带符号，用来说明"原来估计偏低还是偏高"。

# %%
# loss 是当前估计和目标值的差距：本例 (0.30 - 0.50)^2 = 0.04。
# mean() 对一个批次的误差求平均；这里批量大小只有 1，结果仍是 0.04。
loss = ((current_q - td_target) ** 2).mean()

td_error_signed = (td_target - current_q).item()
print("带符号的 TD 误差 (target - current_q)：", round(td_error_signed, 4),
      "→ 正数表示原来低估了这个动作")
print("损失 loss：", round(loss.item(), 4))

# %% [markdown]
# ## 区块 8 · 反向传播并更新在线网络
#
# 三步顺序固定：
#
# 1. `optimizer.zero_grad()`：清空上一次留下的梯度（PyTorch 默认把梯度累加，不清就会把两次的梯度叠在一起）；
# 2. `loss.backward()`：沿着计算图求出每个需要梯度的参数的梯度，此时参数值还没变；
# 3. `optimizer.step()`：按梯度更新参数。
#
# **谁会被改？** 只有交给优化器的参数。下面 `optimizer` 只接收了 `online.parameters()`，所以即使 `target` 参与了前向计算，它的参数也不会被这一步改动。同时 `target` 的前向是在 `no_grad()` 里做的，它连梯度都没有累积——这不是"冻结"，而是**从来没有被放进需要求梯度的路径里**。

# %%
# optimizer 是更新参数的工具：只把 online 的参数交给它，target 不会被这次更新改动。
# SGD 表示随机梯度下降；lr=0.5 是本次梯度更新的步长，不是上面的折扣系数 gamma。
optimizer = torch.optim.SGD(online.parameters(), lr=0.5)

print("优化器管理的参数：")
for name, param in online.named_parameters():
    print(f"  online.{name}  requires_grad={param.requires_grad}")
print("target 是否在优化器里：", any(p is target.weight for group in optimizer.param_groups
                                      for p in group["params"]))

# 第 1 步：清空上一次留下的梯度（PyTorch 默认会把多次梯度累加）。
optimizer.zero_grad()
# 第 2 步：根据 loss 反向计算 online 参数的梯度；此时参数还没有改变。
loss.backward()
print()
print("backward 之后、step 之前的 online 梯度：")
print(" ", online.weight.grad.tolist())
print("backward 之后、step 之前的 online 权重：")
print(" ", online.weight.detach().tolist(), "（与更新前完全相同）")

# 第 3 步：用算出的梯度更新 online 参数；这个简单例子会把动作 0 的 Q 值推到 0.50。
# target 没交给 optimizer，也没有手动同步，所以其参数保持不变。
optimizer.step()

print()
print("step 之后的 online 权重：")
print(" ", online.weight.detach().tolist())
print("target 的梯度：", target.weight.grad, "（从未参与需要梯度的计算，因此没有梯度）")

# %% [markdown]
# ## 区块 9 · 检查更新前后的参数变化
#
# 把更新前保存的权重快照和现在的权重逐项相减，就能看出**到底哪些数字动了**。
#
# 预期结果：只有动作 0 那一行、第一个特征那一列（`[0][0]`）从 `0.30` 变成 `0.50`，其余保持不变。原因是本例的状态输入是 `[1, 0]`：第二列权重乘的是 `0`，它对输出没有贡献，因此梯度也是 `0`，不会被更新。这不是 DQN 的一般性质，而是这个刻意设计的输入造成的。
#
# 一般网络里，一次更新会同时改动很多参数，动作 1 的估值也可能跟着变化。

# %%
# 重新用 online(s) 计算更新后的值；旧的 current_q 不会自动变成新值。
q_after = online(s)

print("更新前 online(s)：", q_values.detach().tolist())
print("更新后 online(s)：", q_after.detach().tolist())
print()
print("权重变化（更新后 - 更新前）：")
delta = (online.weight.detach() - online_weight_before).tolist()
for row_index, row in enumerate(delta):
    print(f"  动作 {row_index} 的一行：{row}")

print()
print("动作 0 的估值变化：", round(q_values[0, 0].item(), 4), "→", round(q_after[0, 0].item(), 4))
print("动作 1 的估值变化：", round(q_values[0, 1].item(), 4), "→", round(q_after[0, 1].item(), 4))

# 一般网络不保证"更新后恰好等于 target"，本例恰好等于 0.50，
# 是因为线性层、输入 [1,0]、学习率 0.5 和一次 SGD 步共同作用的结果。
print()
print("更新后的动作 0 估值是否等于 TD target：",
      round(q_after[0, 0].item(), 6) == round(td_target.item(), 6))

# %% [markdown]
# ### 形状检查
#
# 把这条经验涉及的 8 个张量抽出来对齐看一遍。记住两条约定：
#
# - **最外层维度是批量**：`[1, 2]` 表示"1 条状态、2 个动作"，不是"2 个状态"；
# - **`squeeze` 只去掉长度为 1 的维度**，不改数值，只是让标量和标量之间能直接相减。

# %%
# ============================================================
# 形状检查：8 个张量分别在哪个位置、是什么形状
# ============================================================

shape_rows = [
    ("state", s, "当前状态：1 条经验、2 个特征"),
    ("q_values", q_values, "在线网络对两个动作的估值：1 条经验、2 个动作"),
    ("action", action, "实际执行的动作编号：1 条经验、1 个动作"),
    ("current_q", current_q, "gather 之后的当前估计：1 条经验"),
    ("next_q_values", next_q_values, "目标网络对下一状态两个动作的估值"),
    ("next_q", next_q, "下一状态的最大估值：1 条经验"),
    ("target", td_target, "TD target：1 条经验"),
    ("loss", loss, "标量损失：所有经验求平均后只剩 1 个数"),
]

for name, tensor, note in shape_rows:
    print(f"{name:14s} shape={tuple(tensor.shape)}   {note}")

# 形状对不上的地方要能被发现，而不是靠广播悄悄算出一个数。
assert q_values.shape == (1, 2), "q_values 应为 [1, 2]"
assert current_q.shape == (1,), "current_q 应为 [1]"
assert next_q.shape == (1,) and td_target.shape == (1,)
assert loss.shape == torch.Size([]), "loss 应为 0 维标量"
print()
print("形状断言全部通过。")

# %% [markdown]
# ## 区块 10 · 目标网络同步
#
# 真实 DQN 的目标网络不是一开始就长成另一个样子，而是**先复制在线网络的参数，之后每隔固定步数再复制一次**。同步动作本身不是训练：
#
# ```python
# target.load_state_dict(online.state_dict())   # 复制参数，不是梯度下降
# ```
#
# - 同步改的是 `target`，`online` 不变；
# - 同步与梯度无关，没有 `backward()`、没有学习率；
# - 两次同步之间，`target` 的参数保持不动，所以用同一条经历算出的 TD target 也是固定的。
#
# 下面演示同步前后：先看两个网络的权重，再同步，再看一次；最后重新计算同一条经历的 TD target，看看目标有没有变。

# %%
# ============================================================
# 10. 目标网络同步演示
# ============================================================

print("同步前的权重：")
print("  online：", online.weight.detach().tolist())
print("  target：", target.weight.detach().tolist())

online_weight_at_sync = online.weight.detach().clone()

# 复制当前在线网络参数到目标网络：这一步是参数复制，不是训练。
target.load_state_dict(online.state_dict())

print()
print("同步后的权重：")
print("  online：", online.weight.detach().tolist())
print("  target：", target.weight.detach().tolist())

same_online = torch.equal(online.weight.detach(), online_weight_at_sync)
same_target_now = torch.equal(target.weight.detach(), online.weight.detach())
print()
print("online 在同步过程中是否被改动：", not same_online, "（应为 False）")
print("同步后 target 是否与 online 完全一致：", same_target_now)
print("target 相对同步前是否变化：",
      not torch.equal(target.weight.detach(), target_weight_before))

# 用同步后的目标网络重新算同一条经历的 TD target。
with torch.no_grad():
    next_q_after_sync = target(s_next).max(dim=1).values
    td_target_after_sync = reward + gamma * next_q_after_sync

print()
print("同步前 target(s_next)：", next_q_values.detach().tolist(),
      "→ 最大", round(next_q.item(), 4))
print("同步后 target(s_next)：", target(s_next).detach().tolist(),
      "→ 最大", round(next_q_after_sync.item(), 4))
print("同一条经历的 TD target：", round(td_target.item(), 4),
      "→", round(td_target_after_sync.item(), 4))

# %% [markdown]
# ### 同步后的 target 为什么变成 $-0.04$？
#
# 上面打印出的同步后 target 输出是 `[0.0, 0.0]`——这不是算错，而是这份演示的在线网络**第二列权重恰好全是 0**：
#
# ```text
# online 权重（同步前，更新后）        s_next = [0, 1]
# 动作0: [0.50, 0.0]          →  0*0.50 + 1*0.0 = 0.0
# 动作1: [0.70, 0.0]          →  0*0.70 + 1*0.0 = 0.0
# ```
#
# 所以同步之后，目标网络在 `s_next=[0,1]` 上的输出就是 `[0.0, 0.0]`，最大值为 $0$，TD target 退化为 $r + \gamma\times0 = -0.04$。
#
# 这与前面"抽象数值题里同步后 target 从 $0.50$ 变成 $0.68$"并不矛盾：那道题的假设是目标网络在该状态下输出 $0.8$（即第二列权重不全为 0）。**抽象例子和这份代码的矩阵不同，必须按当前矩阵重新计算，不能直接搬用 $0.68$。**
#
# 这也正是教学权重需要说明的地方：为了让 `online(s)=[0.30, 0.70]` 与 `target(s_next)=[0.60, 0.20]` 都能一眼心算，两组权重都只用了第一列，代价就是同步之后第二列的信息量为零。
#
# ### 为什么只更新 online，target 不会"自动"跟上？
#
# | 问题 | 答案 |
# |---|---|
# | 优化器里放了谁？ | 只有 `online.parameters()` |
# | `target` 参与前向计算了吗？ | 参与了，但被 `no_grad()` 包住，不累积梯度 |
# | `loss.backward()` 会改 `target` 吗？ | 不会，它只算梯度，不改参数 |
# | `optimizer.step()` 会改 `target` 吗？ | 不会，`target` 不在参数组里 |
# | 那 `target` 什么时候变？ | 只有显式调用 `load_state_dict` / `copy_` 时 |
#
# 所以"目标网络保持稳定"不是自动发生的，而是靠**不把它交给优化器 + 定期手动同步**这两件事共同保证的。

# %% [markdown]
# ## 常见误区
#
# 1. **把 `gather` 取的值当成"最优动作的 Q 值"。** `gather` 取的是回放记录里**实际执行**的动作；当前哪个动作估值最高与这条经验无关。
# 2. **把 `max` 用到当前状态上。** 当前状态取实际动作，下一状态取最大值——两边口径不同，这正是 DQN 更新里最容易写错的一行。
# 3. **以为 `target` 的梯度会被算出来。** 目标值放在 `no_grad()` 内，`target.weight.grad` 始终是 `None`；它不参与反向传播。
# 4. **以为 `no_grad()` 会永久冻结网络。** 它只作用于所在代码块；离开之后 `online` 照常可训练。
# 5. **以为 `copy_` 是训练。** 它是原地赋值，没有任何梯度、学习率参与。
# 6. **把 `terminated` 与 `truncated` 混为一谈。** 真正终止时未来项为 0；被步数上限截断时任务本可继续，仍要保留未来项。
# 7. **把训练损失的平均当成评估指标。** `loss.mean()` 是对一个批次的误差求平均；测试回报的平均是对多局完整回合的成绩求平均。
# 8. **以为一次更新后 Q 值一定等于 TD target。** 本例恰好相等是刻意构造；一般网络只会朝目标移动一步。
#
# ## 本节边界与自检
#
# 本 Notebook 覆盖的是一次更新的全部中间量：当前估计、下一状态估值、TD target、损失、参数变化、目标网络同步。**未覆盖**：环境交互、回放池的存取与采样、批量训练、$\varepsilon$-greedy 行为策略、`target_sync` 的步数计数、CartPole 一类的连续状态任务。这些属于完整 DQN 训练循环，不在本节范围内。
#
# 自检题：
#
# 1. 若把 `action` 从 `[[0]]` 改成 `[[1]]`，`gather` 会取出哪个值？TD target 会不会跟着变？
# 2. 训练期间 `reward` 会不会因为网络更新而改变？
# 3. `target` 网络还没同步时，在线网络的输出从 `0.30` 变成 `0.80`，TD target 会不会立刻跟着变？
# 4. 两条经历的当前估值是 `[[0.30, 0.70], [0.40, 0.20]]`，实际动作是 `[[0], [1]]`，批量 `gather` 得到什么？若第二条的实际动作改成 `0`，第二个取出的值是多少？
# 5. 环境步数走到 1000、`target_sync=500` 时，这一步要不要同步目标网络？计数单位是环境交互步还是测试局数？
