# Generated from the adjacent notebook by scripts/marl.py export.

# %% [markdown]
# # Actor-Critic：从动作概率到价值更新
#
# 这个 Notebook 用一个非常小的环境，完整演示一次 Actor-Critic 训练循环。
#
# ```text
# Actor：
#     看到状态后，输出每个动作的选择概率。
#
# Critic：
#     看到状态后，估计从这个状态继续走下去能得到多少回报。
#
# TD 误差：
#     比较“新的目标”与 Critic 原来的估计，告诉 Actor 这次动作比预期好还是差。
# ```
#
# 本实验的环境只有两个非终止状态：
#
# ```text
# 起点 s=0 ──动作0 / 奖励0──> 中间状态 s=1 ──任意动作 / 奖励2──> 结束
#          └─动作1 / 奖励1──────────────────────────────────────> 结束
# ```
#
# 设折扣因子 $\gamma=0.9$，那么从起点选动作 0 的**折扣回报**是 $0+0.9\times2=1.8$；选动作 1 的回报是 $1$。所以从起点看，动作 0 更好。
#
# 代码会训练一个共享主体、带 Actor 头和 Critic 头的神经网络。每一步计算 $y=r+\gamma(1-d)V(s')$，再计算 TD 误差 $\delta=y-V(s)$：Actor 用 $\delta$ 调整本次所选动作的概率，Critic 让 $V(s)$ 靠近 $y$。其中 $d=1$ 表示真正终止，终止后不再考虑未来价值。
#
# **读输出时留意口径**：训练中的 `episode_return` 与测试中的 `average_test_return` 都把环境奖励直接相加，走动作 0 会得到 $0+2=2$；起点的 $V(s)$ 则估计折扣回报，理想值约为 $1.8$。这两个数字不同并不是训练出错。环境和贪心动作都是确定性的，所以 100 局测试会重复同一条路径；此处保留平均值只是展示通用评估写法。

# %% [markdown]
# ## 这个实验是什么，不是什么
#
# | 是 | 不是 |
# |---|---|
# | 单步 Actor–Critic：每走一步就更新一次参数 | 完整 PPO：没有采样一批轨迹后再多轮裁剪更新 |
# | 一个共享主体 + Actor 头 + Critic 头的最小实现 | 通用算法框架：没有并行环境、没有 GAE、没有裁剪比率 |
# | 用 TD 误差当优势信号的教学写法 | 完整的优势估计：没有多步回报、没有 λ 加权 |
# | 只有 2 个非终止状态、2 个动作 | 任何真实任务 |
#
# 判断边界的一个简单办法：**每执行一步环境交互，参数就被更新一次**。PPO 不会这样做，它先收集一段轨迹，再用同一批数据反复更新若干轮，并用概率比值把更新幅度限制住。这两件事在本 Notebook 里都没有出现。

# %%
import torch
from torch import nn
from torch.distributions import Categorical


# ============================================================
# 1. 固定随机种子
# ============================================================

# 固定 PyTorch 随机种子后，网络初始化和动作采样更容易复现。
# 这不是强化学习算法必须的步骤，而是为了方便学习和排查问题。
torch.manual_seed(7)

# %% [markdown]
# ## 2. 定义教学环境
#
# 这一部分只负责描述“智能体所处的世界”：当前状态、执行动作后到达的下一个状态、即时奖励，以及回合是否结束。
#
# 先把环境单独读懂，再看神经网络如何学习这个环境。

# %%
# ============================================================
# 2. 定义一个极简环境
# ============================================================

class TinyChainEnv:
    """
    一个只有两个非终止状态的教学环境。

    状态 0：起点
        动作 0：走到状态 1，得到奖励 0，回合继续。
        动作 1：直接结束，得到奖励 1。

    状态 1：中间状态
        无论选择哪个动作，都结束并得到奖励 2。

    这样从状态 0 出发：
        选择动作 0 的折扣回报约为 0 + 0.9 * 2 = 1.8；
        选择动作 1 的回报为 1；
        因此训练后应该更偏向动作 0。
    """

    def __init__(self):
        # 一共有两个可以被网络看到的状态：0 是起点，1 是中间状态。
        # 终止状态不需要再输入网络，所以不把它计入这里。
        self.n_states = 2

        # 两个动作：动作 0 和动作 1。
        self.n_actions = 2

        # current_state 保存环境当前所在的状态编号。
        self.current_state = 0

    def reset(self):
        """把环境恢复到起点，并返回起点状态编号。"""

        self.current_state = 0
        return self.current_state

    def step(self, action):
        """
        在当前状态执行一个动作。

        返回四个东西：
            next_state: 执行动作后到达的状态编号
            reward:     这一步立刻获得的奖励
            terminated: 任务是否真正结束
            info:       额外信息；本小实验不使用，所以返回空字典
        """

        # 为了避免把非法动作悄悄当成合法动作，先做一个简单检查。
        if action not in (0, 1):
            raise ValueError("action 必须是 0 或 1")

        if self.current_state == 0:
            if action == 0:
                # 走较长路线：当前只拿 0 分，但还没有结束。
                next_state = 1
                reward = 0.0
                terminated = False
            else:
                # 走较短路线：立即结束，得到 1 分。
                next_state = None
                reward = 1.0
                terminated = True
        elif self.current_state == 1:
            # 从中间状态走到终点。
            # 两个动作在这里效果相同，只是为了保持动作空间大小一致。
            next_state = None
            reward = 2.0
            terminated = True
        else:
            # 正常代码不会在终止后继续调用 step；这里的报错能帮助定位错误。
            raise RuntimeError("当前回合已经结束，请先调用 reset()")

        # 结束时 next_state 是 None；将它存入 current_state，
        # 下次若忘记 reset() 又调用 step()，上面的 RuntimeError 就会提醒我们。
        self.current_state = next_state

        return next_state, reward, terminated, {}

# %% [markdown]
# ## 3. 状态编码与 Actor–Critic 网络
#
# `state_to_tensor()` 把状态编号变成 one-hot 张量；`ActorCritic` 再把这个张量同时交给 Actor 和 Critic。
#
# Actor 输出动作的 logits，Critic 输出当前状态的价值 $V(s)$。

# %%
def state_to_tensor(state, n_states):
    """
    把状态编号转换成神经网络可以接收的 one-hot 张量。

    例如 n_states=2 时：
        状态 0 -> [[1.0, 0.0]]
        状态 1 -> [[0.0, 1.0]]

    最外层的 [ ... ] 是批量维度。
    即使现在只输入一个状态，也保留形状 [1, n_states]，这样和批量训练的形状一致。
    """

    state_tensor = torch.zeros((1, n_states), dtype=torch.float32)
    state_tensor[0, state] = 1.0
    return state_tensor


# ============================================================
# 3. 定义 Actor-Critic 网络
# ============================================================

class ActorCritic(nn.Module):
    """
    一个共享主体、带两个输出头的 Actor-Critic 网络。

    输入：
        一个状态的 one-hot 表示，例如 [[1.0, 0.0]]。

    输出：
        logits：Actor 的原始动作分数，后面交给 Categorical 转成概率。
        value： Critic 对当前状态 V(s) 的估计。
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()

        # shared 是 Actor 和 Critic 共同使用的特征提取部分。
        # 这里用一个很小的全连接层，方便演示，不追求复杂性能。
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 16),
            nn.Tanh(),
        )

        # actor_head 输出每个动作的 logits。
        # 注意：logits 还不是概率，也不要求加起来等于 1。
        self.actor_head = nn.Linear(16, action_dim)

        # critic_head 只输出一个数，也就是 V(s)。
        self.critic_head = nn.Linear(16, 1)

    def forward(self, state_tensor):
        """
        对状态做一次前向计算，同时得到 Actor 和 Critic 的输出。
        """

        features = self.shared(state_tensor)
        logits = self.actor_head(features)

        # critic_head 输出形状是 [batch_size, 1]；squeeze(-1) 后变为 [batch_size]。
        value = self.critic_head(features).squeeze(-1)

        return logits, value

# %% [markdown]
# ## 4. 创建环境、模型与优化器
#
# 前面的代码只是定义工具，还没有真正开始训练。
#
# 这一段负责创建环境和网络，并设置 `gamma`、Critic 损失权重、学习率以及 Adam 优化器。

# %%
# 创建环境、网络和优化器。
env = TinyChainEnv()
model = ActorCritic(
    state_dim=env.n_states,
    action_dim=env.n_actions,
)

# gamma 控制未来奖励的折扣程度。
# critic_coef 控制 Critic loss 在总 loss 中的权重。
gamma = 0.9
critic_coef = 0.5

# 学习率决定每次根据梯度修改参数的幅度。
learning_rate = 0.01

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
)

# %% [markdown]
# ## 5. 辅助函数：查看策略与做贪心测试
#
# `get_policy_info()` 查看动作概率和 $V(s)$；`run_greedy_episode()` 在测试时选概率最高的动作，并记录一条完整轨迹。
#
# 这两个函数只读取网络、与环境交互，**不做反向传播、不改参数**。
#
# 注意 `run_greedy_episode()` 里的推进逻辑：只有 `terminated=True` 才跳出循环，否则把 `next_state` 交给 `state` 继续。这不是可选写法——起点执行动作 0 之后 `terminated=False`，如果在这里就结束，智能体永远走不到终点，也就永远拿不到状态 1 的奖励 2。

# %%
# ============================================================
# 5. 一些辅助函数：查看策略和做贪心评估
# ============================================================

def get_policy_info(model, state, n_states):
    """
    查看某个状态下的：
        action_probs：Actor 给出的动作概率
        state_value：Critic 给出的 V(s)

    这里只是查看，不训练，所以使用 no_grad()。
    """

    state_tensor = state_to_tensor(state, n_states)

    with torch.no_grad():
        logits, value = model(state_tensor)
        distribution = Categorical(logits=logits)
        action_probs = distribution.probs.squeeze(0)
        state_value = value.item()

    return action_probs.tolist(), state_value


def run_greedy_episode(model, env):
    """
    用贪心策略跑一局测试。

    训练时 Actor 会按照概率采样动作；
    测试时为了观察当前策略，直接选择 logits 最大的动作，不再随机采样。
    """

    state = env.reset()
    total_return = 0.0
    trajectory = []

    while True:
        state_tensor = state_to_tensor(state, env.n_states)

        with torch.no_grad():
            logits, _ = model(state_tensor)
            action = int(logits.argmax(dim=1).item())

        next_state, reward, terminated, _ = env.step(action)
        trajectory.append((state, action, reward))
        total_return += reward

        if terminated:
            break

        state = next_state

    return total_return, trajectory

# %% [markdown]
# ## 5.5 先把两条路线和两种回报口径摆清楚
#
# 环境里有两条合法路线，它们都会结束：
#
# ```text
# 长路线：状态0 --动作0，奖励0--> 状态1 --动作0或1，奖励2--> 终点
# 短路线：状态0 --动作1，奖励1--> 终点
# ```
#
# 两种口径必须分开算：
#
# | 口径 | 长路线 | 短路线 | 谁在用 |
# |---|---:|---:|---|
# | 不折扣实际回报（奖励直接相加） | $0+2=2.0$ | $1.0$ | 测试时打印的 `episode_return` |
# | 折扣回报（$\gamma=0.9$） | $0+0.9\times2=1.8$ | $1.0$ | Critic 要学的 $V(s)$ |
#
# **两者不是一回事**：训练后期起点 $V(s)$ 会靠近 $1.8$，而同一局打印出来的实际回报是 $2.0$。看到这两个数字不一致，不要当成训练出错。

# %%
# ============================================================
# 5.5 两条路线的回报对照
# ============================================================

routes = {
    "长路线（动作0 → 状态1 → 终点）": [0.0, 2.0],
    "短路线（动作1 直接结束）": [1.0],
}

print(f"折扣因子 gamma = {gamma}")
for name, rewards in routes.items():
    undiscounted = sum(rewards)
    discounted = sum((gamma ** step) * r for step, r in enumerate(rewards))
    print(f"{name}")
    print(f"    奖励序列：{rewards}")
    print(f"    不折扣实际回报：{undiscounted:.3f}")
    print(f"    折扣回报：      {discounted:.3f}")

print()
print("两条路线都会结束，区别在于折扣回报：1.800 > 1.000，")
print("所以 Actor 应当逐渐把起点的动作 0 概率推高。")

# %% [markdown]
# ## 6. 训练：每走一步就更新一次
#
# 按顺序看这段代码：**观察状态 → 抽动作 → 获得奖励 → 算 TD target 和 advantage → 分别计算 Actor/Critic loss → 更新参数**。
#
# 这一整段只需要运行一次；它内部会循环训练 1000 局。
#
# 关于循环的收尾，有三条必须记牢：
#
# 1. `terminated=False` 时**必须**把 `state = next_state` 交回去，否则下一步会拿旧状态重复做决定，长路线永远走不到终点；
# 2. `terminated=True` 时才 `break`，结束这一局；
# 3. 环境每次交互都要经过 `env.step(action)`，网络不会自己"想"出下一状态。
#
# 训练循环里还记录了两条损失的历史值，用于在检查点打印 Actor loss 与 Critic loss 的差别：**Actor loss 调整动作概率，Critic loss 把 $V(s)$ 拉向 TD target**，二者量纲和符号含义都不同。

# %%
# ============================================================
# 6. Actor-Critic 训练循环
# ============================================================

num_episodes = 1000
loss_log = []          # 记录每一步的 (actor_loss, critic_loss) 数值，只用于观察

for episode in range(num_episodes):
    # 每一局都从起点重新开始。
    state = env.reset()
    episode_return = 0.0

    while True:
        # --------------------------------------------------------
        # 第 1 步：Actor 和 Critic 同时观察当前状态。
        # --------------------------------------------------------
        state_tensor = state_to_tensor(state, env.n_states)
        logits, value = model(state_tensor)

        # 把 Actor 的 logits 交给 Categorical。
        # 它会根据 logits 创建一个离散动作分布，并自动完成 softmax。
        distribution = Categorical(logits=logits)

        # 按概率随机抽取一个动作。本例通过采样探索，
        # 如果在训练时总是 argmax，就可能一直不尝试另一条路线。
        action_tensor = distribution.sample()
        action = int(action_tensor.item())

        # log_prob 是“这次实际抽中的动作”的对数概率。
        # Actor 后面要用它和 advantage 一起构造策略损失。
        log_prob = distribution.log_prob(action_tensor).squeeze(0)

        # value 的形状原本是 [1]，squeeze(0) 后变成一个标量。
        current_value = value.squeeze(0)

        # --------------------------------------------------------
        # 第 2 步：和环境交互，得到一条真实经历。
        # --------------------------------------------------------
        next_state, reward, terminated, _ = env.step(action)
        episode_return += reward

        # --------------------------------------------------------
        # 第 3 步：用 Critic 计算 TD target。
        # --------------------------------------------------------
        # 真正终止时没有未来状态价值；未终止时才使用 V(s_next) bootstrap。
        with torch.no_grad():
            if terminated:
                next_value = torch.tensor(0.0)
            else:
                next_state_tensor = state_to_tensor(next_state, env.n_states)
                _, next_value = model(next_state_tensor)
                next_value = next_value.squeeze(0)

            # TD target = 当前奖励 + 折扣后的下一状态价值。
            # terminated 为 True 时，后面的乘数变成 0，目标就只剩 reward。
            not_done = 0.0 if terminated else 1.0
            td_target = torch.tensor(float(reward)) + gamma * next_value * not_done

        # --------------------------------------------------------
        # 第 4 步：计算 advantage，也就是 TD 误差。
        # --------------------------------------------------------
        # 如果 td_target > current_value，说明这次结果比 Critic 原来的估计好。
        # 如果 td_target < current_value，说明这次结果比预期差。
        advantage = td_target - current_value

        # --------------------------------------------------------
        # 第 5 步：分别构造 Actor loss 和 Critic loss。
        # --------------------------------------------------------
        # Actor loss：
        #   advantage > 0 -> 倾向于提高这次动作的概率；
        #   advantage < 0 -> 倾向于降低这次动作的概率。
        #
        # detach() 表示 Actor 可以读取 advantage 的数值，
        # 但 Actor loss 的梯度不要沿着 advantage 反向传回 Critic。
        actor_loss = -log_prob * advantage.detach()

        # Critic loss：让 V(s) 靠近 TD target。
        # 上面的 no_grad() 已经切断了 target 的梯度；这里再次 detach() 只是强调目标是常量。
        critic_loss = (current_value - td_target.detach()).pow(2)

        # 总损失同时训练两个角色。
        # critic_coef 只是平衡 Critic loss 的相对权重。
        total_loss = actor_loss + critic_coef * critic_loss

        # --------------------------------------------------------
        # 第 6 步：反向传播并更新网络参数。
        # --------------------------------------------------------
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # 记录这一步两条损失的数值，只用于后面打印，不参与计算。
        loss_log.append((actor_loss.item(), critic_loss.item()))

        # 如果回合结束，就开始下一局；否则继续当前回合。
        if terminated:
            break

        state = next_state

    # 每隔一段时间打印一次起点处的策略，观察 Actor 是否越来越偏向动作 0。
    if episode == 0 or (episode + 1) % 200 == 0:
        start_probs, start_value = get_policy_info(
            model,
            state=0,
            n_states=env.n_states,
        )
        middle_probs, middle_value = get_policy_info(
            model,
            state=1,
            n_states=env.n_states,
        )

        recent_actor_loss = sum(x[0] for x in loss_log[-100:]) / len(loss_log[-100:])
        recent_critic_loss = sum(x[1] for x in loss_log[-100:]) / len(loss_log[-100:])

        print(f"第 {episode + 1:4d} 局")
        print(f"  本局实际回报：{episode_return:.3f}")
        print(f"  起点动作概率：[动作0={start_probs[0]:.3f}, 动作1={start_probs[1]:.3f}]")
        print(f"  起点 V(s)：{start_value:.3f}")
        print(f"  中间状态 V(s)：{middle_value:.3f}")
        print(f"  最近 100 步平均 Actor loss：{recent_actor_loss:+.4f}"
              f"   平均 Critic loss：{recent_critic_loss:.4f}")
        print()

# %% [markdown]
# ### 两条损失在算什么
#
# | | Actor loss | Critic loss |
# |---|---|---|
# | 表达式 | $-\log\pi_\theta(a\mid s)\cdot \hat A$ | $\big(V(s)-y\big)^2$ |
# | 调整什么 | 本次动作被选中的概率 | 状态价值估计 $V(s)$ |
# | 用到 advantage 吗 | 用（并且 `detach()`，只当系数） | 不用，只用 TD target 与当前估计之差 |
# | 符号含义 | 乘以负号是为了"最大化收益"能写成"最小化损失" | 平方误差，恒非负 |
# | 数值大小 | 与 $\log$ 概率和 advantage 同阶 | 与 TD 误差的平方同阶 |
#
# 两者的量级和含义都不同，所以总损失里才需要一个 `critic_coef` 来平衡。`advantage.detach()` 的作用是：Actor 读取 advantage 的**数值**，但不让 Actor loss 的梯度沿着它回传到 Critic——否则同一份 TD 误差会被两个目标同时拉扯。

# %% [markdown]
# ## 6.5 把一局拆开：reward、return、advantage 各自出现在哪里
#
# 下面在训练完成后跑一局带采样的完整回合，把每一步的中间量都打印出来。这一节**只观察，不更新参数**：TD target、advantage、两条损失都在 `no_grad()` 里算，没有 `backward()`，也没有 `optimizer.step()`。
#
# 三个量要分清：
#
# - **reward**：这一步环境给了多少分；
# - **return**：这一局从开始到当前累计拿了多少分（不断更新）；
# - **advantage**：这一步的 TD target 比 Critic 原来的估计高多少，决定 Actor 往哪边调。

# %%
# ============================================================
# 6.5 单局逐步追踪：reward / return / advantage / 两条损失
# ============================================================

def trace_one_episode(model, env, gamma):
    """
    跑一局带采样的回合，逐步记录中间量。

    注意：这里的损失数值只是"算出来看"，没有 backward，也不更新参数。
    """
    rows = []
    state = env.reset()
    episode_return = 0.0

    while True:
        state_tensor = state_to_tensor(state, env.n_states)

        with torch.no_grad():
            logits, value = model(state_tensor)
            distribution = Categorical(logits=logits)
            probs = distribution.probs.squeeze(0)
            action_tensor = distribution.sample()
            action = int(action_tensor.item())
            log_prob = distribution.log_prob(action_tensor).squeeze(0)
            current_value = value.squeeze(0)

        next_state, reward, terminated, _ = env.step(action)
        episode_return += reward

        with torch.no_grad():
            if terminated:
                next_value = torch.tensor(0.0)
            else:
                _, next_value = model(state_to_tensor(next_state, env.n_states))
                next_value = next_value.squeeze(0)

            not_done = 0.0 if terminated else 1.0
            td_target = torch.tensor(float(reward)) + gamma * next_value * not_done
            advantage = td_target - current_value
            actor_loss = -log_prob * advantage
            critic_loss = (current_value - td_target).pow(2)

        rows.append({
            "state": state,
            "action": action,
            "probs": [round(p, 4) for p in probs.tolist()],
            "reward": reward,
            "return": episode_return,
            "current_value": round(current_value.item(), 4),
            "next_value": round(next_value.item(), 4),
            "td_target": round(td_target.item(), 4),
            "advantage": round(advantage.item(), 4),
            "actor_loss": round(actor_loss.item(), 4),
            "critic_loss": round(critic_loss.item(), 5),
            "terminated": terminated,
        })

        if terminated:
            break

        # terminated=False 时必须推进状态，否则下一轮还在原地做决定。
        state = next_state

    return rows


trace_rows = trace_one_episode(model, env, gamma)
for i, row in enumerate(trace_rows, start=1):
    print(f"第 {i} 步：状态={row['state']}  动作={row['action']}  "
          f"动作概率={row['probs']}")
    print(f"    reward={row['reward']:+.2f}  累计 return={row['return']:+.2f}  "
          f"terminated={row['terminated']}")
    print(f"    V(s)={row['current_value']:+.4f}  V(s_next)={row['next_value']:+.4f}  "
          f"td_target={row['td_target']:+.4f}")
    print(f"    advantage={row['advantage']:+.4f}  actor_loss={row['actor_loss']:+.4f}  "
          f"critic_loss={row['critic_loss']:.5f}")
print()
print(f"这一局的 reward 之和（不折扣）= {sum(r['reward'] for r in trace_rows):.3f}")
print(f"最后一步的累计 return        = {trace_rows[-1]['return']:.3f}")
print("advantage 的符号说明：正数 = 这一步比 Critic 原来的估计好；负数 = 比预期差。")

# ------------------------------------------------------------
# 对照：同样的一局，在"训练开始前"是什么样子
# ------------------------------------------------------------
# 训练好的模型已经收敛，advantage 接近 0，看不出信号的方向。
# 换一个刚初始化的网络跑同一局，advantage 会明显偏离 0。
torch.manual_seed(11)
fresh_model = ActorCritic(state_dim=env.n_states, action_dim=env.n_actions)
fresh_rows = trace_one_episode(fresh_model, env, gamma)

print()
print("对照：刚初始化、还没训练过的网络，同样跑一局")
for i, row in enumerate(fresh_rows, start=1):
    print(f"  第 {i} 步：状态={row['state']} 动作={row['action']}  "
          f"reward={row['reward']:+.2f}  V(s)={row['current_value']:+.4f}  "
          f"td_target={row['td_target']:+.4f}  advantage={row['advantage']:+.4f}")
print("  训练前 V(s) 离 1.8 很远，advantage 也不接近 0；")
print("  训练的过程就是让 Critic 把 V(s) 拉到折扣回报附近、Actor 把长路线概率推高。")

# %% [markdown]
# ### 如果忘记推进状态会怎样
#
# 写循环时最容易漏的一行是 `state = next_state`。漏掉之后，环境内部其实已经走到了状态 1，但循环变量还停在状态 0：下一步会拿着旧状态去问网络，再做一次"起点决定"，于是要么原地打转，要么与环境状态错位。
#
# 环境对这种情况做了防护：一局结束后如果还继续调用 `step()`，会直接报 `RuntimeError`。下面这个刻意写错的小循环会触发它——报错本身就是最好的提醒。

# %%
# ============================================================
# 6.6 反例：忘了把 next_state 交回 state
# ============================================================

env_demo = TinyChainEnv()
state = env_demo.reset()
print("反例：每轮都从旧状态出发，且不检查 terminated。")
try:
    for i in range(1, 4):
        next_state, reward, terminated, _ = env_demo.step(0)   # 动作 0：状态0 → 状态1
        print(f"  第 {i} 次调用 step：拿到 reward={reward:+.1f}  terminated={terminated}"
              f"  next_state={next_state}，但循环变量 state 仍然是 {state}")
        # 故意不写 state = next_state；也不在 terminated 时 break。
except RuntimeError as exc:
    # i 停在引发异常的那一次调用上，不要写死次数。
    print(f"  第 {i} 次调用 step 时环境报错：{exc}")

print()
print("正确写法：")
state = env_demo.reset()
steps = 0
while True:
    next_state, reward, terminated, _ = env_demo.step(0)
    steps += 1
    if terminated:
        break
    state = next_state      # ← 未终止时必须推进
print(f"  推进状态 + 到终点才停：走了 {steps} 步到达终点（reward={reward:+.1f}）")

# %% [markdown]
# ## 7. 评估：看看学到的策略实际怎么走
#
# 这里不再训练，只查看动作概率和价值估计，然后用贪心策略走出轨迹。
#
# 注意：打印的实际奖励总和是未折扣的 $2$，起点 $V(s)$ 则接近折扣回报 $1.8$。

# %%
# ============================================================
# 7. 训练结束后的贪心评估
# ============================================================

print("训练完成，开始进行贪心评估。")

final_start_probs, final_start_value = get_policy_info(
    model,
    state=0,
    n_states=env.n_states,
)

print(
    "起点最终动作概率："
    f"[动作0={final_start_probs[0]:.3f}, "
    f"动作1={final_start_probs[1]:.3f}]"
)
print(f"起点最终价值 V(s)：{final_start_value:.3f}")

# 测试阶段关闭随机采样，使用 logits.argmax() 选择当前最优动作。
# 当前环境完全确定，重复 100 局得到的轨迹相同；平均值只是演示通用评估方式。
test_returns = []
long_path_count = 0

for _ in range(100):
    test_return, trajectory = run_greedy_episode(model, env)
    test_returns.append(test_return)

    # 起点执行动作 0 代表走较长路线。
    if trajectory[0][1] == 0:
        long_path_count += 1

average_test_return = sum(test_returns) / len(test_returns)
long_path_rate = long_path_count / len(test_returns)

print(f"贪心测试平均实际回报（未折扣）：{average_test_return:.3f}")
print(f"贪心测试选择长路线的比例：{long_path_rate:.1%}")

# 打印一条最终的贪心轨迹，帮助观察智能体实际走了哪条路线。
one_return, one_trajectory = run_greedy_episode(model, env)
print(f"一条贪心轨迹：{one_trajectory}")
print(f"这条轨迹的实际回报：{one_return:.3f}")

# %% [markdown]
# ### 轨迹检查：确认真的走完了长路线
#
# 打印出来的 `[(0, 0, 0.0), (1, 0, 2.0)]` 有三层含义，缺一层都不算"走完"：
#
# 1. 第 1 个元素 `(0, 0, 0.0)`：在**状态 0** 选了**动作 0**，拿到奖励 $0$；
# 2. 第 2 个元素 `(1, 0, 2.0)`：在**状态 1** 又做了一次决定，拿到奖励 $2$；
# 3. 轨迹长度为 2，说明起点那一步 `terminated=False` 时**确实继续推进了状态**，而不是提前结束。
#
# 下面用断言把这些条件固定下来。如果策略退化、只走短路线，或者循环提前退出，这里会直接报错。

# %%
# ============================================================
# 7.1 轨迹断言：真的走完长路线
# ============================================================

print(f"轨迹：{one_trajectory}")

assert len(one_trajectory) == 2, "长路线应当包含两步（状态0一次、状态1一次）"
assert one_trajectory[0][0] == 0, "第一步应当发生在起点状态 0"
assert one_trajectory[0][1] == 0, "起点应当选择动作 0（长路线入口）"
assert one_trajectory[0][2] == 0.0, "起点这一步的奖励应为 0"
assert one_trajectory[1][0] == 1, "第二步应当发生在中间状态 1，说明状态被正确推进"
assert one_trajectory[1][2] == 2.0, "最后一步应当拿到终点奖励 2.0"

# 轨迹能走完，说明循环在 terminated=False 时执行了 state = next_state，
# 并且在 terminated=True 时才结束。
print("轨迹检查通过：从起点出发、经过中间状态、在终点结束。")

# 再检查一次终止语义：在终止状态之后继续调用 step 应当报错。
env_check = TinyChainEnv()
env_check.reset()
env_check.step(1)          # 短路线：直接结束
try:
    env_check.step(1)
    raise AssertionError("终止之后不应再允许 step")
except RuntimeError:
    print("终止语义检查通过：回合结束后继续 step 会报错，必须先 reset()。")

# 评估阶段的网络参数不应被测试过程改动。
params_before_test = [p.detach().clone() for p in model.parameters()]
run_greedy_episode(model, env)
unchanged = all(torch.equal(before, after) for before, after in
                zip(params_before_test, model.parameters()))
print(f"100 局测试之后参数是否保持不变：{unchanged}")
assert unchanged, "评估阶段不应更新任何参数"

# %% [markdown]
# ## 7.2 四项对照
#
# | 指标 | 含义 | 谁负责算 |
# |---|---|---|
# | 折扣回报（约 $1.8$） | Critic 要学的 $V(s)$，含折扣 | Critic 的价值头 |
# | 不折扣实际回报（$2.0$） | 一局里奖励直接相加 | 环境给出的奖励序列 |
# | 起点动作 0 的概率 | Actor 当前认为长路线有多好 | Actor 的动作头 |
# | 长路线比例（$100\%$） | 贪心策略在 100 局测试中走长路线的比例 | 测试轨迹统计 |
#
# 这四个数字回答的是四个不同问题。把它们放在一起看，才能确认"策略学到了长路线"这件事：**Actor 的概率高、Critic 的价值靠近 1.8、贪心轨迹真的走到终点**。

# %%
# ============================================================
# 7.2 四项对照一次打印
# ============================================================

theoretical_discounted = 0.0 + gamma * 2.0

comparison = [
    ("折扣回报（理论值，Critic 的学习目标）", f"{theoretical_discounted:.3f}"),
    ("起点 V(s)（训练后 Critic 的估计）", f"{final_start_value:.3f}"),
    ("不折扣实际回报（测试平均）", f"{average_test_return:.3f}"),
    ("起点动作 0 的概率（Actor 输出）", f"{final_start_probs[0]:.3f}"),
    ("起点动作 1 的概率（Actor 输出）", f"{final_start_probs[1]:.3f}"),
    ("长路线成功比例（贪心 100 局）", f"{long_path_rate:.1%}"),
]

for name, value in comparison:
    print(f"{name:38s}{value}")

print()
print(f"折扣回报 {theoretical_discounted:.3f} 与不折扣实际回报 "
      f"{average_test_return:.3f} 的差：{average_test_return - theoretical_discounted:.3f}")
print("这个差来自折扣：状态 1 的奖励 2 在起点看来只值 0.9 × 2 = 1.8。")

# %% [markdown]
# ## 常见误区
#
# 1. **把 `log_prob` 取成概率最大动作的对数概率。** 它对应**本次实际采样**到的动作；采样到动作 1 就取 $\log\pi(\text{动作1}\mid s)$，与哪个动作概率最大无关。
# 2. **把终止转移的 target 再乘一次 $\gamma$。** 终止时 $y = r$；写成 $\gamma r$ 就把当前奖励也打了折。折扣只作用于**未来**价值，不作用于当前奖励。
# 3. **忘记 `state = next_state`。** 起点执行动作 0 后 `terminated=False`，不推进状态就永远到不了终点，也永远拿不到奖励 2。本节 6.6 的反例演示了后果。
# 4. **把 advantage 的符号看反。** $\hat A>0$ 表示这一步比 Critic 原来的估计好，Actor 倾向提高该动作概率；$\hat A<0$ 则相反。这与"奖励是正还是负"不是一回事——即时奖励为负但后续价值更高时，advantage 仍可能为正。
# 5. **以为训练时也该用 `argmax`。** 训练必须采样，否则另一条路线可能永远不被尝试，策略会被最初的概率分布锁死；`argmax` 只用在测试阶段。
# 6. **把整局回报直接当单步优势。** 本节用的是单步 TD 误差 $y-V(s)$；原始 REINFORCE 用整局回报 $G_t$，两者是不同的优势估计方式。
# 7. **以为 `advantage.detach()` 会改变数值。** 它只切断梯度路径，不改变数字；去掉它会让 Actor loss 的梯度额外流回 Critic。
# 8. **把 `critic_loss` 当成评估指标。** 它是训练用的平方误差，与测试回报不是同一个量。
#
# ## 本节边界与自检
#
# 能运行核对的内容：单步 Actor–Critic 更新、TD target 与 advantage 的数值、两条损失的分工、轨迹断言、四项对照。
#
# 尚未覆盖的内容：多步回报与 GAE、概率比值裁剪（PPO 的核心）、批量采样后多轮更新、熵正则、并行环境、连续动作空间。
#
# 自检题：
#
# 1. 起点动作 0 的那一步：`reward`、`td_target`、`current_value`、`advantage` 各是多少？为什么 advantage 与 $V(1)$ 有关？
# 2. 为什么终止时 `td_target = reward`？如果写成 `reward + gamma * V(s_next)` 会怎样？
# 3. 若 `advantage < 0`，Actor 会把本次动作的概率往哪边调？Critic 会把 $V(s)$ 往哪边调？
# 4. `actor_loss` 与 `critic_loss` 分别用了 advantage 和 TD target 的哪个部分？为什么 Actor 那边要 `detach()`？
# 5. 训练时把 `distribution.sample()` 换成 `argmax`，会发生什么？
