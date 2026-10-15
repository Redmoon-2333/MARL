# Generated from the adjacent notebook by scripts/marl.py export.

# %% [markdown]
# # Day1 · 从奖励、回报到 Q-learning 与 SARSA
#
# 这个 Notebook 只处理一个问题：**在没有环境模型、只能靠自己试错的条件下，怎样把一次交互的结果变成对动作价值的修正。**
#
# 先约定后面反复出现的几个词：
#
# | 术语 | 在这个实验里的含义 | 记号 |
# |---|---|---|
# | 状态 state | 智能体当前处在哪个格/哪个环节 | $S$ |
# | 动作 action | 当前状态下可以做的事 | $a$ |
# | 奖励 reward | 走完一步之后环境立刻给的分数 | $r$ |
# | 回报 return | 从某一时刻开始，往后所有奖励累加起来的总分 | $G_t$ |
# | 折扣因子 discount | 未来奖励打折的比例，越远越不值钱 | $\gamma$ |
# | 策略 policy | 看到状态后决定选哪个动作的规则 | $\pi$ |
# | 动作价值 Q | 在状态 $s$ 做动作 $a$ 大概能拿多少总分 | $Q(s,a)$ |
#
# 要区分的是**奖励和回报**：奖励是单步反馈，可能很小甚至是负的；回报是把后续每一步都算进来的总量。第一步拿负分不代表这个动作不好，只要它把后面更大的收益接了上来。

# %% [markdown]
# ## 1. 先从两条路线看"回报"是怎么算的
#
# 悬崖网格里有两条到终点的路线（这个环境在第 2 个 Notebook 里会用代码搭出来）：
#
# ```text
# 路线 A：起点 → 普通格 ×6 → 终点        共 7 步
# 路线 B：起点 → 普通格 ×8 → 终点        共 9 步
#
# 普通格奖励 = -0.04     终点奖励 = +1     悬崖 = 失败
# ```
#
# 两条路线的终点奖励一样，差别只在普通格数量。下面用代码把两种口径都算一遍：不打折（$\gamma=1$）和打折（$\gamma=0.9$）。

# %%
# ============================================================
# 1. 手算 + 代码核对：奖励怎么累加成回报
# ============================================================

# 环境给出的单步奖励。
STEP_REWARD = -0.04
GOAL_REWARD = 1.0

# 路线 A 走 6 个普通格后到终点；路线 B 走 8 个普通格。
rewards_A = [STEP_REWARD] * 6 + [GOAL_REWARD]
rewards_B = [STEP_REWARD] * 8 + [GOAL_REWARD]

# 不打折的回报：把所有奖励直接相加，等价于 gamma = 1。
return_A = sum(rewards_A)
return_B = sum(rewards_B)

print("路线 A 奖励序列：", rewards_A)
print("路线 B 奖励序列：", rewards_B)
print(f"不打折回报：A = {return_A:.2f}   B = {return_B:.2f}")
# 6 * (-0.04) + 1 = 0.76   8 * (-0.04) + 1 = 0.68
# 两条路线终点奖励相同，A 的优势来自更少的步数成本，
# 不需要"终点奖励特别大"这个额外假设。


def discounted_return(rewards, gamma):
    """从第一个奖励开始，把折扣后的奖励累加成一个回报。"""
    total = 0.0
    for step, reward in enumerate(rewards):
        # gamma ** step：越靠后的奖励打的折越多。
        total += (gamma ** step) * reward
    return total


gamma = 0.9
print(f"打折回报(gamma=0.9)：A = {discounted_return(rewards_A, gamma):.3f}   "
      f"B = {discounted_return(rewards_B, gamma):.3f}")
# 打折后两条路线的差距被放大：前面几个负奖励被 gamma 缩得更小，
# 但 A 少走两步的差距依然存在，排序不变。

# %% [markdown]
# **回报的递推写法。** 上面的累加可以改写成"这一步的奖励 + 后面剩余的折扣回报"：
#
# $$
# G_t = r_{t+1} + \gamma G_{t+1}
# $$
#
# 这不是数学技巧，而是后面所有算法的骨架：**一次更新只需要当前奖励和下一状态的价值**，不必等整条轨迹算完。若第 1 步奖励为 $-0.04$、剩余回报为 $0.9\times1=0.9$，那么从出发时看的完整回报就是 $-0.04+0.9=0.86$。
#
# 超出本次范围的边界：动态规划要求已知完整转移规则，这里的智能体并不知道；它只能从采样到的经验里估计 $Q(s,a)$，这正是接下来的内容。

# %% [markdown]
# ## 2. 从固定规则到从经验学习
#
# 第一个可运行实验是一个只有两个状态的小世界：
#
# ```text
# 起点 ──动作 A──> 终点        奖励 +1，结束
#      └─动作 B──> 中途 ──"继续"──> 终点   奖励 0，再到 +2
# ```
#
# - **策略**是选动作的规则。这里用 $\varepsilon$-greedy：以 $\varepsilon$ 的概率随机试一个动作（探索），其余时候选当前 Q 值最大的动作（利用）。
# - **Q 表**记录每个状态、每个动作的当前估值，一开始全是 0，智能体并不知道答案。
# - **学习率** $\alpha$ 决定这次修正的幅度：新估计保留一部分旧估计，再加上一部分新经验，不是把旧值直接覆盖掉。
#
# 学习率的写法是：
#
# $$
# Q(s,a) \leftarrow Q(s,a) + \alpha\big(\text{target} - Q(s,a)\big)
# $$
#
# 括号里的 `target - Q(s,a)` 就是 **TD 误差**：这次的目标比原来的估计高还是低。

# %%
import random

# 固定随机种子，重复运行时探索到的动作顺序一致。
rng = random.Random(7)
# Q 表：每个状态对应一组动作价值；起点有 A/B 两个动作，中途只有“继续”。
q = {"起点": [0.0, 0.0], "中途": [0.0]}  # 起点动作 0=A、1=B
# alpha 控制每次更新的幅度，gamma 给未来奖励打折，epsilon 是随机探索的概率。
alpha = 0.2
gamma = 0.9
epsilon = 0.2

# 每一轮从起点重新开始；A 一步结束，B 需要经过中途再走一步。
# epoch 是从 0 开始的回合索引，用来定位训练过程中的检查点。
for epoch in range(400):
    state = "起点"

    while True:
        # 可选动作的编号由当前状态的 Q 值列表长度决定。
        actions = range(len(q[state]))
        if rng.random() < epsilon:
            action = rng.choice(list(actions))  # 探索
        else:
            action = max(actions, key=lambda a: q[state][a])  # 利用

        # 环境转移：A 立即得 1；B 先得 0，到中途后继续得 2。
        if state == "起点" and action == 0:
            reward, next_state, terminated = 1.0, None, True
        elif state == "起点":  # 选择 B，先去中途
            reward, next_state, terminated = 0.0, "中途", False
        else:  # 从中途走到终点
            reward, next_state, terminated = 2.0, None, True

        # 真正结束时没有后续价值；否则取下一状态中最大的 Q 值作为未来估计。
        future = 0.0 if terminated else max(q[next_state])
        target = reward + gamma * future
        # 新 Q 值 = 旧 Q 值 + 学习率 × (这次的目标值 - 旧 Q 值)。
        q[state][action] += alpha * (target - q[state][action])

        # 一局结束就重开；没结束则进入下一状态，继续做动作选择。
        if terminated:
            break
        state = next_state

    # 每隔 100 回合打印一次 Q 表快照；epoch=0 是第 1 回合结束后的值。
    if(epoch % 100 == 0):
        print("epoch:", epoch)
        print("Q(起点, A) =", round(q["起点"][0], 3))
        print("Q(起点, B) =", round(q["起点"][1], 3))
        print("Q(中途, 继续) =", round(q["中途"][0], 3))

# A 约为 1；B 约为 0 + 0.9 × 2 = 1.8，因此学到的策略更偏向 B。
print("Q(起点, A) =", round(q["起点"][0], 3))
print("Q(起点, B) =", round(q["起点"][1], 3))
print("Q(中途, 继续) =", round(q["中途"][0], 3))

# %% [markdown]
# ### 输出怎么读
#
# - `Q(中途, 继续)` 收敛到 $2$：这个状态只有一条路，走一步就拿 $2$ 分，不需要打折。
# - `Q(起点, A)` 收敛到 $1$：动作 A 立刻拿到 $1$ 分，之后没有未来了。
# - `Q(起点, B)` 收敛到 $0 + 0.9\times2=1.8$：动作 B 这一步奖励是 $0$，但它把智能体送到了还能拿 $2$ 分的中途状态，折扣之后值 $1.8$。
#
# 所以**起点位置最终更偏好 B**：单步奖励为 $0$ 的动作反而是更值钱的动作。这与"第一步拿负分也可能是好动作"是同一件事。

# %% [markdown]
# ## 3. 把一次更新拆开打印
#
# 上面的循环把 `future`、`target`、TD 误差都算出来了，但没有打印。下面用已经训练好的 Q 表跑一局贪心轨迹，把每一步的中间变量摊开：
#
# - `future`：下一步的价值估计（终止时为 0）
# - `target`：这一步的 TD 目标 = 当前奖励 + 折扣 × future
# - `td_error`：`target - Q_old`，正数表示"原来低估了"，负数表示"原来高估了"

# %%
# ============================================================
# 3. 逐步打印：一局贪心轨迹里的 future / target / TD 误差
# ============================================================

def run_greedy_episode_with_trace(q, gamma, alpha, verbose=True):
    """
    用当前 Q 表跑一局贪心轨迹。

    每一行记录：
        state       当前状态
        action      选中的动作编号
        q_old       更新前的 Q(s, a)
        reward      这一步的即时奖励
        future      下一步的价值估计（终止为 0）
        target      当前奖励 + 折扣 × future
        td_error    target - q_old
        q_new       更新后的 Q(s, a)

    注意：这个函数只读取和更新当前状态的 Q 值；
    它不改变动作选择规则（贪心），也不额外引入探索。
    """
    trace = []
    state = "起点"

    while True:
        action = max(range(len(q[state])), key=lambda a: q[state][a])  # 贪心

        if state == "起点" and action == 0:
            reward, next_state, terminated = 1.0, None, True
        elif state == "起点":
            reward, next_state, terminated = 0.0, "中途", False
        else:
            reward, next_state, terminated = 2.0, None, True

        q_old = q[state][action]
        future = 0.0 if terminated else max(q[next_state])
        target = reward + gamma * future
        td_error = target - q_old
        q[state][action] = q_old + alpha * td_error

        trace.append({
            "state": state, "action": action, "q_old": q_old, "reward": reward,
            "future": future, "target": target, "td_error": td_error,
            "q_new": q[state][action], "terminated": terminated,
        })

        if terminated:
            break
        state = next_state

    if verbose:
        for i, row in enumerate(trace):
            print(f"第 {i + 1} 步 状态={row['state']}  动作={row['action']}")
            print(f"    q_old={row['q_old']:.4f}  reward={row['reward']:.2f}  "
                  f"future={row['future']:.4f}")
            print(f"    target={row['target']:.4f}  td_error={row['td_error']:+.4f}  "
                  f"q_new={row['q_new']:.4f}  terminated={row['terminated']}")
    return trace


trace = run_greedy_episode_with_trace(q, gamma=gamma, alpha=alpha)
total_reward = sum(row["reward"] for row in trace)
print()
print(f"这一局的未折扣实际回报 = {total_reward:.2f}")
print(f"起点 Q 值在更新前是 {trace[0]['q_old']:.4f}，更新后变成 {trace[0]['q_new']:.4f}")

# %%
# ============================================================
# 4. 补一个"训练前"的更新：TD 误差不为 0 时 Q 值怎么动
# ============================================================

# 训练开始时 Q 表全是 0。假设第一条经历是：
# 在起点执行动作 0（直接到终点），拿到奖励 1，回合结束。
q_start = [0.0, 0.0]
action = 0
reward = 1.0
terminated = True

# 终止时没有未来，future 为 0。
future = 0.0 if terminated else float("nan")
target = reward + gamma * future

q_old = q_start[action]
td_error = target - q_old

# 这里把 alpha 设为 0.5，方便与手算对照：0 + 0.5 × (1 - 0) = 0.5
alpha_demo = 0.5
q_new = q_old + alpha_demo * td_error

print(f"future   = {future:.2f}   （回合结束，后面没有收益）")
print(f"target   = {reward:.2f} + {gamma} × {future:.2f} = {target:.2f}")
print(f"q_old    = {q_old:.2f}")
print(f"td_error = {target:.2f} - {q_old:.2f} = {td_error:+.2f}")
print(f"q_new    = {q_old:.2f} + {alpha_demo} × {td_error:+.2f} = {q_new:.2f}")
print()
print("一次更新只走一半（alpha=0.5），旧估计 0 与新经验 1 各占一部分；")
print("如果 alpha=1，才会一步跳到 1.00。")

# %% [markdown]
# ### 这几列之间的关系
#
# | 列 | 用什么算出来 | 含义 |
# |---|---|---|
# | `future` | 下一状态的最大 Q 值（终止时 0） | 对"以后还能拿多少"的当前估计 |
# | `target` | `reward + gamma * future` | 这一步**希望**Q 值靠近的数 |
# | `td_error` | `target - q_old` | 这次经验说明原来估计偏低（正）还是偏高（负） |
# | `q_new` | `q_old + alpha * td_error` | 修正后的估计 |
#
# 注意两点：
#
# 1. **TD 误差为 0 不代表动作不好**，只表示这次的结果与原来的估计一致，没有新信息可学。
# 2. **终止之后没有未来**：`terminated=True` 时 `future` 直接取 0，不是"下一状态价值恰好是 0"。若把终止状态的价值也拿来做 bootstrap，等于凭空给回合结束后加了收益。

# %% [markdown]
# ## 4. Q-learning 与 SARSA：差别只在 target 用哪个值
#
# 两者都从同一条经验 $(s, a, r, s')$ 学习，公式外形也完全一样，唯一区别是 target 里"下一步"取什么：
#
# $$
# \begin{aligned}
# \text{Q-learning:}\quad & \text{target} = r + \gamma \max_{a'} Q(s', a') \\
# \text{SARSA:}\quad & \text{target} = r + \gamma\, Q(s', a')
# \end{aligned}
# $$
#
# - Q-learning 假设下一状态会挑**当前估计最好**的动作，即使实际没走那一步。
# - SARSA 只用**行为策略实际选中**的下一个动作 $a'$ 的价值。
#
# 下面这个实验可以直接对比：同一条经验、同一个旧估计，只换 target 的口径。

# %% [markdown]
# | | Q-learning | SARSA |
# |---|---:|---:|
# | 采用的后续价值 | 最大值 `0.8` | 实际选中动作的值 `0.2` |
# | target | $-0.1+0.9\times0.8=0.62$ | $-0.1+0.9\times0.2=0.08$ |
# | 更新后的 $Q(S,a)$ | $0.4+0.5(0.62-0.4)=0.51$ | $0.4+0.5(0.08-0.4)=0.24$ |

# %%
old_q = 0.4
reward = -0.1
gamma = 0.9
alpha = 0.5

next_q = [0.8, 0.2]
actual_next_action = 1  # 假设探索实际选中了第二个动作

q_learning_target = reward + gamma * max(next_q)
sarsa_target = reward + gamma * next_q[actual_next_action]

q_learning_new = old_q + alpha * (q_learning_target - old_q)
sarsa_new = old_q + alpha * (sarsa_target - old_q)

print("Q-learning:", q_learning_target, q_learning_new)
print("SARSA:     ", sarsa_target, sarsa_new)

# 补上中间变量：两个算法的差别可以完全用 target 和 TD 误差解释。
print()
print("Q-learning 的 TD 误差:", q_learning_target - old_q)
print("SARSA      的 TD 误差:", sarsa_target - old_q)
print("两个 target 的差:", q_learning_target - sarsa_target)
print("两个新 Q 值的差:", q_learning_new - sarsa_new)

# %% [markdown]
# ### 手算复核
#
# | 量 | Q-learning | SARSA |
# |---|---|---|
# | 后续价值 | $\max(0.8, 0.2)=0.8$ | $Q(s', a'=1)=0.2$ |
# | target | $-0.1+0.9\times0.8=0.62$ | $-0.1+0.9\times0.2=0.08$ |
# | TD 误差 | $0.62-0.4=0.22$ | $0.08-0.4=-0.32$ |
# | 新 $Q$ | $0.4+0.5\times0.22=0.51$ | $0.4+0.5\times(-0.32)=0.24$ |
#
# 同一个旧估计 $0.4$，Q-learning 把它往上推（$0.51$），SARSA 把它往下压（$0.24$）。原因是这次实际选了动作 $1$（价值只有 $0.2$），而 Q-learning 不关心实际选了谁，只认最大值 $0.8$。
#
# 由此可以预期两者在行为上的差别：Q-learning 学的是"假设以后每次都选最好动作"的价值，策略更激进；SARSA 学的是"按当前这套还在探索的策略继续走"的价值，会把探索带来的风险算进去。

# %% [markdown]
# ## 5. 常见误区
#
# 1. **把 TD target 当成"正确答案"。** target 是用当前估计搭出来的，本身也会随训练变化；它不是回放池里存好的标准答案。
# 2. **"取了 max 就等于 Q 值一定变大"。** 变不变取决于 TD 误差的符号。上面 SARSA 的例子 TD 误差为负，Q 值被压低。
# 3. **把 `max` 写成 `argmax` 的语义混用。** `max` 返回最大值本身（用于构造 target），`argmax` 返回取到最大值的动作编号（用于选动作）。两者在同一个更新里都会出现，但用途不同。
# 4. **乘方和乘法的记号。** Python 里 `0.6**0.9` 是乘方（约为 $0.632$），不是 $0.6\times0.9$；折扣项要写成 `0.6 * 0.9`。
# 5. **以为小环境里 SARSA 和 Q-learning 一定有差别。** 本例"中途"只有一个动作，$\max$ 和"实际选中"取到同一个值，两者暂时合流；差别需要在有多个后续动作的状态上才显现。
# 6. **以为学习率是覆盖式的。** $\alpha=0.5$ 时新估计只走一半，这是"旧经验 + 新经验"的加权，不是丢弃旧值。
#
# ## 6. 本节边界与自检
#
# 已经在这个 Notebook 里可运行核对的内容：奖励与回报的累加、折扣回报、$\varepsilon$-greedy 的 Q 表更新、`future`/`target`/TD 误差的逐项数值、Q-learning 与 SARSA 的 target 分叉。
#
# 本节的自我检查题（答不上来就回到对应小节）：
#
# 1. 一条确定路线的奖励依次为 $-0.04$、$-0.04$、$+1$，不打折时完整回报是多少？执行第一步后剩余的回报是多少？两者与"这一步奖励 + 后面剩余的回报"是什么关系？
# 2. 已知 $Q(S,A)=0.4$、$r=-0.1$、$\max_{a'}Q(S',a')=0.6$、$\gamma=0.9$、$\alpha=0.5$，写出 target 和更新后的 $Q(S,A)$。
# 3. 同一条经验下，为什么 Q-learning 的 target 不会低于 SARSA 的 target？
# 4. 到达终点的那一步，`future` 为什么必须是 $0$？
#
# 超出本节的内容：这里的 Q 表把每个状态单独存一行。状态一多、状态连续，表格就放不下了——这是第 2 个 Notebook 要处理的问题。
