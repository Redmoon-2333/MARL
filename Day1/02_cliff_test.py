# Generated from the adjacent notebook by scripts/marl.py export.

# %% [markdown]
# # Day1 · 悬崖环境：把一次完整轨迹走通
#
# 这个 Notebook 用一个 4×6 的悬崖网格，把"探索 → 更新 Q 表 → 用贪心策略走一条完整轨迹"整条链路跑通。
#
# ```text
# 列 0    1     2     3     4     5
# 行 0  .    .     .      .     .     .
# 行 1  .    .     .      .     .     .
# 行 2  .    .     .      .     .     .
# 行 3  S    X     X      X     X     G     ← 底部中间 4 格是悬崖
# ```
#
# | 事件 | 奖励 | 回合是否结束 |
# |---|---:|---|
# | 普通移动（含撞墙） | $-0.04$ | 否 |
# | 掉进悬崖 | $-1$ | 是 |
# | 到达终点 | $+1$ | 是 |
#
# 目的是让"少扣分、早到终点"这件事可以直接看见：沿第 2 行向右走 5 格、再向下一步进终点，一共 7 步、6 次普通步扣分，回报为 $6\times(-0.04)+1=0.76$。

# %%
import random


# =========================
# 1. 定义环境
# =========================

ROWS, COLS = 4, 6

START = (3, 0)
GOAL = (3, 5)

# 底部中间的几个格子是悬崖。
CLIFF = {
    (3, 1),
    (3, 2),
    (3, 3),
    (3, 4),
}

# 动作顺序：
# 0 = 上，1 = 下，2 = 左，3 = 右
ACTIONS = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
]

ARROWS = ["↑", "↓", "←", "→"]

N_STATES = ROWS * COLS
N_ACTIONS = len(ACTIONS)

STEP_REWARD = -0.04
CLIFF_REWARD = -1.0
GOAL_REWARD = 1.0

MAX_STEPS = 100


# 固定随机种子，方便重复运行时得到相同的实验结果。
rng = random.Random(7)


def state_id(cell):
    """
    把二维坐标转换成 Q 表使用的一维状态编号。

    例如：
    (0, 0) -> 0
    (0, 1) -> 1
    (1, 0) -> 6
    (3, 0) -> 18
    """
    row, col = cell
    return row * COLS + col


def step(state, action):
    """
    执行动作，返回：

    next_state: 下一状态
    reward: 当前这一步得到的奖励
    terminated: 这一步是否导致任务真正结束
    """

    # 把一维状态编号还原成二维坐标。
    row, col = divmod(state, COLS)

    # 取出动作造成的行、列变化。
    d_row, d_col = ACTIONS[action]

    # 计算移动后的坐标。
    # min/max 的作用是让智能体不能走出网格边界。
    next_row = min(max(row + d_row, 0), ROWS - 1)
    next_col = min(max(col + d_col, 0), COLS - 1)

    next_cell = (next_row, next_col)
    next_state = state_id(next_cell)

    # 到达终点：得到 +1，并且回合真正结束。
    if next_cell == GOAL:
        return next_state, GOAL_REWARD, True

    # 掉进悬崖：得到 -1，并且回合真正结束。
    if next_cell in CLIFF:
        return next_state, CLIFF_REWARD, True

    # 普通移动：得到小额负奖励，回合继续。
    return next_state, STEP_REWARD, False

# %% [markdown]
# ## 2. 环境转移检查
#
# 在训练之前先把环境本身验证一遍：如果环境的转移规则有错，后面学到的策略再"好看"也没有意义。下面逐项检查 7 件事：
#
# 1. 起点和终点的状态编号；
# 2. 起点、终点、悬崖三者在网格中的位置关系；
# 3. 普通移动的奖励与 `terminated`；
# 4. 越界移动（撞墙）后是否停在原地、是否继续回合；
# 5. 到达终点时 `terminated=True`；
# 6. 掉进悬崖时 `terminated=True`；
# 7. 终止状态的 `next_state` 是否还能被当作"还有未来"使用。
#
# 任何一项不符合预期，下面的 `assert` 会直接报错，而不是让错误悄悄带进训练。

# %%
# =========================
# 2. 环境转移检查
# =========================

def check_transitions():
    """逐项检查环境转移，并把每一步的输入输出打印出来。"""

    print("网格与坐标：")
    for row in range(ROWS):
        line = []
        for col in range(COLS):
            cell = (row, col)
            if cell == START:
                line.append("S")
            elif cell == GOAL:
                line.append("G")
            elif cell in CLIFF:
                line.append("X")
            else:
                line.append(".")
        print("  " + " ".join(line))
    print()

    # ---------- 检查 1：起点与终点的状态编号 ----------
    assert state_id(START) == 18, "起点 (3,0) 的状态编号应为 18"
    assert state_id(GOAL) == 23, "终点 (3,5) 的状态编号应为 23"
    print(f"[1] state_id(START) = {state_id(START)}，state_id(GOAL) = {state_id(GOAL)}")

    # ---------- 检查 2：三者位置关系 ----------
    assert START not in CLIFF and GOAL not in CLIFF, "起点和终点不能是悬崖"
    # 同在第 3 行，起点在第 0 列、终点在第 5 列，中间 4 列全是悬崖。
    assert {c for c in CLIFF if c[0] == 3} == {(3, 1), (3, 2), (3, 3), (3, 4)}
    print(f"[2] 悬崖格：{sorted(CLIFF)}（4 格，位于起点与终点之间）")

    # ---------- 逐项转移检查 ----------
    cases = [
        ("普通移动：起点向右上方走", START, 0, (2, 0), STEP_REWARD, False),
        ("越界：从 (0,0) 继续向上", (0, 0), 0, (0, 0), STEP_REWARD, False),
        ("越界：从 (0,0) 继续向左", (0, 0), 2, (0, 0), STEP_REWARD, False),
        ("普通移动：终点上方下行", (2, 5), 1, GOAL, GOAL_REWARD, True),
        ("掉崖：起点向右一步", START, 3, (3, 1), CLIFF_REWARD, True),
        ("掉崖：从 (2,2) 向下", (2, 2), 1, (3, 2), CLIFF_REWARD, True),
    ]

    for i, (name, cell, action, exp_cell, exp_reward, exp_term) in enumerate(cases, start=3):
        next_state, reward, terminated = step(state_id(cell), action)
        next_cell = divmod(next_state, COLS)
        assert next_cell == exp_cell, f"{name}：期望到 {exp_cell}，实际到 {next_cell}"
        assert reward == exp_reward, f"{name}：期望奖励 {exp_reward}，实际 {reward}"
        assert terminated is exp_term, f"{name}：期望 terminated={exp_term}"
        print(f"[{i}] {name}")
        print(f"     {cell} --{ARROWS[action]}--> {next_cell}   "
              f"reward={reward:+.2f}  terminated={terminated}")

    # ---------- 检查 7：终止的两条出口 ----------
    _, cliff_reward, cliff_terminated = step(state_id(START), 3)
    _, goal_reward, goal_terminated = step(state_id((2, 5)), 1)
    assert cliff_terminated and goal_terminated, "掉崖和到达终点都必须终止回合"
    print()
    print(f"[7] 终止出口检查：掉崖 terminated={cliff_terminated}（reward={cliff_reward:+.2f}），"
          f"到达终点 terminated={goal_terminated}（reward={goal_reward:+.2f}）")
    print("    终止意味着这一步之后没有未来状态，Q 更新时 future 必须取 0，")
    print("    不能继续读下一格的价值。")

    print()
    print("全部环境转移检查通过。")


check_transitions()

# %% [markdown]
# ### 检查结果说明了什么
#
# - **越界被"夹"在边界内**：从 $(0,0)$ 继续向上，`next_cell` 还是 $(0,0)$，奖励仍是普通步的 $-0.04$。智能体撞墙不会出界，也不会额外受罚，只是白白走了一步。
# - **终点和悬崖都给 `terminated=True`，但奖励相反**：$(2,5)$ 向下一步得 $+1$ 并结束；起点向右一步进 $(3,1)$ 得 $-1$ 并结束。
# - **`terminated` 决定"有没有未来"**：这两种结束都意味着不存在 $s'$ 的价值可读，所以 Q 更新里 `future` 必须取 $0$。如果误把终止当成普通转移，就会给已经结束的回合凭空加上未来收益。

# %% [markdown]
# ## 3. ε-greedy 动作选择
#
# ```text
# 以 ε 的概率随机试一个动作（探索）
# 以 1-ε 的概率选当前 Q 值最大的动作（利用）
# ```
#
# 训练阶段必须保留探索：只有走到不同格子、比较不同路线，Q 表才可能学到"第 2 行比底部安全"。如果训练时一直贪心，智能体可能在第一次撞进悬崖后就再也不敢靠近那一带，学不出长距离绕行。

# %%
# =========================
# 3. 定义 ε-greedy 动作选择
# =========================

def choose_action(q_row, epsilon):
    """
    根据当前状态对应的一行 Q 值选择动作。

    epsilon 概率：探索，随机选动作。
    1 - epsilon 概率：利用，选当前 Q 值最大的动作。
    """

    # 探索：暂时不相信当前 Q 表，随机尝试一个动作。
    if rng.random() < epsilon:
        return rng.randrange(N_ACTIONS)

    # 利用：选择当前估计价值最高的动作。
    # 如果多个动作并列，max 默认返回编号较小的动作。
    return max(
        range(N_ACTIONS),
        key=lambda action: q_row[action],
    )

# %% [markdown]
# ## 4. 初始化 Q 表与超参数
#
# Q 表的形状是 `[状态][动作]`：24 个状态 × 4 个动作，初始估值全为 0。
#
# | 超参数 | 取值 | 作用 |
# |---|---:|---|
# | `alpha` | 0.5 | 每次更新走多大幅度 |
# | `gamma` | 0.95 | 未来奖励的折扣，越接近 1 越看重长远 |
# | `epsilon` | 0.2 → 0.02 | 探索概率随训练线性下降 |
# | `num_episodes` | 3000 | 训练回合数 |

# %%
# =========================
# 4. 初始化 Q 表和超参数
# =========================

# Q 表的形状是：
#
#     [状态][动作]
#
# 一共有 24 个状态，每个状态有 4 个动作。
# 初始时所有动作价值都设为 0。
q = [
    [0.0 for _ in range(N_ACTIONS)]
    for _ in range(N_STATES)
]

alpha = 0.5
gamma = 0.95

# 初期多探索，后期逐渐减少探索。
epsilon_start = 0.2
epsilon_min = 0.02

num_episodes = 3000

# %% [markdown]
# ## 5. Q-learning 训练循环
#
# 训练阶段与测试阶段的分工，从这一节开始就要分清：
#
# | | 训练阶段 | 测试阶段 |
# |---|---|---|
# | 动作怎么选 | $\varepsilon$-greedy，保留随机探索 | 每步 `argmax`，不再探索 |
# | Q 表 | 每一步都更新 | 只读取，不更新 |
# | 回报的作用 | 无——更新用的是 TD target | 用于统计这套策略的表现 |
#
# 循环里每一步做四件事：选动作 → 与环境交互 → 用 `reward + gamma * max Q(s')` 算 target → 用 TD 误差修正 $Q(s,a)$。

# %%
# =========================
# 5. Q-learning 训练循环
# =========================

for episode in range(num_episodes):

    # 每个 episode 都从起点开始。
    state = state_id(START)

    # 让 epsilon 随训练逐渐下降。
    # 前期多试错，后期更多利用已经学到的 Q 值。
    epsilon = max(
        epsilon_min,
        epsilon_start
        - (epsilon_start - epsilon_min)
        * episode
        / num_episodes,
    )

    # 一回合最多走 MAX_STEPS 步。
    for _ in range(MAX_STEPS):

        # 根据当前状态的 Q 值选择动作。
        action = choose_action(q[state], epsilon)

        # 与环境交互，获得一次经验：
        #
        # (state, action, reward, next_state, terminated)
        next_state, reward, terminated = step(state, action)

        # 如果已经到达终点或掉进悬崖，
        # 后面就不存在未来奖励，因此 future 必须是 0。
        #
        # 如果没有结束，就参考下一状态中价值最大的动作。
        future = (
            0.0
            if terminated
            else max(q[next_state])
        )

        # Q-learning 的 TD target：
        #
        # target = 当前奖励
        #        + 折扣后的下一状态最大 Q 值
        target = reward + gamma * future

        # Q-learning 更新：
        #
        # Q_new = Q_old
        #       + alpha * (target - Q_old)
        q[state][action] += alpha * (
            target - q[state][action]
        )

        # 状态向前推进。
        state = next_state

        # 如果本回合已经结束，就开始下一回合。
        if terminated:
            break

# %% [markdown]
# ## 6. 把 Q 表读成策略图
#
# 对每个格子取 $\arg\max_a Q(s,a)$，得到"如果站在这里该往哪走"。`S`、`G`、`X` 分别标出起点、终点和悬崖——这三种格子不显示箭头：终点和悬崖没有"下一步"，起点只用来标记出发位置。
#
# 要提醒的是：**策略图只是每个格子各自的推荐动作，不是一条实际轨迹。** 一个格子的箭头指向哪里，与智能体这次是否真的会走到那个格子无关。

# %%
# =========================
# 6. 把 Q 表转换成策略图
# =========================

def render_policy(q):
    """
    对每个状态选择 Q 值最大的动作，
    将学到的策略打印成网格图。
    """

    for row in range(ROWS):
        symbols = []

        for col in range(COLS):
            cell = (row, col)
            state = state_id(cell)

            if cell == START:
                symbols.append("S")
            elif cell == GOAL:
                symbols.append("G")
            elif cell in CLIFF:
                symbols.append("X")
            else:
                # 测试阶段不再探索，只选择 Q 值最大的动作。
                best_action = max(
                    range(N_ACTIONS),
                    key=lambda action: q[state][action],
                )
                symbols.append(ARROWS[best_action])

        print(" ".join(symbols))

# %% [markdown]
# ## 7. 走一条真实轨迹：确认智能体真的到达终点
#
# 策略图不能回答"这一局到底走没走到终点"。下面按贪心规则逐步推进环境，把每一步的坐标、动作、奖励、累计回报和 `terminated` 全部打印出来。
#
# 这条轨迹也顺带检查两件事：
#
# - 智能体是否**真的踩到终点格**，而不是在附近绕圈直到撞上 `MAX_STEPS`；
# - 走完之后是否立即停止，而不是在终止后继续读取"下一格价值"。

# %%
# =========================
# 7. 打印一条完整的贪心轨迹
# =========================

def run_greedy_trajectory(q, verbose=True):
    """
    用贪心策略走一局，返回完整轨迹。

    每个元素记录：
        step       第几步
        cell       当前格子
        action     选中的动作编号
        next_cell  走到的格子
        reward     这一步的奖励
        terminated 是否结束
        total      累计回报
    """
    trajectory = []
    state = state_id(START)
    total_return = 0.0

    for step_index in range(1, MAX_STEPS + 1):
        action = max(
            range(N_ACTIONS),
            key=lambda action: q[state][action],
        )
        next_state, reward, terminated = step(state, action)
        total_return += reward

        trajectory.append({
            "step": step_index,
            "cell": divmod(state, COLS),
            "action": action,
            "next_cell": divmod(next_state, COLS),
            "reward": reward,
            "terminated": terminated,
            "total": total_return,
        })

        state = next_state
        if terminated:
            break

    if verbose:
        for row in trajectory:
            print(f"第 {row['step']:2d} 步：{row['cell']} --{ARROWS[row['action']]}--> "
                  f"{row['next_cell']}  reward={row['reward']:+.2f}  "
                  f"累计={row['total']:+.2f}  terminated={row['terminated']}")

    return trajectory


trajectory = run_greedy_trajectory(q)

last = trajectory[-1]
reached_goal = last["next_cell"] == GOAL and last["terminated"]
entered_cliff = any(row["next_cell"] in CLIFF for row in trajectory)

print()
print(f"总步数：{len(trajectory)}")
print(f"本局回报：{last['total']:+.3f}")
print(f"最后一步是否踩到终点并结束回合：{reached_goal}")
print(f"轨迹中是否经过悬崖：{entered_cliff}")
print(f"是否因为步数上限而中断：{len(trajectory) >= MAX_STEPS and not last['terminated']}")

# 到达终点就必须结束，不能出现"到了终点还在继续走"的情况。
assert reached_goal, "贪心轨迹没有真正到达终点，请检查训练结果"
assert not entered_cliff, "贪心轨迹经过了悬崖，与预期策略不符"

# %% [markdown]
# 轨迹打印的读法：
#
# - 每一行是**一步真实转移**，不是策略图上某个格子的推荐动作；
# - `累计` 是这一局到目前为止的未折扣奖励之和，与第 1 个 Notebook 里的"回报"是同一个量；
# - 最后一个 `terminated=True` 意味着回合在终点结束。若在这里继续读 $Q(s')$，就是把没有未来当成"未来价值恰好为 0"，两者在数值上巧合地一致，但在语义上是错的。

# %% [markdown]
# ## 8. 训练阶段与测试阶段：动作选择与评估指标
#
# | | 训练阶段 | 测试阶段 |
# |---|---|---|
# | 选动作 | $\varepsilon$-greedy（含随机探索，$\varepsilon$ 从 0.2 衰减到 0.02） | 每步 $\arg\max_a Q(s,a)$ |
# | 是否更新 Q 表 | 更新 | 不更新 |
# | 为什么要这样 | 不探索就访问不到别的格子，学不出绕行路线 | 评估要固定策略，否则重复测量没有可比性 |
#
# 下面这段评估代码只做统计：每局从起点出发、每步取 `argmax`、把奖励累加成回报，最后算成功率和平均回报。**平均回报不参与动作选择，也不更新 Q 表**；它只是对这套已经固定下来的策略做汇总。

# %%
# =========================
# 8. 用贪心策略进行评估
# =========================

def evaluate(q, episodes=100):
    """
    关闭探索，只使用当前 Q 表认为最好的动作进行测试。

    每一步的动作由当前状态的 argmax 决定；
    平均回报只在一局结束后用于汇总多局成绩。
    它不会参与动作选择，也不会更新 Q 表。

    返回：
    success_rate: 到达终点的比例
    average_return: 平均回报
    """

    # 成功局数只用于计算成功率，不参与动作选择或 Q 值更新。
    success = 0

    # returns 保存每一局的完整回报。
    # 例如安全走完 7 步时，回报是 6 * (-0.04) + 1 = 0.76。
    returns = []

    # episodes 表示要重复测试多少局。
    # 当前环境起点固定、转移确定、测试阶段不探索，
    # 因此多局很可能得到同一条轨迹和同一个回报。
    # 保留多局评估，是为了统一接口，也方便以后加入随机性。
    for _ in range(episodes):
        # 每局都从起点重新开始，不会把所有格子遍历一遍。
        state = state_id(START)
        # total_return 只记录当前这一局的累计奖励。
        total_return = 0.0

        for _ in range(MAX_STEPS):

            # 评估阶段不再随机探索。
            # 每一步都对当前状态执行 argmax，
            # 不是对所有动作求平均，也不是提前平均所有路线。
            action = max(
                range(N_ACTIONS),
                key=lambda action: q[state][action],
            )

            state, reward, terminated = step(
                state,
                action,
            )

            # 只累加当前这一局实际走过的奖励。
            total_return += reward

            if terminated:
                if reward == GOAL_REWARD:
                    success += 1
                break

        # 当前这一局结束后，才把成绩放入 returns。
        returns.append(total_return)

    # 成功率回答：多局测试中，有多少局到达终点？
    success_rate = success / episodes
    # 平均回报回答：这套固定策略重复执行时，平均每局拿多少分？
    # 它是评估指标，不会反过来改变上面的动作选择。
    average_return = sum(returns) / len(returns)

    return success_rate, average_return


# =========================
# 9. 打印最终结果
# =========================

print("学到的贪心策略：")
# 这里打印的是“每个状态的推荐动作”，不是一条完整实际轨迹。
render_policy(q)

success_rate, average_return = evaluate(q)

print()
print(f"成功率: {success_rate:.1%}")
# 平均回报只是对测试局成绩的汇总；真正选动作的是上面的 max。
print(f"平均回报: {average_return:.3f}")

# %% [markdown]
# ### 三组结果之间的关系
#
# | 量 | 这次的值 | 怎么来的 |
# |---|---:|---|
# | 策略图 | 第 2 行一路向右、最后一列向下 | 对每个格子取 `argmax` |
# | 单局贪心轨迹 | 7 步到终点，回报 $+0.760$ | 从起点逐步执行 `argmax` |
# | 100 局平均回报 | $0.760$，成功率 $100\%$ | 每局独立走一遍再取平均 |
#
# 三者一致不是巧合：环境和贪心动作都是确定性的，所以每局都走同一条路、拿同一个回报。**多局平均在这个环境里不增加信息**，它存在的意义是把评估接口写成通用的——真实任务里转移可能有随机性、起点可能随机、奖励可能有噪声，那时同一套策略每局的回报会不同，才需要多局平均来估计策略的期望表现。

# %% [markdown]
# ## 为什么测试时要多次取平均？
#
# 在当前这个悬崖网格实验中，测试路径实际上是固定的：
#
# 1. 每一局都从同一个起点 `START` 出发；
# 2. 环境是确定性的，同一个状态执行同一个动作，会得到同一个下一状态和奖励；
# 3. 测试阶段关闭探索，每一步都选择当前 Q 值最大的动作：
#
# $$
# a = \mathop{\mathrm{argmax}}_{a'} Q(s,a')
# $$
#
# 因此，每一局都会走同一条路线，得到同一个回报。例如安全路线的回报都是：
#
# $$
# 6 \times (-0.04) + 1 = 0.76
# $$
#
# 所以在当前环境中：
#
# ```text
# 测试 1 局的回报       = 0.76
# 测试 100 局的平均回报 = 0.76
# ```
#
# **结论：在这个完全确定性的教学环境里，多次取平均是冗余的，测试一局就足够了。**
#
# 保留多局评估，主要是因为强化学习中的真实环境经常带有随机性。例如：
#
# - 环境转移可能是随机的：执行同一个动作，不一定每次都到达同一个状态；
# - 策略可能是随机的：有时选择当前最优动作，有时进行探索；
# - 初始状态可能是随机的：每一局不一定从同一个位置开始；
# - 奖励可能带有噪声：完成相同任务时，每次获得的奖励可能不同。
#
# 这时同一套策略可能产生不同的回报：
#
# ```text
# 第 1 局： 0.76
# 第 2 局： 0.52
# 第 3 局：-1.00
# 第 4 局： 0.76
# ```
#
# 只测试一局，可能刚好遇到一次好运或坏运；多次测试并取平均，可以更稳定地估计这套策略的期望表现：
#
# $$
# J(\pi)=\mathbb{E}[G \mid \pi]
# $$
#
# 这里要区分两个概念：
#
# - **动作选择**：由当前状态下 Q 值最大的动作决定，依赖 `max` 或 `argmax`；
# - **策略评估**：统计这套策略执行多次后的整体表现，可以使用平均回报。
#
# 平均回报不会参与动作选择，也不会更新 Q 表。在本实验中，它只是一个通用的评估接口；在确定性环境中，它与单局回报相同。

# %% [markdown]
# ## 常见误区
#
# 1. **把策略图当成轨迹。** 策略图是 24 个格子各自的建议动作，其中的箭头可能永远不会被走到；要确认"真的到终点"，必须看第 7 节的逐步轨迹。
# 2. **训练时不探索。** 把 $\varepsilon$ 设为 0，智能体只走当前认为最好的路，其他格子的 Q 值长期保持为 0，策略会被一开始的偶然选择锁死。
# 3. **终止后继续 bootstrap。** 到终点或掉崖时 `terminated=True`，`future` 必须取 0；把终止状态的价值也读进来等于给结束后的回合加收益。
# 4. **用平均回报去改动作。** 平均回报只在每局结束后统计，它既不参与 `argmax`，也不进入 TD target。
# 5. **越界动作当成"惩罚"。** 撞墙后停在原地并拿到普通步奖励，本身没有额外惩罚，但会白白消耗一步并扣 $-0.04$。
# 6. **以为不同列的策略箭头一定有路可走。** 若某个格子从没被访问过，它的 Q 值可能全是初始值 0，此时 `max` 按"编号最小"选中动作 0，箭头指向的是"没学过"而不是"学过的最好选择"。
#
# ## 本节边界与自检
#
# 可运行核对的内容：环境转移 7 项检查、$\varepsilon$-greedy 训练、策略图、单局贪心轨迹、多局评估。
#
# 自检题：
#
# 1. 从 $(0,0)$ 向上走一步，`next_state`、`reward`、`terminated` 分别是什么？为什么不是"出界惩罚"？
# 2. 掉进悬崖的那一步，TD target 等于多少？为什么？
# 3. 训练阶段和测试阶段在"选动作"和"改 Q 表"上各有什么差别？
# 4. 为什么这个环境里"测试 1 局"和"测试 100 局取平均"结果相同？什么情况下会不同？
#
# 超出本节的内容：这里的 Q 表能存下 24 个状态，是因为状态少且离散。状态一多，为每个状态单独存一行就不可行了——这是第 2 个 Notebook 的出发点。
