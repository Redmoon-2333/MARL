# -*- coding: utf-8 -*-
"""生成强化学习入门笔记的技术图（全部本地可复现）。

输出：默认在本脚本所在目录的 figures/ 保存 SVG、img/ 保存同名 PNG
运行：python make_figures.py [输出目录]

图中所有公式、数值、箭头与标签均由本脚本直接绘制，不使用生成式图片工具；
数据类图（策略网格、轨迹、回报对比）由下方代码用与 Notebook 相同的随机种子实算得出，
并在绘制前用断言核对，避免图文不一致。
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# ---------------------------------------------------------------- 全局样式
INK = "#1a1a1a"
PAPER = "#FAFAF7"
RED = "#B31B1B"
BLUE = "#2F5D8C"
GREEN = "#2E6B4F"
GREY = "#6B6B6B"
LIGHT = "#EFEFEA"
CLIFF_FILL = "#F0DAD8"

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["svg.fonttype"] = "path"

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
FIG_DIR = OUT / "figures"          # 矢量源（SVG），笔记正文引用这里
IMG_DIR = OUT / "img"              # 位图导出（PNG），用于不支持 SVG 的场合
FIG_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)


def new_axes(width, height):
    """坐标系统一为 x:0-100、y:0-100；figure 的宽高决定单位形状。"""
    fig, ax = plt.subplots(figsize=(width, height))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


# 版面登记：box() 记录“框 + 框内文字”，arrow() 记录箭头标签；save() 前统一检查。
BOXES, LABELS = [], []


def box(ax, x0, y0, x1, y1, text, fc="white", ec=INK, fs=11, lw=1.4):
    patch = ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0.5,rounding_size=1.4",
        linewidth=lw, edgecolor=ec, facecolor=fc, mutation_aspect=1.0))
    label = ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center",
                    fontsize=fs, color=INK, linespacing=1.5)
    BOXES.append((patch, label))


def arrow(ax, p0, p1, text="", color=INK, fs=10.5, rad=0.0, lw=1.6,
          toff=(0, 0), ha="center", va="center"):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=15, linewidth=lw, color=color,
        connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))
    if text:
        mx, my = (p0[0] + p1[0]) / 2 + toff[0], (p0[1] + p1[1]) / 2 + toff[1]
        LABELS.append(ax.text(mx, my, text, ha=ha, va=va, fontsize=fs, color=color,
                              linespacing=1.4,
                              bbox=dict(boxstyle="round,pad=0.2", fc=PAPER, ec="none")))


def elbow(ax, points, color=INK, lw=1.6):
    """折线路径（最后一个点带箭头），用于干净地绕开中间的元素。"""
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=lw,
                solid_capstyle="round", zorder=3)
    ax.add_patch(FancyArrowPatch(
        points[-2], points[-1], arrowstyle="-|>", mutation_scale=15,
        linewidth=lw, color=color, shrinkA=0, shrinkB=1, zorder=4))


def _overlap(a, b):
    return a.x0 < b.x1 and b.x0 < a.x1 and a.y0 < b.y1 and b.y0 < a.y1


def check_layout(fig, name):
    """文字不出框、不出画布、互不重叠；箭头标签不压框。发现问题直接报错。"""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    problems = []
    margin = 4  # 像素：框内文字与边框至少留出的距离
    for patch, label in BOXES:
        pb = patch.get_window_extent(renderer)
        tb = label.get_window_extent(renderer)
        if (tb.x0 < pb.x0 + margin or tb.x1 > pb.x1 - margin
                or tb.y0 < pb.y0 + margin or tb.y1 > pb.y1 - margin):
            problems.append(f"文字超出框：{label.get_text()[:18]!r}")
    texts = [t for t in fig.findobj(matplotlib.text.Text)
             if t.get_visible() and t.get_text().strip()]
    extents = [(t, t.get_window_extent(renderer)) for t in texts]
    for t, tb in extents:
        if tb.x0 < canvas.x0 or tb.x1 > canvas.x1 or tb.y0 < canvas.y0 or tb.y1 > canvas.y1:
            problems.append(f"文字超出画布：{t.get_text()[:18]!r}")
    for i, (a, ab) in enumerate(extents):
        for b, bb in extents[i + 1:]:
            if _overlap(ab, bb):
                problems.append(f"文字重叠：{a.get_text()[:12]!r} / {b.get_text()[:12]!r}")
    for label in LABELS:
        lb = label.get_window_extent(renderer)
        for patch, _ in BOXES:
            if _overlap(lb, patch.get_window_extent(renderer)):
                problems.append(f"箭头标签压框：{label.get_text()[:18]!r}")
    BOXES.clear()
    LABELS.clear()
    if problems:
        raise SystemExit(f"[{name}] 版面检查失败：\n  " + "\n  ".join(problems))


def save(fig, name):
    check_layout(fig, name)
    svg = FIG_DIR / f"{name}.svg"
    png = IMG_DIR / f"{name}.png"
    fig.savefig(svg, format="svg", facecolor=fig.get_facecolor())
    fig.savefig(png, format="png", dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  figures/{svg.name}  {svg.stat().st_size / 1024:6.1f} KB    img/{png.name}")


# ================================================================ 图 1
def fig_rl_loop():
    fig, ax = new_axes(9.6, 4.6)

    ax.text(50, 94, "强化学习的交互闭环", ha="center", va="center",
            fontsize=14, fontweight="bold", color=INK)
    box(ax, 6, 32, 34, 78, "智能体 Agent\n策略 $\\pi(a\\mid s)$\n动作价值 $Q(s,a)$\n"
        "依据当前价值选动作\n并修正价值估计", fc="white", ec=BLUE, fs=11.5)
    box(ax, 66, 32, 94, 78, "环境 Environment\n状态 $s$\n奖励 $r$\n"
        "执行动作、给出奖励\n并转移到新状态", fc="white", ec=GREEN, fs=11.5)

    arrow(ax, (36, 64), (64, 64), "动作 $a_t$", color=RED, fs=12, toff=(0, 5), lw=1.9)
    arrow(ax, (64, 48), (36, 48), "下一状态 $s_{t+1}$\n奖励 $r_{t+1}$", color=BLUE,
          fs=11.5, toff=(0, -8), lw=1.9)

    ax.text(50, 18, "$G_t=r_{t+1}+\\gamma\\,r_{t+2}+\\gamma^2 r_{t+3}+\\cdots$",
            ha="center", va="center", fontsize=12.5, color=INK)
    ax.text(50, 7, "奖励是单步反馈；回报是往后所有奖励的折扣累计",
            ha="center", va="center", fontsize=11.5, color=GREY)
    save(fig, "day1-rl-loop")


# ================================================================ 图 2
def fig_td_target():
    fig, ax = new_axes(11.0, 5.0)

    ax.text(50, 95, "Q-learning 与 SARSA：只差 target 里取哪个后续价值",
            ha="center", va="center", fontsize=14.5, fontweight="bold", color=INK)
    box(ax, 2, 28, 22, 72, "同一条经验\n状态 $s$\n动作 $a$\n奖励 $r=-0.1$\n下一状态 $s'$",
        fc=LIGHT, ec=INK, fs=11)

    box(ax, 30, 53, 98, 87,
        "Q-learning\n"
        "$\\mathrm{target}=r+\\gamma\\,\\max\\,Q(s',a')=0.62$\n"
        "后续价值取最大值 $\\max(0.8,\\,0.2)=0.8$：假设下一步会挑最好的动作\n"
        "$Q(s,a)\\leftarrow 0.4+0.5\\times(0.62-0.4)=0.51$",
        fc="white", ec=RED, fs=11.5)
    box(ax, 30, 12, 98, 46,
        "SARSA\n"
        "$\\mathrm{target}=r+\\gamma\\,Q(s',a')=0.08$\n"
        "后续价值取行为策略实际选中的动作：$a'=1$，$Q(s',a')=0.2$\n"
        "$Q(s,a)\\leftarrow 0.4+0.5\\times(0.08-0.4)=0.24$",
        fc="white", ec=BLUE, fs=11.5)

    arrow(ax, (23, 60), (29, 70), color=GREY, rad=0.15)
    arrow(ax, (23, 40), (29, 30), color=GREY, rad=-0.15)

    ax.text(50, 3.5, "同一份旧估计 $Q(s,a)=0.4$、同一个折扣 $\\gamma=0.9$、同一个学习率 $\\alpha=0.5$",
            ha="center", va="center", fontsize=10.5, color=GREY)
    save(fig, "day1-td-target")


# ================================================================ 图 3
def fig_dqn_update():
    fig, ax = new_axes(11.0, 6.0)

    ax.text(50, 96, "DQN 的一次更新：一条经验怎样改动在线网络",
            ha="center", va="center", fontsize=14.5, fontweight="bold", color=INK)

    box(ax, 8, 83, 92, 90,
        "一条经验（批量 = 1）：$s=[1,0]$，实际执行动作 $a=0$，"
        "$r=-0.04$，下一状态 $s'=[0,1]$", fc=LIGHT, ec=INK, fs=11.5)

    elbow(ax, [(50, 82), (50, 79), (30, 79), (30, 76)], color=INK)
    elbow(ax, [(50, 79), (74, 79), (74, 76)], color=INK)

    box(ax, 10, 55, 50, 75, "online 网络（会被优化器更新）\n"
        "$Q_{\\mathrm{online}}(s)=[0.30,\\,0.70]$\n"
        "$\\mathrm{gather}(a{=}0)\\rightarrow$ 当前估计 $0.30$", fc="white", ec=BLUE, fs=11)
    box(ax, 54, 55, 94, 75, "target 网络（本步不更新）\n"
        "$Q_{\\mathrm{target}}(s')=[0.60,\\,0.20]$\n"
        "$\\max\\rightarrow$ 后续价值 $0.60$", fc="white", ec=GREEN, fs=11)

    arrow(ax, (30, 54), (30, 50), color=BLUE)
    arrow(ax, (74, 54), (74, 50), color=GREEN)

    box(ax, 12, 27, 92, 49,
        "TD target：$y=r+\\gamma\\,\\max\\,Q_{\\mathrm{target}}(s',a')"
        "=-0.04+0.9\\times0.60=0.50$\n"
        "损失：$L=(Q_{\\mathrm{online}}(s,a)-y)^2=(0.30-0.50)^2=0.04$\n"
        "若这条经历是终止转移，则 $y=r$，不再加未来价值",
        fc="white", ec=RED, fs=11.5)

    arrow(ax, (52, 26), (52, 22), color=RED, lw=1.9)

    box(ax, 12, 6, 92, 21,
        "反向传播 + $\\mathrm{step}()$：只更新 online\n"
        "更新后 $Q_{\\mathrm{online}}(s,0)=0.50$；target 必须手动同步才会改变",
        fc="#F7EDEC", ec=RED, fs=11.5)
    save(fig, "day2-dqn-update")


# ================================================================ 图 4
def fig_ac_dataflow():
    fig, ax = new_axes(11.0, 6.4)

    ax.text(50, 96, "Actor–Critic：单步更新里的数据流",
            ha="center", va="center", fontsize=14.5, fontweight="bold", color=INK)

    box(ax, 12, 80, 32, 91, "状态 $s$\n（one-hot）", fc=LIGHT, ec=INK, fs=11)
    box(ax, 40, 80, 60, 91, "共享主体\n$\\mathrm{Linear}+\\tanh$", fc="white", ec=INK, fs=11)
    arrow(ax, (33, 85.5), (39, 85.5), color=INK)

    box(ax, 4, 61, 38, 72, "Actor 头\n$\\mathrm{logits}\\rightarrow\\pi_\\theta(a\\mid s)$",
        fc="white", ec=RED, fs=11.5)
    box(ax, 62, 61, 96, 72, "Critic 头\n$V(s)$", fc="white", ec=GREEN, fs=11.5)
    # 两条分支从共享主体的底边出发，不再穿过框内文字
    arrow(ax, (46, 78.5), (26, 73.5), color=RED)
    arrow(ax, (54, 78.5), (74, 73.5), color=GREEN)

    box(ax, 22, 34, 78, 51,
        "按概率采样动作 $a$ → 环境 → $r,\\,s',\\,d$\n"
        "$y=r+\\gamma(1-d)V(s')$\n"
        "$\\delta=y-V(s)$（advantage）", fc="white", ec=BLUE, fs=11)
    arrow(ax, (24, 59.5), (36, 52.5), "动作 $a$", color=RED, fs=10.5, toff=(-9, 0))
    arrow(ax, (76, 59.5), (64, 52.5), "读取 $V(s)$", color=GREEN, fs=10.5, toff=(10, 0))

    box(ax, 4, 9, 44, 25,
        "Actor loss\n$-\\log\\pi_\\theta(a\\mid s)\\cdot\\delta_{\\mathrm{detach}}$\n"
        "调整本次动作的概率", fc="#F7EDEC", ec=RED, fs=11)
    box(ax, 56, 9, 96, 25,
        "Critic loss\n$(V(s)-y)^2$\n把 $V(s)$ 拉向 $y$",
        fc="#EDF3EF", ec=GREEN, fs=11)
    arrow(ax, (36, 32.5), (26, 26.5), "advantage $\\delta$", color=BLUE, fs=10.5,
          toff=(-12, 0))
    arrow(ax, (64, 32.5), (74, 26.5), "$y$ 与 $V(s)$", color=GREEN, fs=10.5,
          toff=(10, 0))

    ax.text(50, 3, "两条损失相加后反传：每走一步更新一次（单步 Actor–Critic，不是 PPO）",
            ha="center", va="center", fontsize=12, color=INK)
    save(fig, "day3-ac-dataflow")


# ================================================================ 图 5
def fig_two_routes():
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    gamma = 0.9
    labels = ["长路线\n动作0 → 状态1 → 终点", "短路线\n动作1 直接结束"]
    undiscounted = [0.0 + 2.0, 1.0]
    discounted = [0.0 + gamma * 2.0, 1.0]

    x = [0, 1]
    w = 0.32
    b1 = ax.bar([i - w / 2 for i in x], undiscounted, w, label="不折扣实际回报（奖励直接相加）",
                color=BLUE, edgecolor=INK, linewidth=0.9)
    b2 = ax.bar([i + w / 2 for i in x], discounted, w,
                label="各路线折扣回报（$\\gamma=0.9$）",
                color=RED, edgecolor=INK, linewidth=0.9)
    for bars in (b1, b2):
        for rect in bars:
            ax.annotate(f"{rect.get_height():.2f}",
                        xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        xytext=(0, 4), textcoords="offset points", ha="center",
                        fontsize=11.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11.5)
    ax.set_ylim(0, 2.55)
    ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5])
    ax.set_ylabel("回报", fontsize=12)
    ax.set_title("TinyChainEnv：两条路线的回报口径对比", fontsize=13.5, color=INK, pad=12)
    ax.legend(fontsize=10.5, frameon=False, loc="upper right")
    fig.tight_layout()
    ax.tick_params(axis="both", labelsize=10.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GREY)
    ax.grid(axis="y", color="#DDDDD5", linewidth=0.8)
    ax.set_axisbelow(True)
    save(fig, "day3-two-routes-returns")


# ================================================================ 图 6
ROWS, COLS = 4, 6
START, GOAL = (3, 0), (3, 5)
CLIFF = {(3, 1), (3, 2), (3, 3), (3, 4)}
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ARROWS = ["↑", "↓", "←", "→"]
STEP_REWARD, CLIFF_REWARD, GOAL_REWARD = -0.04, -1.0, 1.0


def state_id(cell):
    return cell[0] * COLS + cell[1]


def step(state, action):
    row, col = divmod(state, COLS)
    d_row, d_col = ACTIONS[action]
    next_row = min(max(row + d_row, 0), ROWS - 1)
    next_col = min(max(col + d_col, 0), COLS - 1)
    next_cell = (next_row, next_col)
    if next_cell == GOAL:
        return state_id(next_cell), GOAL_REWARD, True
    if next_cell in CLIFF:
        return state_id(next_cell), CLIFF_REWARD, True
    return state_id(next_cell), STEP_REWARD, False


def train_q():
    """与 Day1/02_cliff_test.ipynb 完全相同的环境、种子与超参数。"""
    rng = random.Random(7)
    q = [[0.0] * len(ACTIONS) for _ in range(ROWS * COLS)]
    alpha, gamma = 0.5, 0.95
    eps_start, eps_min, episodes = 0.2, 0.02, 3000

    def choose(q_row, eps):
        if rng.random() < eps:
            return rng.randrange(len(ACTIONS))
        return max(range(len(ACTIONS)), key=lambda a: q_row[a])

    for episode in range(episodes):
        state = state_id(START)
        eps = max(eps_min, eps_start - (eps_start - eps_min) * episode / episodes)
        for _ in range(100):
            action = choose(q[state], eps)
            next_state, reward, terminated = step(state, action)
            future = 0.0 if terminated else max(q[next_state])
            target = reward + gamma * future
            q[state][action] += alpha * (target - q[state][action])
            state = next_state
            if terminated:
                break
    return q


def greedy_trajectory(q):
    traj, state, total = [], state_id(START), 0.0
    for index in range(1, 101):
        action = max(range(len(ACTIONS)), key=lambda a: q[state][a])
        next_state, reward, terminated = step(state, action)
        total += reward
        traj.append((divmod(state, COLS), action, divmod(next_state, COLS),
                     reward, terminated, total))
        state = next_state
        if terminated:
            break
    return traj


def fig_cliff_policy():
    q = train_q()
    traj = greedy_trajectory(q)
    policy = [[ARROWS[max(range(len(ACTIONS)), key=lambda a: q[state_id((r, c))][a])]
               for c in range(COLS)] for r in range(ROWS)]

    # 与 Notebook 实际输出核对，避免图文不一致。
    expected = [["↓", "↓", "↓", "→", "↓", "↓"],
                ["↓", "→", "↓", "↓", "↓", "↓"],
                ["→", "→", "→", "→", "→", "↓"]]
    assert policy[:3] == expected, f"策略与 Notebook 输出不一致：{policy[:3]}"
    assert len(traj) == 7 and traj[-1][2] == GOAL and traj[-1][4], "贪心轨迹未在 7 步内到达终点"
    assert abs(traj[-1][5] - 0.76) < 1e-9, f"回报应为 0.76，实际 {traj[-1][5]}"

    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    # 格内布局：策略箭头居中；S/G/X 标签放在格子左上角，给轨迹圆点让位
    for r in range(ROWS):
        for c in range(COLS):
            cell = (r, c)
            if cell in CLIFF:
                face, label = CLIFF_FILL, "X"
            elif cell == START:
                face, label = "#E4EEF7", "S"
            elif cell == GOAL:
                face, label = "#E3F0E7", "G"
            else:
                face, label = "white", policy[r][c]
            ax.add_patch(Rectangle((c, ROWS - 1 - r), 1, 1, facecolor=face,
                                   edgecolor="#BFBFB5", linewidth=1.1))
            if cell in (START, GOAL):
                ax.text(c + 0.12, ROWS - 1 - r + 0.86, label, ha="left", va="center",
                        fontsize=13, color=INK, fontweight="bold")
            elif cell in CLIFF:
                ax.text(c + 0.5, ROWS - 1 - r + 0.5, label, ha="center", va="center",
                        fontsize=15, color=RED, fontweight="bold", alpha=0.55)
            else:
                # 轨迹经过的格子：箭头移到格子上半部，不被红线覆盖
                y = ROWS - 1 - r + (0.76 if r == 2 else 0.5)
                ax.text(c + 0.5, y, label, ha="center", va="center",
                        fontsize=15 if r != 2 else 13, color=INK if r != 2 else GREY)

    # 轨迹折线：起点 + 每一步真正走到的格子（最后一点就是终点 G）
    points = [START] + [item[2] for item in traj]
    xs = [c + 0.5 for (r, c) in points]
    ys = [ROWS - 1 - r + 0.5 for (r, c) in points]
    ax.plot(xs, ys, color=RED, linewidth=2.6, solid_capstyle="round", zorder=5, alpha=0.92)
    ax.scatter(xs, ys, s=60, color=RED, zorder=6, edgecolor="white", linewidth=1.4)
    for i, (x, y) in enumerate(zip(xs, ys)):
        if i == 0:
            continue          # 起点不用标号，S 已经标出
        # 步号放在圆点右下方，避开水平轨迹线与竖直段
        ax.annotate(str(i), xy=(x, y), xytext=(-14 if i >= 6 else 12, -14),
                    textcoords="offset points", ha="center", va="center",
                    fontsize=10, color=RED, fontweight="bold", zorder=8)

    ax.set_xlim(-0.15, COLS + 0.15)
    ax.set_ylim(-1.05, ROWS + 0.15)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_title("训练后的贪心策略与实际轨迹（S 起点 · G 终点 · X 悬崖）",
                 fontsize=13.5, color=INK, pad=10)
    ax.text(COLS / 2, -0.50,
            "箭头 = 每个格子 Q 值最大的动作；红色折线 = 从起点出发的真实贪心轨迹，\n"
            "7 步到达终点，回报 $6\\times(-0.04)+1=0.76$",
            ha="center", va="center", fontsize=10.5, color=GREY, linespacing=1.6)
    fig.tight_layout()
    save(fig, "day1-cliff-trajectory")


if __name__ == "__main__":
    print("生成技术图：")
    fig_rl_loop()
    fig_td_target()
    fig_dqn_update()
    fig_ac_dataflow()
    fig_two_routes()
    fig_cliff_policy()
    print("完成，输出目录：", OUT)
