# Generated from the adjacent notebook by scripts/marl.py export.

# %%
import torch
from torch.distributions import Categorical

# 两个可训练的动作得分。
# 用 log([0.8, 0.2]) 初始化，是为了让初始概率恰好为 [0.8, 0.2]。
logits = torch.nn.Parameter(torch.log(torch.tensor([0.8, 0.2])))

# SGD 根据梯度修改 logits；学习率决定每次修改的幅度。
optimizer = torch.optim.SGD([logits], lr=0.1)

# 根据当前得分建立动作分布。
distribution = Categorical(logits=logits)

# 模拟刚才那条经历：实际选中动作1，后续折扣回报为2.8。
action = torch.tensor(1)
G = 2.8

# 只取实际选中动作1的对数概率。
log_prob = distribution.log_prob(action)
loss = -log_prob * G

print("更新前概率：", [round(p, 4) for p in distribution.probs.tolist()])
print("本次损失：", round(loss.item(), 4))

optimizer.zero_grad()  # 清除以前的梯度
loss.backward()       # 计算损失对两个动作得分的梯度

print("两个得分的梯度：", [round(g, 4) for g in logits.grad.tolist()])

optimizer.step()      # 根据梯度修改两个动作得分

new_distribution = Categorical(logits=logits)
print("更新后概率：", [round(p, 4) for p in new_distribution.probs.tolist()])

# %%
import torch

# 每组数字依次表示：动作 0 的得分、动作 1 的得分。
for scores in ([1.0, 0.0], [1.0, 1.0], [2.0, 1.0]):
    logits = torch.tensor(scores)

    # 将这一组得分转换成总和为 1 的动作概率。
    probs = torch.softmax(logits, dim=0)

    # 保留四位小数，方便比较。
    print(scores, "->", [round(p, 4) for p in probs.tolist()])

# %%
# 按实际发生的时间顺序，保存这一局的奖励。
rewards = [0.0, 1.0, 2.0]
gamma = 0.9  # 折扣因子。

G = 0.0       # 终点之后的回报为 0。
returns = []  # 用来保存各个时间步的回报。

# 从最后一个奖励开始，依次处理 2、1、0。
for reward in reversed(rewards):
    # 赋值前：G 是下一步的回报。
    # 赋值后：G 变成当前这一步的回报。
    G = reward + gamma * G
    returns.append(G)

# 刚才存入的顺序是 G2、G1、G0。
# 翻转后，恢复为时间顺序 G0、G1、G2。
returns.reverse()

# 只对显示结果四舍五入，不修改原来的计算数据。
print([round(value, 2) for value in returns])
