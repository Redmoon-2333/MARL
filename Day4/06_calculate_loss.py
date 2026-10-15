# Generated from the adjacent notebook by scripts/marl.py export.

# %%
import torch

# 教学示例中的三个时间步的 log probability。
log_probs = torch.tensor([-0.7, -0.4, -0.2])

# 与三个时间步一一对应的 reward-to-go。
returns = torch.tensor([2.52, 2.8, 2.0])

# 每一步单独计算：
# log_prob_0 配 G0
# log_prob_1 配 G1
# log_prob_2 配 G2
loss_terms = -log_probs * returns

# 把整条轨迹中所有时间步的损失相加。
policy_loss = loss_terms.sum()

print("每一步的损失项：", loss_terms.tolist())
print("整条轨迹的 policy_loss：", policy_loss.item())
