"""AD-MLP（仅自车状态）规划模型。

8 维自车状态 -> 未来 num_poses 个相对位姿 (x, y, heading)。
对齐 navsim `ego_status_mlp_agent`，与 StyleDrive/mini 侧训练口径一致。

with_style=True 时按 StyleDrive `EgoStatusMLPAgent` 的口径拼接风格机制：
input_dims 从 8 变为 11（8 维 ego_status + 3 维风格 one-hot），拼接在调用方
（planner / train 脚本）完成，本模型只负责按 input_dim 建 MLP。
"""
import torch
import torch.nn as nn

# 风格 one-hot 维度（StyleDrive STYLE_MAP = {"A": 0, "N": 1, "C": 2}，激进/正常/保守）
STYLE_DIM = 3


class EgoStatusMLP(nn.Module):
    def __init__(self, hidden_dim: int = 512, num_poses: int = 8, input_dim: int = 8, with_style: bool = False):
        super().__init__()
        if with_style:
            input_dim = input_dim + STYLE_DIM  # 对齐 StyleDrive：ego_status(8) + style_feature(3) = 11
        self.num_poses = num_poses
        self.input_dim = input_dim
        self.with_style = with_style
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_poses * 3),
        )

    def forward(self, x):  # x: [B, input_dim]
        return self.mlp(x).reshape(-1, self.num_poses, 3)
