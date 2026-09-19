"""STAGE 向量消融版模型：ACT(CVAE-DETR) 去掉图像分支，仅用 256 维向量输入。

与 STAGE-main 的 DETRVAE 对齐，只改两处：
1. 去掉 ResNet18 图像主干 + input_proj(Conv2d)，transformer 的 memory 只用 [latent, proprio] 两个 token；
2. input_state_dim = 256（ego_state6 + lane_detector40 + navi_info10 + history_info200）。

输出：traj_action [B,num_queries,2]（局部未来路点）+ steer_throttle [B,1,2] + style_value [B,1]。

依赖 vendored 进 baseline/model/stage_vec/detr/ 的 detr 模块，无需外部 STAGE 仓库。
"""
import torch
import torch.nn as nn

from baseline.model.stage_vec.detr.models.transformer import build_transformer
from baseline.model.stage_vec.detr.models.detr_vae import (
    build_encoder,
    reparametrize,
    get_sinusoid_encoding_table,
)

# 向量各分量维度（与 baseline/data_process/featurizers.py 保持一致）
VEC_DIMS = {
    "ego_state": 6,
    "lane_detector": 40,
    "navi_info": 10,
    "history_info": 200,
}
INPUT_STATE_DIM = sum(VEC_DIMS.values())  # 256


def stack_vec(vec, vec_dims):
    """按固定顺序拼接向量分量 -> [B, 256]。vec 为已归一化 dict。"""
    order = ["ego_state", "lane_detector", "navi_info", "history_info"]
    parts = []
    for k in order:
        v = vec[k]
        if v.dim() == 1:
            v = v.unsqueeze(0)
        parts.append(v.reshape(v.shape[0], -1))
    return torch.cat(parts, dim=1)


class DETRVAEVec(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        dim_feedforward=2048,
        nheads=8,
        enc_layers=4,
        dec_layers=1,
        dropout=0.1,
        pre_norm=False,
        num_queries=8,
        latent_dim=32,
        state_dim=2,
        input_state_dim=INPUT_STATE_DIM,
        style_pattern="stage",
    ):
        super().__init__()
        self.num_queries = num_queries
        self.latent_dim = latent_dim
        self.style_pattern = style_pattern

        class _Args:
            pass

        args = _Args()
        args.hidden_dim = hidden_dim
        args.dim_feedforward = dim_feedforward
        args.nheads = nheads
        args.enc_layers = enc_layers
        args.dec_layers = dec_layers
        args.dropout = dropout
        args.pre_norm = pre_norm

        self.transformer = build_transformer(args)
        self.encoder = build_encoder(args)
        self.d_model = hidden_dim

        # 输出头
        self.steer_throttle_head = nn.Linear(hidden_dim, state_dim)
        self.action_head = nn.Linear(hidden_dim, state_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries + 1, hidden_dim)

        # 向量输入投影（替代原 DETRVAE 的 input_proj_robot_state）
        self.input_proj_robot_state = nn.Linear(input_state_dim, hidden_dim)

        # CVAE 编码器参数
        self.cls_embed = nn.Embedding(1, hidden_dim)
        self.encoder_action_proj = nn.Linear(state_dim, hidden_dim)
        self.encoder_joint_proj = nn.Linear(input_state_dim, hidden_dim)

        if style_pattern in ("stage", "bc", "bc+vae", "bc+preference"):
            self.latent_proj = nn.Linear(hidden_dim, latent_dim * 2 + 1)
        elif style_pattern == "classifier":
            self.latent_proj = nn.Linear(hidden_dim, latent_dim * 2 + 3)
        else:
            raise NotImplementedError(style_pattern)

        self.register_buffer(
            "pos_table", get_sinusoid_encoding_table(1 + 1 + 1 + num_queries, hidden_dim)
        )

        # 解码器：latent 与 proprio 两个 token 的位置编码
        self.latent_out_proj = nn.Linear(latent_dim + 1, hidden_dim)
        self.additional_pos_embed = nn.Embedding(2, hidden_dim)

    def forward(self, vec, actions=None, is_pad=None, style_control=None):
        """
        vec: [B, 256]（已归一化、已拼接）
        actions (训练): dict{'steer_throttle':[B,1,2], 'traj_action':[B,num_queries,2]}
        is_pad (训练): [B, num_queries+1] bool
        style_control (推理): [B,1] 或标量
        """
        is_training = actions is not None
        bs = vec.shape[0]
        device = vec.device

        if is_training:
            action_seq = torch.cat([actions["steer_throttle"], actions["traj_action"]], dim=1)  # [B, nq+1, 2]
            action_embed = self.encoder_action_proj(action_seq)  # [B, nq+1, H]
            vec_embed = self.encoder_joint_proj(vec).unsqueeze(1)  # [B,1,H]
            cls_embed = self.cls_embed.weight.unsqueeze(0).repeat(bs, 1, 1)  # [B,1,H]
            encoder_input = torch.cat([cls_embed, vec_embed, action_embed], dim=1)  # [B, nq+2, H]
            encoder_input = encoder_input.permute(1, 0, 2)  # [nq+2, B, H]

            cls_joint_is_pad = torch.full((bs, 2), False, device=device)
            is_pad = torch.cat([cls_joint_is_pad, is_pad], dim=1)  # [B, nq+2]
            pos_embed = self.pos_table[:, : encoder_input.shape[0]].clone().detach()  # [1, nq+2, H]
            pos_embed = pos_embed.permute(1, 0, 2)  # [nq+2, 1, H]

            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0]  # cls token 输出 [B,H]

            latent_info = self.latent_proj(encoder_output)  # [B, 2L+1]
            mu = latent_info[:, : self.latent_dim]
            logvar = latent_info[:, self.latent_dim : -1]
            style_value = latent_info[:, -1:]
            latent_sample = reparametrize(mu, logvar)
            latent_sample = torch.cat([latent_sample, style_value], dim=1)
            latent_input = self.latent_out_proj(latent_sample)  # [B,H]
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], device=device)
            if style_control is None:
                style_value = torch.zeros([bs, 1], device=device)
            else:
                style_value = torch.as_tensor(style_control, dtype=torch.float32, device=device)
                if style_value.dim() == 0:
                    style_value = style_value.view(1, 1)
                elif style_value.dim() == 1:
                    style_value = style_value.view(-1, 1)
            if self.style_pattern in ("bc", "bc+vae", "bc+preference"):
                style_value = torch.zeros_like(style_value)
            latent_sample = torch.cat([latent_sample, style_value], dim=1)
            latent_input = self.latent_out_proj(latent_sample)  # [B,H]

        # 解码：无图像 -> memory = [latent_input, proprio_input]（batch-first [B,2,H]）
        proprio_input = self.input_proj_robot_state(vec)  # [B,H]
        src = torch.stack([latent_input, proprio_input], dim=1)  # [B, 2, H]
        pos = self.additional_pos_embed.weight  # [2, H]
        query_embed = self.query_embed.weight  # [nq+1, H]

        _, hs = self.transformer(src, None, query_embed, pos)
        hs = hs[0]  # [B, nq+1, H]

        traj_hat = self.action_head(hs[:, :-1, :])  # [B, nq, 2]
        steer_throttle_hat = self.steer_throttle_head(hs[:, -1:, :])  # [B, 1, 2]
        is_pad_hat = self.is_pad_head(hs)  # [B, nq+1, 1]
        a_hat = {"steer_throttle": steer_throttle_hat, "traj_action": traj_hat}
        return a_hat, is_pad_hat, [mu, logvar], style_value


def build_vec_model(num_queries=8, hidden_dim=256, dim_feedforward=1024):
    return DETRVAEVec(
        hidden_dim=hidden_dim,
        dim_feedforward=dim_feedforward,
        nheads=8,
        enc_layers=4,
        dec_layers=1,
        num_queries=num_queries,
        latent_dim=32,
        state_dim=2,
    )
