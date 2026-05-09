import torch
import torch.nn.functional as F


def nll_loss_gmm_direct(pred_scores, pred_trajs, gt_trajs, use_square_gmm=False,
                        log_std_range=(-1.609, 5.0), rho_limit=0.5):
    """
    GMM Loss 核心实现 (移植自 wayformer.py)
    gt_trajs: [B, T, 3] -> (x, y, mask)
    """
    # 检查预测维度: 5 (x, y, log_sig_x, log_sig_y, rho)
    if use_square_gmm:
        assert pred_trajs.shape[-1] == 3
    else:
        assert pred_trajs.shape[-1] == 5

    batch_size = pred_trajs.shape[0]
    gt_valid_mask = gt_trajs[..., -1]  # [B, T]

    # 1. 找到距离最近的 Mode (Winner-Takes-All)
    # pred: [B, K, T, 2], gt: [B, 1, T, 2]
    distance = (pred_trajs[:, :, :, 0:2] - gt_trajs[:, None, :, :2]).norm(dim=-1)
    # 只考虑有效帧的距离
    distance = (distance * gt_valid_mask[:, None, :]).sum(dim=-1)
    nearest_mode_idxs = distance.argmin(dim=-1)  # [B]

    nearest_mode_bs_idxs = torch.arange(batch_size, device=pred_trajs.device)

    # 2. 取出最近 Mode 的预测值
    nearest_trajs = pred_trajs[nearest_mode_bs_idxs, nearest_mode_idxs]  # [B, T, 5]

    # 3. 计算残差
    res_trajs = gt_trajs[..., :2] - nearest_trajs[:, :, 0:2]
    dx = res_trajs[:, :, 0]
    dy = res_trajs[:, :, 1]

    # 4. 提取 GMM 参数
    if use_square_gmm:
        log_std1 = log_std2 = torch.clamp(nearest_trajs[:, :, 2], min=log_std_range[0], max=log_std_range[1])
        std1 = std2 = torch.exp(log_std1)
        rho = torch.zeros_like(log_std1)
    else:
        log_std1 = torch.clamp(nearest_trajs[:, :, 2], min=log_std_range[0], max=log_std_range[1])
        log_std2 = torch.clamp(nearest_trajs[:, :, 3], min=log_std_range[0], max=log_std_range[1])
        std1 = torch.exp(log_std1)
        std2 = torch.exp(log_std2)
        rho = torch.clamp(nearest_trajs[:, :, 4], min=-rho_limit, max=rho_limit)

    # 5. 计算 Regression Loss (NLL)
    # NLL = log(std1) + log(std2) + 0.5*log(1-rho^2) + (mahalanobis_distance)
    reg_gmm_log_coefficient = log_std1 + log_std2 + 0.5 * torch.log(1 - rho ** 2)

    # 避免除以零的保护
    reg_gmm_exp = (0.5 * 1 / (1 - rho ** 2 + 1e-6)) * (
            (dx ** 2) / (std1 ** 2) + (dy ** 2) / (std2 ** 2) - 2 * rho * dx * dy / (std1 * std2)
    )

    reg_loss = ((reg_gmm_log_coefficient + reg_gmm_exp) * gt_valid_mask).sum(dim=-1)

    # 6. 计算 Classification Loss (Cross Entropy)
    # 目标是让 nearest_mode_idxs 对应的概率最大
    loss_cls = F.cross_entropy(input=pred_scores, target=nearest_mode_idxs, reduction='none')

    # 7. 总 Loss
    return (reg_loss + loss_cls).mean()