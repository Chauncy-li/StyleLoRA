"""
训练/验证 epoch 级循环工具。

功能说明：
1. 提供通用 `train_epoch` 与 `validate_epoch` 循环；
2. 处理 batch -> 输入字典、归一化、loss 与指标聚合；
3. 兼容 DDP 下的多卡 loss/metric 汇总。

备注：
- 该文件保留历史训练逻辑，便于与旧实验结果对齐；
- 当前 baseline 主训练入口是 `baseline/train.py`。
"""

from tqdm import tqdm
import torch
from torch import nn

from baseline.common.data_augmentation import StatePerturbation
from baseline.train.train_utils import get_epoch_mean_loss
from baseline.utils import ddp


def train_epoch(data_loader, model, optimizer, args, ema, aug: StatePerturbation = None, epoch=None):
    epoch_loss = []
    # 用于收集每个 batch 的指标
    epoch_metrics = []

    model.train()

    if args.ddp:
        torch.cuda.synchronize()

    desc_text = f"Training Epoch {epoch}" if epoch is not None else "Training"
    with tqdm(data_loader, desc=desc_text, unit="batch") as data_epoch:
        for batch in data_epoch:
            '''
            data structure in batch: Tuple(Tensor) 
            0: ego_current_state,   # Dim = 10  x, y, cos(h), sim(h), vx, vy, ax, ay, steering_angle, yaw_rate
            1: ego_future_gt,   # Dim = 3  x, y, heading
            2: neighbor_agents_past,    # Dim = (num, time_horizon 11) x, y, cos(h), sim(h), vx, vy, width, length, 3D onehot(3)
            3: neighbors_future_gt,     # Dim = (pred_num, pred_horizon, 3)  x, y, heading
            4: lanes,
            5: lanes_speed_limit,
            6: lanes_has_speed_limit,
            7: route_lanes,
            8: route_lanes_speed_limit,
            9: route_lanes_has_speed_limit,
            10: static_objects,
            
            11: ego_agent_past
            12: neighbor_agents_past_mask
            13: neighbor_agents_future_mask
            14: lanes_mask
            15: route lane mask
            '''

            # 1. 构建初始 Inputs 字典 (移动到 GPU)
            inputs = {
                'ego_current_state': batch[0].to(args.device),
                'neighbor_agents_past': batch[2].to(args.device),
                'lanes': batch[4].to(args.device),
                'lanes_speed_limit': batch[5].to(args.device),
                'lanes_has_speed_limit': batch[6].to(args.device),
                'route_lanes': batch[7].to(args.device),
                'route_lanes_speed_limit': batch[8].to(args.device),
                'route_lanes_has_speed_limit': batch[9].to(args.device),
                'static_objects': batch[10].to(args.device),

                'ego_agent_past': batch[11].to(args.device),
                'neighbor_agents_past_mask': batch[12].to(args.device),
                'neighbor_agents_future_mask': batch[13].to(args.device),
                'lanes_mask': batch[14].to(args.device),
                'route_lanes_mask': batch[15].to(args.device)
            }

            # 单独提取 GT (初始为 Raw Meters)
            ego_future_gt = batch[1].to(args.device)
            neighbors_future_gt = batch[3].to(args.device)

            # 2. 数据增强 (Augmentation)
            # 逻辑：aug 会计算一个扰动后的新坐标系，并将 inputs 中的历史轨迹、车道线
            # 以及传入的 GT 全部转换到这个新坐标系下。
            if aug is not None:
                # [关键] 必须接收返回的变换后 GT
                inputs, ego_future_gt, neighbors_future_gt = aug(inputs, ego_future_gt, neighbors_future_gt)

            # 3. 准备 Normalization
            # 先将当前的 GT (可能是原始的，也可能是 Aug 变换后的) 放入 inputs
            # 这样 Normalizer 如果配置了处理 GT，可以正常工作
            inputs['ego_future_gt'] = ego_future_gt
            inputs['neighbors_future_gt'] = neighbors_future_gt

            # 补全 Mask (如果不存在)
            if 'ego_future_mask' not in inputs:
                B, T, _ = ego_future_gt.shape
                inputs['ego_future_mask'] = torch.ones((B, T), device=args.device, dtype=torch.bool)

            # 4. 归一化 (Normalization)
            # 注意：这通常会将 inputs 里的坐标转换为 Sigma 分布或相对值
            inputs = args.observation_normalizer(inputs)

            # 5. [核心修正] 恢复米制 GT 用于 Loss 计算
            # AlphaPlanner 的 Collision Loss 依赖 SafetyShield (阈值如 2.5m)。
            # 如果使用归一化后的 GT 计算距离，会导致 Collision Loss 失效或异常。
            # 因此，我们强制将 米制 GT (Aug 变换后的) 覆盖回 inputs。
            inputs['ego_future_gt'] = ego_future_gt
            inputs['neighbors_future_gt'] = neighbors_future_gt

            # 映射 AlphaPlanner 需要的键名 'neighbor_agents_future'
            inputs['neighbor_agents_future'] = neighbors_future_gt

            # 6. 前向传播与 Loss 计算
            optimizer.zero_grad()

            # (1) Forward
            model_output = model(inputs)

            # (2) Compute Loss
            raw_model = ddp.get_model(model, args.ddp)
            loss_dict = raw_model.compute_loss(model_output, inputs)

            # 7. 计算并收集 Metrics (用于日志显示)
            if hasattr(raw_model, 'compute_metrics'):
                # compute_metrics 内部通常使用 no_grad
                batch_metrics = raw_model.compute_metrics(model_output, inputs)
                metrics_cpu = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in batch_metrics.items()}
                epoch_metrics.append(metrics_cpu)

            # 8. 反向传播
            total_loss = loss_dict['loss']
            total_loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            # EMA 更新
            ema.update(model)

            if args.ddp:
                torch.cuda.synchronize()

            # 9. 记录日志
            loss_val = total_loss.item()
            data_epoch.set_postfix(loss='{:.4f}'.format(loss_val))

            loss_cpu = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
            epoch_loss.append(loss_cpu)

    # 10. Epoch 结束统计
    epoch_mean_loss = get_epoch_mean_loss(epoch_loss)

    # 汇总 Metrics 并合并到 Loss 字典
    if len(epoch_metrics) > 0:
        epoch_mean_metrics = get_epoch_mean_loss(epoch_metrics)
        if args.ddp:
            epoch_mean_metrics = ddp.reduce_and_average_losses(epoch_mean_metrics, torch.device(args.device))
        epoch_mean_loss.update(epoch_mean_metrics)

    if args.ddp:
        epoch_mean_loss = ddp.reduce_and_average_losses(epoch_mean_loss, torch.device(args.device))

    if ddp.get_rank() == 0:
        # 打印关键指标
        msg = f"epoch train loss: {epoch_mean_loss['loss']:.4f}"
        if 'brier_fde' in epoch_mean_loss:
            msg += f", brier_fde: {epoch_mean_loss['brier_fde']:.2f}"
        if 'loss_collision' in epoch_mean_loss:
            msg += f", coll_loss: {epoch_mean_loss['loss_collision']:.4f}"
        print(msg + "\n")

    return epoch_mean_loss, epoch_mean_loss['loss']


@torch.no_grad()
def validate_epoch(val_loader, model, cfg, args):
    """
    验证循环逻辑
    重点：确保 Metric 计算使用原始物理单位 (Meters)，且补全 AlphaPlanner 所需的所有 Keys
    """
    model.eval()
    device = cfg.training.device
    model_module = model.module if cfg.distributed.ddp else model

    metric_sums = {'minADE': 0.0, 'minFDE': 0.0, 'miss_rate': 0.0, 'brier_fde': 0.0}
    total_samples = 0

    for batch in val_loader:
        # 1. 移动数据到设备
        batch = [t.to(device) for t in batch]

        # 提取 Raw GT (Meters)
        raw_ego_future_gt = batch[1]
        raw_neighbors_future_gt = batch[3]

        # 2. 构建输入字典 (需与 Dataset 输出完全对齐)
        inputs = {
            'ego_current_state': batch[0],
            'ego_future_gt': raw_ego_future_gt,  # 初始放入 Raw
            'neighbor_agents_past': batch[2],
            'neighbors_future_gt': raw_neighbors_future_gt,
            'lanes': batch[4],
            'lanes_speed_limit': batch[5],
            'lanes_has_speed_limit': batch[6],
            'route_lanes': batch[7],
            'route_lanes_speed_limit': batch[8],
            'route_lanes_has_speed_limit': batch[9],
            'static_objects': batch[10],

            # 补全 AlphaPlanner/Encoder 所需的额外 Keys
            'ego_agent_past': batch[11],
            'neighbor_agents_past_mask': batch[12],
            'neighbor_agents_future_mask': batch[13],
            'lanes_mask': batch[14],
            'route_lanes_mask': batch[15],

            # 映射 AlphaPlanner 需要的 Key 名称
            'neighbor_agents_future': raw_neighbors_future_gt,
            'map_polylines': batch[4],  # 兼容部分代码可能用旧名字
        }

        # 补全 Ego Mask
        if 'ego_future_mask' not in inputs:
            B, T, _ = inputs['ego_future_gt'].shape
            inputs['ego_future_mask'] = torch.ones(B, T, device=device)

        # 3. 归一化 (Normalization)
        # args.observation_normalizer 会就地修改 inputs 中的 tensor (Meter -> Sigma)
        inputs = args.observation_normalizer(inputs)

        # 4. [关键] 恢复 Raw GT 用于 Metrics 计算
        # Metrics (ADE/FDE/Brier) 需要米制单位
        inputs['ego_future_gt'] = raw_ego_future_gt
        inputs['neighbors_future_gt'] = raw_neighbors_future_gt
        inputs['neighbor_agents_future'] = raw_neighbors_future_gt

        # 5. 前向传播
        model_output = model(inputs)

        # 6. 指标计算
        if hasattr(model_module, 'compute_metrics'):
            # compute_metrics 内部会使用 inputs['ego_future_gt'] (现在是 Meters)
            metrics = model_module.compute_metrics(model_output, inputs)

            batch_size = inputs['ego_future_gt'].shape[0]
            for k, v in metrics.items():
                if k in metric_sums:
                    metric_sums[k] += v.item() * batch_size
            total_samples += batch_size

    # DDP 环境下的多卡汇总
    if cfg.distributed.ddp:
        stats_tensor = torch.tensor([
            metric_sums['minADE'], metric_sums['minFDE'],
            metric_sums['miss_rate'], metric_sums['brier_fde'],
            total_samples
        ], device=device)
        torch.distributed.all_reduce(stats_tensor)

        metric_sums['minADE'], metric_sums['minFDE'], \
            metric_sums['miss_rate'], metric_sums['brier_fde'], \
            total_samples = stats_tensor.tolist()

    return {k: v / total_samples if total_samples > 0 else 0.0 for k, v in metric_sums.items()}
