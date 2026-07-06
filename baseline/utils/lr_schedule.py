from torch.optim.lr_scheduler import SequentialLR, LinearLR, MultiplicativeLR


def CosineAnnealingWarmUpRestarts(optimizer, epoch, warm_up_epoch, start_factor=0.1):
    """
    创建一个包含线性预热和固定学习率阶段的组合学习率调度器

    参数:
        optimizer: PyTorch优化器对象
        epoch: 当前训练轮数，必须大于等于预热轮数
        warm_up_epoch: 预热阶段的轮数
        start_factor: 预热阶段的起始学习率因子，默认为0.1

    返回:
        SequentialLR: 组合学习率调度器，先进行线性预热，然后保持固定学习率

    注意:
        当前实现中epoch参数未在函数内部使用，可能存在逻辑问题
    """
    epoch = max(int(epoch), 1)
    warm_up_epoch = max(0, min(int(warm_up_epoch), epoch))

    # Short smoke tests do not have enough epochs to benefit from a warmup stage.
    if warm_up_epoch <= 1:
        return MultiplicativeLR(optimizer, lr_lambda=lambda _: 1.0)

    T_warmup = warm_up_epoch

    # 创建线性预热调度器，从start_factor逐渐增加到1.0
    warmup_scheduler = LinearLR(optimizer, start_factor=start_factor, total_iters=warm_up_epoch - 1)

    # 创建固定学习率调度器，保持学习率不变
    fixed_scheduler = MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 1.0)

    # 组合调度器：先执行预热阶段，然后切换到固定学习率阶段
    scheduler = SequentialLR(optimizer,
                             schedulers=[warmup_scheduler, fixed_scheduler],
                             milestones=[T_warmup])

    return scheduler
