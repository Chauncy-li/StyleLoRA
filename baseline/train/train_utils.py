"""
训练流程基础工具函数。

功能说明：
1. 提供 JSON/NPZ 读取接口（与历史脚本兼容）；
2. 提供随机种子设置；
3. 提供 epoch loss 聚合；
4. 提供 checkpoint 恢复逻辑。

说明：
- 本文件侧重“训练状态管理”，不包含模型前向与损失计算。
"""

import torch
import random
import numpy as np
from mmengine import fileio
import io
import os
import json


def _ema_state_dict(ema):
    """Support either a raw EMA module or timm's ModelEma wrapper."""
    if ema is None:
        return None
    shadow_model = getattr(ema, "ema", None)
    if shadow_model is not None and hasattr(shadow_model, "state_dict"):
        return shadow_model.state_dict()
    if hasattr(ema, "state_dict"):
        return ema.state_dict()
    raise TypeError(f"Unsupported EMA object type: {type(ema)!r}")


def openjson(path):
    """
    读取JSON文件并返回解析后的字典对象

    Args:
        path (str): JSON文件的路径

    Returns:
        dict: 解析后的JSON数据字典
    """
    value = fileio.get_text(path)
    dict = json.loads(value)
    return dict


def opendata(path):
    """
    读取NPZ格式的数据文件

    Args:
        path (str): NPZ文件的路径

    Returns:
        numpy.lib.npyio.NpzFile: 加载的NPZ数据对象
    """

    npz_bytes = fileio.get(path)
    buff = io.BytesIO(npz_bytes)
    npz_data = np.load(buff)

    return npz_data


def set_seed(CUR_SEED):
    """
    设置随机种子以确保实验的可重现性

    Args:
        CUR_SEED (int): 随机种子值
    """
    random.seed(CUR_SEED)
    np.random.seed(CUR_SEED)
    torch.manual_seed(CUR_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_epoch_mean_loss(epoch_loss):
    """
    计算一个epoch中所有损失的平均值

    Args:
        epoch_loss (list): 包含多个批次损失字典的列表

    Returns:
        dict: 每个损失项的平均值字典
    """
    epoch_mean_loss = {}
    for current_loss in epoch_loss:
        for key, value in current_loss.items():
            if key in epoch_mean_loss:
                epoch_mean_loss[key].append(value if isinstance(value, (int, float)) else value.item())
            else:
                epoch_mean_loss[key] = [value if isinstance(value, (int, float)) else value.item()]

    for key, values in epoch_mean_loss.items():
        epoch_mean_loss[key] = np.mean(np.array(values))

    return epoch_mean_loss


def save_model(model, optimizer, scheduler, save_path, epoch, train_loss, wandb_id, ema):
    """
    保存模型、优化器、调度器等训练状态到指定路径

    Args:
        model: 训练模型
        optimizer: 优化器
        scheduler: 学习率调度器
        save_path (str): 模型保存路径
        epoch (int): 当前训练轮次
        train_loss (float): 训练损失
        wandb_id (str): wandb实验ID
        ema: 指数移动平均模型
    """
    save_model = {'epoch': epoch + 1,
                  'model': model.state_dict(),
                  'ema_state_dict': _ema_state_dict(ema),
                  'optimizer': optimizer.state_dict(),
                  'schedule': scheduler.state_dict(),
                  'loss': train_loss,
                  'wandb_id': wandb_id}

    with io.BytesIO() as f:
        torch.save(save_model, f)
        fileio.put(f.getvalue(), f'{save_path}/model_epoch_{epoch + 1}_trainloss_{train_loss:.4f}.pth')
        fileio.put(f.getvalue(), f"{save_path}/latest.pth")


def resume_model(path: str, model, optimizer, scheduler, ema, device):
    """
    从指定路径加载检查点文件恢复训练状态

    Args:
        path (str): 检查点文件路径
        model: 训练模型
        optimizer: 优化器
        scheduler: 学习率调度器
        ema: 指数移动平均模型
        device: 训练设备

    Returns:
        tuple: 包含恢复后的模型、优化器、调度器、起始轮次、wandb_id和ema的元组
    """
    path = os.path.join(path, 'latest.pth')
    ckpt = fileio.get(path)
    with io.BytesIO(ckpt) as f:
        ckpt = torch.load(f)

    # load model
    try:
        model.load_state_dict(ckpt['model'])
    except:
        model.load_state_dict(ckpt)
    print("Model load done")

    # load optimizer
    try:
        optimizer.load_state_dict(ckpt['optimizer'])
        print("Optimizer load done")
    except:
        print("no pretrained optimizer found")

    # load schedule
    try:
        scheduler.load_state_dict(ckpt['schedule'])
        print("Schedule load done")
    except:
        print("no schedule found,")

    # load step
    try:
        init_epoch = ckpt['epoch']
        print("Step load done")
    except:
        init_epoch = 0

    # Load wandb id
    try:
        wandb_id = ckpt['wandb_id']
        print("wandb id load done")
    except:
        wandb_id = None

    try:
        ema.ema.load_state_dict(ckpt['ema_state_dict'])
        ema.ema.eval()
        for p in ema.ema.parameters():
            p.requires_grad_(False)

        print("ema load done")
    except:
        print('no ema shadow found')

    return model, optimizer, scheduler, init_epoch, wandb_id, ema
