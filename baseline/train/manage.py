"""
训练配置与模型文件管理工具。

功能说明：
1. 将 Hydra 配置展平为命令行友好的 `Namespace`；
2. 统一保存 checkpoint（支持自定义文件名）；
3. 管理最优模型数量，自动清理历史 best 模型。
"""

import os
import io
import torch
import glob
import logging
import argparse
from mmengine import fileio
from omegaconf import OmegaConf


def flatten_config(cfg):
    """递归地将 Hydra Config 展平为单层 Namespace，并处理 ListConfig 转换"""
    flat_dict = {}

    def _recurse(conf, parent_key=''):
        for key, value in conf.items():
            if isinstance(value, dict) or OmegaConf.is_dict(value):
                _recurse(value, parent_key=key)
            else:
                if OmegaConf.is_config(value):
                    value = OmegaConf.to_container(value, resolve=True)
                flat_dict[key] = value

    _recurse(cfg)
    return argparse.Namespace(**flat_dict)


def save_model(model, optimizer, scheduler, save_path, epoch, train_loss, wandb_id, ema, filename=None):
    """通用的模型保存函数"""
    save_dict = {
        'epoch': epoch + 1,
        'model': model.state_dict(),
        'ema_state_dict': ema.state_dict() if ema is not None else None,
        'optimizer': optimizer.state_dict(),
        'schedule': scheduler.state_dict(),
        'loss': train_loss,
        'wandb_id': wandb_id
    }

    os.makedirs(save_path, exist_ok=True)

    with io.BytesIO() as f:
        torch.save(save_dict, f)
        content = f.getvalue()

        if filename is not None:
            fileio.put(content, os.path.join(save_path, filename))
        else:
            default_name = f'model_epoch_{epoch + 1}_trainloss_{train_loss:.4f}.pth'
            fileio.put(content, os.path.join(save_path, default_name))
            fileio.put(content, os.path.join(save_path, "latest.pth"))


def manage_best_models(save_path, max_num=3):
    """
    管理最优模型文件，只保留性能最好的前 max_num 个。
    假设文件名格式为: best_model_brier-epoch=XXX-fde=0.123.pth
    """
    # [Fix] 修改匹配模式以包含 epoch
    # 原来: "best_model_brier-fde=*.pth" -> 匹配不到 "brier-epoch=..."
    # 现在: "best_model_brier*.pth" -> 匹配所有以 best_model_brier 开头的文件
    pattern = os.path.join(save_path, "best_model-epoch*.pth")
    best_files = glob.glob(pattern)

    if len(best_files) <= max_num:
        return

    # 提取文件名中的数值并排序
    # 你的文件名格式: ...fde=0.123.pth
    # split('=')[-1] 会取到 "0.123.pth"
    # [:-4] 会取到 "0.123"
    # 这个逻辑依然健壮，只要 fde=value 在文件名的最后即可
    try:
        best_files.sort(key=lambda x: float(x.split('_')[-1][:-4]))
    except ValueError:
        # 如果解析失败，按修改时间排序作为保底方案
        best_files.sort(key=os.path.getmtime)

    # 删除性能较差（数值较大）的模型
    # 因为 sort 默认是升序 (从小到大)，Brier/FDE 越小越好
    # 所以我们要保留前 max_num 个 (最小的)，删除后面的
    files_to_delete = best_files[max_num:]
    for f in files_to_delete:
        if os.path.exists(f):
            os.remove(f)
            logging.info(f"Removed older best model: {os.path.basename(f)}")
