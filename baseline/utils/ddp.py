import os
import torch
import torch.distributed as dist
from torch.distributed import init_process_group
import subprocess


def ddp_setup_universal(verbose=False, args=None):
    """
       初始化分布式数据并行(DDP)环境，支持多种分布式环境配置

       Args:
           verbose (bool): 是否启用详细输出模式
           args: 包含分布式训练参数的对象，需要包含ddp和port属性

       Returns:
           tuple: 包含(rank, gpu, world_size)的元组
                  rank: 当前进程的全局排名
                  gpu: 当前进程使用的GPU编号
                  world_size: 总进程数
       """
    if args.ddp == False:
        print(f"do not use ddp, train on GPU 0")
        return 0, 0, 1

    # 检测是否在标准分布式环境中运行（如torchrun）
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
        os.environ['MASTER_PORT'] = str(getattr(args, 'port', '29529'))
        os.environ["MASTER_ADDR"] = "localhost"
    # 检测是否在SLURM环境中运行
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        gpu = rank % torch.cuda.device_count()
        world_size = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(f'scontrol show hostname {node_list} | head -n1')
        os.environ['MASTER_PORT'] = str(args.port)
        os.environ['MASTER_ADDR'] = addr
    else:
        print("Not using DDP mode")
        return 0, 0, 1

    # 设置环境变量确保一致性
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(gpu)
    os.environ['RANK'] = str(rank)

    torch.cuda.set_device(gpu)
    dist_backend = 'nccl'
    dist_url = "env://"
    print('| distributed init (rank {}): {}, gpu {}'.format(rank, dist_url, gpu), flush=True)
    init_process_group(backend=dist_backend, world_size=world_size, rank=rank)
    torch.distributed.barrier()
    if verbose:
        setup_for_distributed(rank == 0)
    return rank, gpu, world_size


def setup_for_distributed(is_master):
    """
       这个函数在非主进程时禁用打印功能

       Args:
           is_master (bool): 指示当前进程是否为主进程
       """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def get_world_size():
    """
       获取分布式训练中的总进程数

       Returns:
           int: 如果未初始化分布式环境则返回1，否则返回实际的world_size
       """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """
       获取当前进程的全局排名

       Returns:
           int: 如果未初始化分布式环境则返回0，否则返回实际的rank
       """
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_model(model, use_ddp):
    """
       根据是否使用DDP获取模型实例

       Args:
           model: 原始模型对象
           use_ddp (bool): 是否使用分布式数据并行

       Returns:
           object: 如果使用DDP则返回model.module，否则返回原始model
       """
    if use_ddp:
        return model.module
    else:
        return model


def is_dist_avail_and_initialized():
    """
       检查分布式训练是否可用并已初始化

       Returns:
           bool: 如果分布式训练可用且已初始化则返回True，否则返回False
       """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def reduce_and_average_losses(loss_dict, device):
    """
       在所有进程中对损失字典进行归约并计算平均值

       Args:
           loss_dict (dict): 包含损失值的字典
           device: 用于创建张量的设备

       Returns:
           dict: 归约并平均化后的损失字典
       """
    torch.distributed.barrier()
    world_size = dist.get_world_size()
    for key in loss_dict.keys():
        loss_tensor = torch.tensor([loss_dict[key].item()]).to(device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        loss_dict[key] = loss_tensor.item() / world_size
    return loss_dict
