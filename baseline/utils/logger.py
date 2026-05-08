import os
from typing import Any, Dict

from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

try:
    import swanlab
except ImportError:  # pragma: no cover
    swanlab = None


def _args_to_dict(args: Any) -> Dict[str, Any]:
    if args is None:
        return {}
    if isinstance(args, dict):
        return dict(args)
    if hasattr(args, "__dict__"):
        try:
            return dict(vars(args))
        except Exception:
            return {}
    return {}


def _normalize_backend(name: str) -> str:
    value = str(name or "wandb").strip().lower()
    alias = {
        "none": "disabled",
        "off": "disabled",
        "no": "disabled",
    }
    return alias.get(value, value)


class WandbLogger:
    """兼容历史接口的在线实验记录器：支持 wandb / swanlab / disabled。"""

    def __init__(self, run_name, notes, args, wandb_resume_id, save_path, proj_name='Diffusion-Planner', rank=0):
        self.args = args
        self.writer = None
        self.id = None
        self.wandb_writer = None
        self.swanlab_run = None

        self.use_online = bool(getattr(args, "use_wandb", False))
        self.online_backend = _normalize_backend(getattr(args, "online_logger", "wandb"))

        if rank != 0:
            return

        self.writer = SummaryWriter(log_dir=f'{save_path}/tb')

        if not self.use_online or self.online_backend == "disabled":
            return

        config_payload = _args_to_dict(args)

        if self.online_backend == "wandb":
            if wandb is None:
                raise ImportError("online_logger=wandb but `wandb` is not installed in the current environment")

            os.environ["WANDB_MODE"] = "online"
            init_kwargs = {
                "project": proj_name,
                "name": run_name,
                "notes": notes,
                "resume": "allow",
                "sync_tensorboard": False,
                "dir": f"{save_path}",
            }
            if wandb_resume_id:
                init_kwargs["id"] = wandb_resume_id
            self.wandb_writer = wandb.init(**init_kwargs)
            # 允许恢复训练时配置发生变化，避免 run_name 等字段冲突。
            wandb.config.update(config_payload, allow_val_change=True)
            self.id = self.wandb_writer.id
            return

        if self.online_backend == "swanlab":
            if swanlab is None:
                raise ImportError("online_logger=swanlab but `swanlab` is not installed in the current environment")

            init_kwargs = {
                "project": proj_name,
                "experiment_name": run_name,
                "description": notes,
                "config": config_payload,
                "logdir": f"{save_path}/swanlog",
                "mode": "cloud",
            }
            if wandb_resume_id:
                init_kwargs["id"] = wandb_resume_id
                init_kwargs["resume"] = "allow"
            self.swanlab_run = swanlab.init(**init_kwargs)
            self.id = getattr(self.swanlab_run, "id", None)
            return

        raise ValueError(
            f"Unknown online_logger backend: {self.online_backend}. "
            "Supported values: wandb, swanlab, disabled"
        )

    def log_metrics(self, metrics: dict, step: int):
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)

        if not self.use_online:
            return

        if self.online_backend == "wandb" and self.wandb_writer is not None:
            self.wandb_writer.log(metrics, step=step)
            return

        if self.online_backend == "swanlab" and swanlab is not None:
            if self.swanlab_run is not None and hasattr(self.swanlab_run, "log"):
                self.swanlab_run.log(metrics, step=step)
            else:
                swanlab.log(metrics, step=step)

    def finish(self):
        if self.writer is not None:
            self.writer.close()

        if self.online_backend == "wandb" and self.wandb_writer is not None:
            self.wandb_writer.finish()
            return

        if self.online_backend == "swanlab" and swanlab is not None:
            if self.swanlab_run is not None and hasattr(self.swanlab_run, "finish"):
                self.swanlab_run.finish()
            else:
                swanlab.finish()

