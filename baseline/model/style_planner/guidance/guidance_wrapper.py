"""
Guidance 包装器模块。

作用：
1. 管理一个或多个 guidance 能量函数；
2. 将模型输出映射到物理空间后计算能量；
3. 返回可用于采样引导的总能量项。

兼容性说明：
- 历史代码中存在 `closed_loop_planner` / `nuplan_baseline` 等不同导入路径；
- 这里保留多路径兼容导入，避免在整理目录时破坏现有脚本可运行性。
"""

import torch


from baseline.model.style_planner.library.sde import VPSDE_linear
from baseline.model.style_planner.guidance.collision import collision_guidance_fn


# 预留变量：历史版本用于多样本并行引导，当前默认保持单样本配置。
N = 1
sde = VPSDE_linear()


class GuidanceWrapper:
    def __init__(self):
        self._guidance_fns = [
            collision_guidance_fn
        ]

    def __call__(self, x_in, t_input, cond, *args, **kwargs):
        """执行 guidance 能量计算并返回总能量。"""
        energy = 0
        
        state_normalizer = kwargs["state_normalizer"]
        observation_normalizer = kwargs["observation_normalizer"]
      
        B, P, _ = x_in.shape
        model = kwargs["model"]
        model_condition = kwargs["model_condition"]
      
        x_fix = model(x_in, t_input, **model_condition).detach() - x_in.detach()
        x_fix = x_fix.reshape(B, P, -1, 4)
        x_fix[:, :, 0] = 0.0
        x_in = x_in + x_fix.reshape(B, P, -1)
      
        # x_in = torch.repeat_interleave(x_in, N, dim=0) # [B * N, P, T, 4]
        # t_input = torch.repeat_interleave(t_input, N, dim=0) # [B * N]        
        # kwargs["inputs"] = {k: torch.repeat_interleave(v, N, dim=0) for k, v in kwargs["inputs"].items()}
      
        # sigma_t = sde.marginal_prob_std(t_input)
        # sigma_t = sigma_t / torch.sqrt(1 + sigma_t ** 2)
        # x_in = torch.cat([x_in[:, :1] + sigma_t[:, None, None] * torch.randn_like(x_in[:, :1]), x_in[:, 1:]], dim=1)
      
        x_in = state_normalizer.inverse(x_in.reshape(B, P, -1, 4))
        kwargs["inputs"] = observation_normalizer.inverse(kwargs["inputs"])
      
        for guidance_fn in self._guidance_fns:
            energy += guidance_fn(x_in, t_input, cond, **kwargs)
        # energy1 = self._guidance_fns[0](x_in, t_input, cond, **kwargs)
        # energy2 = self._guidance_fns[1](x_in, t_input, cond, **kwargs)
        
        # energy = energy1 if energy2 < 1 else energy2
        
        assert not torch.isnan(energy).any()
          
        return energy
