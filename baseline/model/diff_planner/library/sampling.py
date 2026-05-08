"""
扩散采样入口函数模块。

该文件封装 DPM-Solver 调用细节，提供统一 `dpm_sampler` 接口：
- 输入模型、初始噪声和条件；
- 输出去噪后的轨迹样本。
"""

from typing import Dict
import torch
import baseline.model.diff_planner.library.dpm_solver_pytorch as dpm


def dpm_sampler(
        model: torch.nn.Module,
        x_T,
        other_model_params: Dict = {},
        diffusion_steps=10,

        noise_schedule_params: Dict = {},
        model_wrapper_params: Dict = {},
        dpm_solver_params: Dict = {},
        sample_params: Dict = {}
):
    with torch.no_grad():
        noise_schedule = dpm.NoiseScheduleVP(
            schedule='linear',
            **noise_schedule_params
        )

        model_fn = dpm.model_wrapper(
            model,  # use your noise prediction model here
            noise_schedule,
            model_type=model.model_type,  # or "x_start" or "v" or "score"
            model_kwargs=other_model_params,
            **model_wrapper_params
        )

        dpm_solver = dpm.DPM_Solver(
            model_fn, noise_schedule, algorithm_type="dpmsolver++", **dpm_solver_params)  # w.o. dynamic thresholding

        # Steps in [10, 20] can generate quite good samples.
        # And steps = 20 can almost converge.
        sample_dpm = dpm_solver.sample(
            x_T,
            steps=diffusion_steps,
            order=2,
            skip_type="logSNR",
            method="multistep",
            denoise_to_zero=True,
            **sample_params
        )

    return sample_dpm
