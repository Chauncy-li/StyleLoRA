# StyleLoRA

StyleLoRA 是面向自动驾驶运动规划的连续个性化方法。它以冻结的 NuPlan Diffusion Planner 为基础，分别学习保守与激进两个轻量适配方向，并用连续变量 `rho ∈ [-1, 1]` 调节行为：负值表示更保守，正值表示更激进，`rho=0` 保持原始规划器。场景条件路由器和顺序响应约束用于让请求强度在不同交通上下文中转化为可观测的轨迹变化。

当前主要模型是从 V4 继续微调的 V5 longitudinal-response 版本。其开环评估关注 route progress、速度、加减速度、jerk、跟车间距/时距、碰撞代理及风格响应；ADE/FDE 仅作为轨迹诊断，不作为主要风格效果指标。闭环使用 NuPlan 仿真验证，安全相关轨迹修复是可选推理功能，不改变已训练模型。

## 代码结构

```text
baseline/       冻结的 Diffusion Planner、数据处理和仿真基线
nuplan-devkit/  NuPlan 场景与仿真框架
stylelora/      偏好数据、编码器、LoRA 训练、开环/闭环评估和反馈接口
results/        本地实验结果目录，不纳入 Git
```

`stylelora/README.md` 提供较完整的数据准备、训练和评测步骤。只做代码检查时，可从仓库根目录运行：

```bash
python -m compileall -q baseline stylelora
python -m pytest stylelora/tests -q
```

## 复现当前 V5 主线

仓库不包含 NuPlan 数据集、训练缓存、预训练权重或实验结果。请先准备这些外部文件，并检查 `stylelora/scripts/run_longitudinal_response_v5.sh` 中默认路径，或通过环境变量覆盖。

服务器上，该脚本依次运行 V5 微调、九个 `rho` 值的完整开环评测，以及十张 3×3 轨迹图；默认输出写入 `CAST_EAAI_PAPER_RESULTS3`：

```bash
conda activate testlocal39
cd /path/to/Nuplan-Diffusion-Baseline
bash stylelora/scripts/run_longitudinal_response_v5.sh
```

脚本需要先前准备好的 V4 adapters/router、baseline checkpoint、偏好编码器、manifest、cache 和场景特征文件。它不适用于从零开始运行整个数据构建流程。

闭环脚本按三个 shard 分别运行；示例（每个 shard 的 `rho` 由调用者指定）：

```bash
bash stylelora/scripts/run_collision_drivable_v5_closed_loop.sh negative "-1,-0.75,-0.5"
bash stylelora/scripts/run_collision_drivable_v5_closed_loop.sh center "-0.25,0,0.25"
bash stylelora/scripts/run_collision_drivable_v5_closed_loop.sh positive "0.5,0.75,1"
```

闭环数据和地图路径可分别用 `NUPLAN_DATA_ROOT`、`NUPLAN_MAPS_ROOT` 指定；结果默认写入 `CAST_EAAI_PAPER_RESULTS3/CLOSED_LOOP`。这些脚本会调用 NuPlan 仿真，运行前请确认外部数据、地图、tokens 和 V5 checkpoint 路径有效。

## 运行时反馈接口

`LoRADiffusionPlanner` 提供可选的相对反馈更新，不调用时原有固定 `rho` 行为不变：

```python
planner.apply_user_feedback("more_aggressive")  # rho 增加默认步长 0.25
planner.apply_user_feedback("more_conservative")
planner.apply_user_feedback("keep")
planner.current_preference_state()
```

`rho` 始终限制在 `[-1, 1]`；该接口是确定性的状态更新，不是从自然语言或用户历史中训练出来的反馈模型。

## Git 中排除的内容

NuPlan 数据、模型权重、实验输出和本地开发配置均不应提交。`.gitignore` 已排除 `results/`、缓存与临时目录、`.agents/`、`.idea/` 及本地历史记录文件。
