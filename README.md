# StyleLoRA

StyleLoRA 是一个让自动驾驶规划器按用户偏好连续调整驾驶风格的方法。运行时用 `rho` 表示调整强度：`rho < 0` 更保守，`rho > 0` 更激进，`rho = 0` 保持原始规划器。

本文先说明新机器怎么配置路径，再说明数据如何一步步变成可训练模型，以及当前推荐的模型微调和评测流程。完整的数据处理命令在 [stylelora/README.md](stylelora/README.md)。

## 第一次下载仓库：先配置自己的路径

仓库不包含 NuPlan 数据、地图和训练缓存；当前使用的 baseline、V5 LoRA 权重，以及配套的 `args.json` 和 `normalization.json` 已放在根目录 `weights/`。**不要把自己电脑或服务器的数据地址写进公共路径模板，也不要提交个人路径文件。**

先把仓库 clone 或下载到 Ubuntu 本机，并在仓库根目录操作。运行数据和模型脚本还需要能用的 NuPlan、PyTorch/CUDA 环境；服务器当前使用 `mdsn_py39`。`environment.yml` 可作为依赖参考，创建环境时要按本机的 CUDA 和 NuPlan 安装情况检查。

在仓库根目录，把示例配置复制成自己的配置。

Ubuntu 本地电脑和服务器都运行下面这组命令：

```bash
if [ ! -f stylelora/config/paths.local.json ]; then
  cp stylelora/config/paths.example.json stylelora/config/paths.local.json
fi
```

然后在 Ubuntu 上打开 `stylelora/config/paths.local.json`，把地址改成**当前这台机器**上的实际位置。你和协作者各自在自己的 Ubuntu 仓库里做一次；不要把你的本地数据路径写进服务器配置，反过来也一样。服务器模板里的默认地址通常不适用于本地电脑。

常用配置项如下：

| 配置项 | 放什么 | 什么时候要改 |
|---|---|---|
| `repo_root` | 这份仓库在本机的位置 | 仓库不在模板写的服务器目录时 |
| `data_root` | NuPlan 数据目录 | 本机数据集放在其他位置时 |
| `maps_root` | NuPlan 地图目录 | 地图放在其他位置时 |
| `record_root` | 缓存、输入和实验文件的总目录 | 本机记录盘/工作盘不同时 |
| `cache_root` | 已处理的 planner cache | cache 不放在默认目录时 |
| `source_root` | 训练要读取的已有输入和基础模型 | 输入文件另放一处时 |
| `output_root` | 新模型、评测结果、图和日志的保存位置 | 想把本次结果写到其他位置时 |
| `style_record_root` | 通用数据流水线使用的另一处记录目录 | 这部分数据放在别处时 |
| `tokens_file` | 闭环评测使用的固定场景列表 | 使用另一份场景列表时 |
| `gpu` | V5 训练/开环默认使用的 GPU 编号 | 默认 GPU 不是要用的那张卡时 |
| `baseline_checkpoint` | 闭环评测使用的 baseline 权重文件 | 权重不在 `source_root/MODELS/` 时才填写 |
| `v5_high_adapter`、`v5_low_adapter`、`v5_router_checkpoint` | V5 闭环评测使用的三个权重文件 | 权重不在 `output_root/MODELS/LONGITUDINAL_RESPONSE_V5/` 时才填写 |
| `args_file`、`normalization_file` | 闭环评测使用的 planner 参数和归一化文件 | 文件不在 `source_root/INPUTS/` 时才填写 |

上面六个路径默认是 `null`，**不用改时保持 `null` 即可**：两个 V5 闭环脚本会继续按原来的 `source_root` 和 `output_root` 目录查找文件。如果改用仓库里的 `weights/`，就把对应项填成实际完整路径；只覆盖填写了路径的文件，其余仍按原位置读取。旧版 `paths.local.json` 即使没有这些字段，也会回退到原路径。示例：

```json
{
"baseline_checkpoint": "${repo_root}/weights/baseline/baseline_diffplanner.pth",
"v5_high_adapter": "${repo_root}/weights/lora/conditional_high_longitudinal_response_v5.pt",
"v5_low_adapter": "${repo_root}/weights/lora/conditional_low_longitudinal_response_v5.pt",
"v5_router_checkpoint": "${repo_root}/weights/lora/conditional_router_longitudinal_response_v5.pt",
"args_file": "${repo_root}/weights/inputs/args.json",
"normalization_file": "${repo_root}/weights/inputs/normalization.json"
}
```

模板里的 `source_root`、`output_root`、`cache_root` 等路径会根据 `record_root` 自动拼出来。只要这些目录仍在同一个总目录下，通常改 `record_root` 就够了；数据集、地图和仓库位置则按实际情况分别填写。

检查程序读到的地址：

```bash
python stylelora/config/runtime_paths.py --show
```

`paths.local.json` 已被 Git 忽略。每个协作者、每台服务器都保留自己的这份文件；更新代码不会覆盖它。以后数据目录搬家时，改对应的配置项即可，不用逐个改训练脚本。

新脚本也能使用这份配置：在 `paths` 里加一个键，再让脚本通过 `get_path("新键")`（Python）或 `python stylelora/config/runtime_paths.py --get 新键`（shell）读取。**只增加配置项不会自动改变仍把地址写死在代码里的脚本**，新脚本需要使用这个读取方式。

旧的 `CAST_*` 和 `NUPLAN_*` 环境变量仍可临时覆盖配置，例如换一张卡或临时指定输出目录；一般不需要每次运行都设置它们。

## 数据怎么变成模型

完整数据准备不是一个命令，而是按下面顺序完成。每一步的详细参数和命令见 [新人运行指南](stylelora/README.md)。

1. **准备原始数据**：NuPlan 数据库、地图、baseline 的配置和 checkpoint。
2. **切分日志并生成 cache**：先确定 train/validation/test 使用哪些日志，再把原始数据库处理成规划器可读取的 cache。
3. **整理训练样本**：从 cache 中筛选支持的场景，生成场景索引和 source manifest。
4. **生成偏好监督**：从轨迹的速度、加减速、跟车等行为指标构造弱排序偏好；训练集拟合排序尺度，验证集沿用同一尺度。
5. **提取场景特征并训练偏好编码器**：编码器把场景和轨迹信息转成训练适配器会用到的偏好表示；评估后导出 latent bank。
6. **训练驾驶风格适配器**：保留基础规划器，训练保守/激进方向的轻量 LoRA 适配器和路由器。
7. **微调最终模型并验证**：训练最终的纵向响应模型，然后运行九档 `rho` 开环评测和轨迹图。

这些训练步骤需要已经准备好的数据、cache、偏好编码器和 V4 初始化适配器；初始化模型与中间训练产物不都包含在仓库里。若只想验证当前最终模型，可使用下方 `weights/` 中的 baseline 和 V5 权重；若要重新训练，则按完整指南准备每一步需要的输入。

## 当前最终模型权重

当前模型和闭环所需配套参数整理在仓库根目录 `weights/`：

- `baseline/baseline_diffplanner.pth`：基础 DiffPlanner 权重。
- `lora/conditional_high_longitudinal_response_v5.pt`：更激进方向的 LoRA 适配器。
- `lora/conditional_low_longitudinal_response_v5.pt`：更保守方向的 LoRA 适配器。
- `lora/conditional_router_longitudinal_response_v5.pt`：根据场景和强度控制两个方向的路由器。
- `inputs/args.json`、`inputs/normalization.json`：闭环评测使用的规划器参数和归一化统计。

三个 LoRA 文件要配合使用。要从 `weights/` 运行闭环评测，在 `paths.local.json` 中按上面的示例填写六个文件路径；注意在本地运行的时候要注意设置成本地的路径保存地址；默认的 `null` 定位到了远程服务器的地址 `source_root/INPUTS/`、`source_root/MODELS/` 和 `output_root/MODELS/LONGITUDINAL_RESPONSE_V5/` 位置。NuPlan 数据、地图和闭环 tokens 仍需另行准备。

## 再次运行微调和开环评测

一键脚本会重新微调最终模型，并运行开环验证。它读取本机路径配置，要求以下输入已经准备好：

- `cache_root` 指向训练/验证 cache；
- `source_root/INPUTS/` 中有 `args.json`、`normalization.json`、训练/验证 manifest、scene feature、latent bank 及索引文件；
- 输入目录中有 baseline checkpoint 和偏好编码器；
- 训练启动脚本要求的初始化 LoRA 适配器和路由器已经放在它预期的位置。

在服务器激活已有的 NuPlan/PyTorch 环境后，从仓库根目录运行：

```bash
bash stylelora/scripts/run_longitudinal_response_v5.sh
```

脚本会依次做短时 V5 微调、九档 `rho` 的完整开环评测，以及 10 个场景的 3×3 轨迹图。模型、报告、图片和日志写入 `output_root` 下的 `MODELS/`、`OPEN_LOOP/` 和 `LOGS/`。训练/开环使用的 GPU 默认读 `gpu`；也可临时指定，例如：

```bash
CAST_GPU=0 bash stylelora/scripts/run_longitudinal_response_v5.sh
```

这一步不会重新处理原始 NuPlan 数据，也不会重新训练偏好编码器或初始化权重。

## 闭环评测

闭环需要 NuPlan 数据和地图、V5 checkpoint、planner 参数与归一化文件，以及 `tokens_file` 指向的共同场景列表。以上六个模型/参数文件既可用 `paths.local.json` 指向 `weights/`，也可保持 `null` 从原有服务器目录读取。把九个 `rho` 分成三个 shard，可分别在三个终端运行：

```bash
CUDA_VISIBLE_DEVICES=0 bash stylelora/scripts/run_collision_drivable_v5_closed_loop.sh negative "-1,-0.75,-0.5"
CUDA_VISIBLE_DEVICES=1 bash stylelora/scripts/run_collision_drivable_v5_closed_loop.sh center "-0.25,0,0.25"
CUDA_VISIBLE_DEVICES=2 bash stylelora/scripts/run_collision_drivable_v5_closed_loop.sh positive "0.5,0.75,1"
```

结果保存在 `output_root/CLOSED_LOOP/`，日志在对应的 `LOGS/` 目录。若只运行某一个 shard，就只执行对应那一行。

## 哪些脚本使用了路径配置

当前本机路径配置已接入 StyleLoRA 通用数据路径、V5 微调/开环脚本和两个 V5 闭环脚本。历史实验启动脚本尚未统一接入；运行它们前请检查脚本中的路径默认值，或使用它们支持的环境变量覆盖。

## 其他说明

- `rho=0` 用来检查是否保留原始规划器；开环主分析关注路线进度、速度、加减速、jerk、跟车距离/时距和风格响应，ADE/FDE 是轨迹诊断项。
- 运行时反馈接口 `apply_user_feedback(...)` 可以按“更保守/更激进”增减当前 `rho`，但它不是从用户历史中学习反馈的模型。
- 数据、cache、实验输出和个人路径配置不应提交到 Git；当前 `weights/` 是有意纳入仓库的模型与推理参数，`results/` 和 `paths.local.json` 则在 `.gitignore` 中排除。
- 快速检查代码：`python -m compileall -q baseline stylelora`；运行测试：`python -m pytest stylelora/tests -q`。
