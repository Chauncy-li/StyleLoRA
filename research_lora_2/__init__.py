"""Research LoRA v2: continuous weak-ordering preference learning pipeline.

研究 LoRA 第二版：连续弱排序偏好学习管线。
- 不修改 research_lora 的任何文件（只读复用其数据结构与数据集）；
- 不读取 aggr/norm/cons 分类标签，仅使用物理方向先验 + 场景内百分位；
- 产出弱排序偏好 manifest 与冻结场景特征 h_c，供后续偏好编码器与双向 LoRA 训练。
"""