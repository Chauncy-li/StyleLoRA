"""步骤 11：运行与偏好空间对齐的 LoRA 开环 rho 扫描。"""

# 开环主实现保留在 evaluate_preference_lora，单独提供这个语义清晰的入口，
# 避免新人误用旧版依赖 aggr/norm/cons 离散标签的评估脚本。
from stylelora.scripts.evaluate_preference_lora import main


if __name__ == "__main__":
    main()

