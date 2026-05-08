import pandas as pd
import os
import glob
import numpy as np
from typing import Optional, Sequence

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover
    pq = None


def get_metric_category(col_name: str) -> str:
    """
    辅助函数：根据指标名称返回分类标签。
    用于在打印报告时将杂乱的指标归类显示。
    """
    col_name = col_name.lower()
    if 'score' in col_name and ('closed_loop' in col_name or 'weighted' in col_name):
        return '🏆 综合评分 (Final Score)'
    elif 'collision' in col_name:
        return '🔴 安全性 (Safety)'
    elif 'progress' in col_name or 'completion' in col_name:
        return '🟢 进度/完成度 (Progress)'
    elif 'comfort' in col_name or 'jerk' in col_name or 'accel' in col_name or 'rate' in col_name:
        return '🔵 舒适度/动力学 (Comfort/Dynamics)'
    elif 'violation' in col_name or 'limit' in col_name or 'compliance' in col_name:
        return '🟠 规则合规 (Compliance)'
    else:
        return '⭐ 其他指标 (General)'


def read_parquet_frame(path: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_parquet(path)
    except Exception:
        if pq is None:
            return None
        try:
            table = pq.read_table(path, use_pandas_metadata=False)
            return pd.DataFrame(table.to_pydict())
        except Exception:
            return None


def find_metric_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lowered = {str(col).lower(): str(col) for col in columns}

    for candidate in candidates:
        low_candidate = str(candidate).lower()
        for low_col, original_col in lowered.items():
            if low_col == low_candidate or low_col == f"{low_candidate}_score":
                return original_col

    for candidate in candidates:
        low_candidate = str(candidate).lower()
        for low_col, original_col in lowered.items():
            if low_candidate in low_col:
                return original_col
    return None


def calculate_weighted_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    [自定义指标计算]
    仿照 NuPlan Challenge 规则计算单场景的综合得分。
    nuplan 评测基准只能输出多个评测场景的平均分数，没有输出每个场景的数据
    Score = Progress * (No Collision) * (Drivable Area) * (Speed Limit) * (Direction) ...
    """
    # 定义主要成分列名（根据 NuPlan 标准命名）
    # 注意：列名可能会因为 config 不同而略有差异，这里采用模糊匹配或标准名

    # 1. 基础分：进度
    progress_col = find_metric_column(df.columns, ['ego_progress_along_expert_route'])

    # 2. 乘数因子 (Multipliers) - 只要有一项是 0，总分就是 0
    multipliers = []

    # 碰撞 (No Collision Score: 1.0 = 无碰撞, 0.0 = 碰撞)
    col_collision = find_metric_column(
        df.columns,
        ['no_ego_at_fault_collisions', 'no_collision_at_fault_collisions']
    )
    if col_collision: multipliers.append(col_collision)

    # 可行驶区域 (Drivable Area Compliance)
    col_area = find_metric_column(df.columns, ['drivable_area_compliance'])
    if col_area: multipliers.append(col_area)

    # 行驶方向 (Driving Direction Compliance)
    col_dir = find_metric_column(df.columns, ['driving_direction_compliance'])
    if col_dir: multipliers.append(col_dir)

    # 限速 (Speed Limit Compliance) - 注：有些 Challenge 把它作为软约束，有些是硬乘数
    col_speed = find_metric_column(df.columns, ['speed_limit_compliance'])
    if col_speed: multipliers.append(col_speed)

    col_ttc = find_metric_column(df.columns, ['time_to_collision_within_bound'])
    col_comfort = find_metric_column(df.columns, ['ego_is_comfortable'])

    # 开始计算
    if progress_col:
        # 复制进度作为基础分
        df['🏆 ClosedLoop_Score_Est'] = df[progress_col].copy()

        # 依次乘以各个合规性系数
        for m_col in multipliers:
            df['🏆 ClosedLoop_Score_Est'] *= df[m_col]

        print(f"🧮 [Metrics] 已基于 {len(multipliers) + 1} 个指标估算单场景综合得分。")

        if col_ttc or col_comfort:
            df['🏆 ClosedLoop_Score_Est_With_TTC_Comfort'] = df['🏆 ClosedLoop_Score_Est'].copy()
            for m_col in [col_ttc, col_comfort]:
                if m_col:
                    df['🏆 ClosedLoop_Score_Est_With_TTC_Comfort'] *= df[m_col]
    else:
        print("⚠️ [Metrics] 未找到进度指标 (ego_progress)，跳过综合得分计算。")

    return df


def load_aggregator_metric(result_dir: str):
    """
    尝试读取 aggregator_metric 文件夹下的官方总分。
    """
    agg_path = os.path.join(result_dir, "aggregator_metric", "*.parquet")
    agg_files = glob.glob(agg_path)

    if not agg_files:
        return None

    try:
        # 通常只有一个聚合文件
        agg_df = read_parquet_frame(agg_files[0])
        # 提取关键分数，通常在 'metric_score' 列，且 'metric_computator' 标识了 challenge 名字
        return agg_df
    except Exception:
        return None


def load_metrics_dataframe(result_dir: str) -> Optional[pd.DataFrame]:
    """
    从结果目录中加载所有的 parquet 文件并合并成宽表。

    Args:
        result_dir: 包含 simulation 结果的文件夹路径

    Returns:
        pd.DataFrame: 透视后的宽表（一行一个场景），如果失败返回 None
    """
    print(f"📂 [Metrics] 正在扫描目录: {result_dir}")

    # 搜索所有指标文件 (递归搜索 metrics 文件夹)
    search_path = os.path.join(result_dir, "**/metrics/*.parquet")
    all_files = glob.glob(search_path, recursive=True)

    # 过滤掉不需要的统计摘要文件
    metric_files = [f for f in all_files if "summary" not in f and "statistics" not in f]

    if not metric_files:
        print("❌ [Metrics] 错误: 未找到任何 metrics parquet 文件，仿真可能未生成结果。")
        return None

    print(f"📦 [Metrics] 发现 {len(metric_files)} 个指标分片文件，正在读取合并...")

    df_list = []
    for f in metric_files:
        try:
            temp_df = read_parquet_frame(f)
            if temp_df is None:
                continue
            # 确保包含必要的列才进行处理
            if {'scenario_name', 'metric_computator', 'metric_score'}.issubset(temp_df.columns):
                # 提取需要的列：场景名、Log名、指标名、分数
                subset = temp_df[['scenario_name', 'log_name', 'metric_computator', 'metric_score']]
                df_list.append(subset)
        except Exception as e:
            print(f"⚠️ [Metrics] 读取文件失败 {f}: {e}")
            pass

    if not df_list:
        print("❌ [Metrics] 错误: 有文件但无法提取有效数据。")
        return None

    # 合并所有小表
    df_raw = pd.concat(df_list, ignore_index=True)

    # 数据透视 (Pivot): 将 metric_computator 的值转为列名
    # 结果：Index=[scenario_name, log_name], Columns=[collision_score, progress_score...]
    df = df_raw.pivot_table(
        index=['scenario_name', 'log_name'],
        columns='metric_computator',
        values='metric_score'
    ).reset_index()

    # 计算综合得分列
    df = calculate_weighted_score(df)

    print(f"✅ [Metrics] 数据加载完成: 共 {len(df)} 个独立场景")
    return df


def print_statistics_report(df: pd.DataFrame, agg_df: Optional[pd.DataFrame] = None):
    """
    打印分类统计报告 (Mean, Min, Max)
    """
    print("=" * 100)
    print("📊 仿真结果统计报告 (Simulation Statistics)")

    # 如果有官方聚合总分，优先打印
    if agg_df is not None and not agg_df.empty:
        print("-" * 100)
        print("🏆 官方挑战总分 (Official Aggregated Score)")
        print("-" * 100)
        # 通常 agg_df 包含 metric_score 列
        if 'metric_score' in agg_df.columns and 'metric_computator' in agg_df.columns:
            for _, row in agg_df.iterrows():
                name = row['metric_computator']
                score = row['metric_score']
                print(f"🌟 {name:<60} : {score:.4f}")
        elif 'score' in agg_df.columns:
            final_row = None
            for name_col in ['scenario', 'scenario_type']:
                if name_col not in agg_df.columns:
                    continue
                mask = agg_df[name_col].astype(str).str.lower().eq('final_score')
                if bool(mask.any()):
                    final_row = agg_df.loc[mask].iloc[-1]
                    break
            if final_row is None and 'num_scenarios' in agg_df.columns:
                scenario_counts = pd.to_numeric(agg_df['num_scenarios'], errors='coerce')
                if bool(scenario_counts.notna().any()):
                    final_row = agg_df.loc[scenario_counts.idxmax()]

            if final_row is not None:
                print(f"🌟 {'final_score':<60} : {float(final_row['score']):.4f}")
                for col in [
                    'no_ego_at_fault_collisions',
                    'time_to_collision_within_bound',
                    'ego_is_comfortable',
                    'drivable_area_compliance',
                    'driving_direction_compliance',
                    'speed_limit_compliance',
                    'ego_progress_along_expert_route',
                ]:
                    if col in agg_df.columns and pd.notna(final_row.get(col)):
                        print(f"   {col:<58} : {float(final_row[col]):.4f}")
            else:
                print(agg_df.head())
        else:
            print(agg_df.head())  # 格式不确定时直接打印前几行

    # 筛选数值类型的列，排除 ID 和 时间戳
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    valid_cols = [c for c in numeric_cols if 'timestamp' not in c and 'id' not in c]

    # 按类别分组指标
    metrics_by_cat = {}
    for col in valid_cols:
        cat = get_metric_category(col)
        if cat not in metrics_by_cat: metrics_by_cat[cat] = []
        metrics_by_cat[cat].append(col)

    # 强制让“综合评分”排在第一个打印
    sorted_cats = sorted(metrics_by_cat.keys(), key=lambda x: (0 if '综合' in x else 1, x))

    for cat in sorted_cats:
        cols = metrics_by_cat[cat]
        if not cols: continue

        print(f"\n{cat}")
        print("-" * 100)
        print(f"{'指标名称 (Metric Name)':<60} | {'Mean':<7} | {'Min':<7} | {'Max':<7} | {'Std':<7}")
        print("-" * 100)
        for col in sorted(cols):
            mean_val = df[col].mean()
            min_val = df[col].min()
            max_val = df[col].max()
            std_val = df[col].std()

            # 高亮显示综合得分
            prefix = "👉 " if 'ClosedLoop_Score' in col else ""
            print(f"{prefix}{col:<60} | {mean_val:.4f}  | {min_val:.4f}  | {max_val:.4f}  | {std_val:.4f}")


def check_failure_cases(df: pd.DataFrame):
    """
    挖掘并打印失败、低分案例 (碰撞或卡死)
    """
    print("=" * 100)
    print("🔍 [DEBUG] 重点失败场景挖掘 (Failure Cases):")

    valid_cols = df.columns.tolist()
    has_failure = False

    # 检查碰撞逻辑: NuPlan 新版本常用 no_ego_at_fault_collisions。
    target_col = find_metric_column(valid_cols, ['no_ego_at_fault_collisions', 'no_collision_at_fault_collisions'])
    if target_col:
        crashes = df[df[target_col] < 1.0]

        if not crashes.empty:
            has_failure = True
            print(f"\n💥 [CRASH] 发现 {len(crashes)} 个场景发生碰撞 (Score < 1.0):")
            # 打印时附带 Log Name 方便定位
            print(crashes[['scenario_name', target_col]].to_string(index=False))
        else:
            print("\n✨ [Safe] 本次测试无碰撞发生 (Clean Run)!")

    # 检查 TTC。它通常不是官方硬碰撞列，但对 closed-loop 安全非常敏感。
    target_col = find_metric_column(valid_cols, ['time_to_collision_within_bound'])
    if target_col:
        ttc_failures = df[df[target_col] < 1.0]
        if not ttc_failures.empty:
            has_failure = True
            print(f"\n⏱️ [TTC] 发现 {len(ttc_failures)} 个场景 TTC 不满足边界 (Score < 1.0):")
            print(ttc_failures[['scenario_name', target_col]].to_string(index=False))

    # 检查进度逻辑: 寻找 'ego_progress' 相关指标且分数 < 0.9 (90%)
    target_col = find_metric_column(valid_cols, ['ego_progress_along_expert_route'])
    if target_col:
        stuck = df[df[target_col] < 0.2]  # 进度小于 20% 视为严重卡死
        if not stuck.empty:
            has_failure = True
            print(f"\n🐢 [STUCK] 严重卡死/未启动场景 (Progress < 20%):")
            print(stuck[['scenario_name', target_col]].to_string(index=False))

    # 综合得分过低
    score_col = '🏆 ClosedLoop_Score_Est'
    if score_col in df.columns:
        # 找出分数低于 0.5 但又不是因为碰撞（碰撞通常分数为0）的场景
        # 这通常意味着违规（逆行、出界）严重
        low_score = df[
            (df[score_col] < 0.7) &
            (df[score_col] > 0.0)
            ]
        if not low_score.empty:
            has_failure = True
            print(f"\n⚠️ [POOR PERFORMANCE] 分数较低的场景 (Score < 0.7, Non-Crash):")
            # 尝试打印主要原因
            cols_to_show = ['scenario_name', score_col]
            # 加上一些违规指标
            violation_cols = [c for c in valid_cols if 'compliance' in c or 'violation' in c]
            cols_to_show.extend(violation_cols[:2])  # 只显示前两个违规指标防止太宽
            print(low_score[cols_to_show].to_string(index=False))

    if has_failure:
        print("\n💡 提示: 请复制 scenario_name 到 NuBoard 搜索回放。")
    else:
        print("\n✅ 完美！本次测试未发现碰撞、卡死或严重低分场景。")

    print("=" * 90)


def evaluate_simulation_results(result_dir: str):
    """
    对外暴露的主函数

    Args:
        result_dir: 结果文件夹路径
    """
    # 加载官方聚合总分 (如果有)
    agg_df = load_aggregator_metric(result_dir)

    # 加载详细数据并计算单场景得分
    df = load_metrics_dataframe(result_dir)

    if df is not None:
        # 打印综合报告
        print_statistics_report(df, agg_df)

        # 挖掘失败案例
        check_failure_cases(df)
