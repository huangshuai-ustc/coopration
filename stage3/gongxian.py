import pandas as pd
import numpy as np
import ast
from collections import defaultdict

# 定义所有机制列表
mechanisms = [
    "cognitive_offload", "information_processing_aid", "decision_heuristics",
    "bias_mitigation_amplification", "ambiguity_reduction", "situation_awareness",
    "cognitive_load_management", "bounded_rationality", "anchoring_adjustment",
    "error_probability_estimation", "preference_learning", "mind_perception",
    "cognition_emotion_interaction", "human_ai_interaction", "human_machine_teaming",
    "cooperation_competition_dynamics", "delegated_decision_making", "responsibility_allocation",
    "trust_calibration", "trust_dynamics", "trust_miscalibration", "trust_repair_strategies",
    "algorithmic_attitudes", "automation_bias", "uncertainty_communication",
    "perceived_fairness", "value_alignment", "social_presence", "anthropomorphism_effects",
    "levels_of_autonomy", "adaptive_automation", "automation_transparency",
    "error_recovery_support", "interaction_fluency", "coordination_ability",
    "team_shared_awareness", "feedback_loops", "task_crafting", "negative_work_rumination",
    "approach_avoidance_dynamics", "ai_empathy_dynamics", "emotion_recognition",
    "affective_transition", "stress_anxiety_regulation", "motivation_engagement",
    "psychological_safety", "self_efficacy", "agency_restoration", "moral_agency",
    "responsibility_gap", "ethical_dissonance", "ai_threat_perceptions", "psychological_expansion"
]


def build_cooccurrence_matrix(df, mechanisms):
    """
    构建机制共现矩阵，并统计每个机制的出现次数
    """
    n = len(mechanisms)
    mechanism2idx = {mech: idx for idx, mech in enumerate(mechanisms)}
    cooccurrence_matrix = np.zeros((n, n), dtype=int)

    # 统计每个机制单独出现的次数
    mechanism_individual_count = {mech: 0 for mech in mechanisms}

    print("正在构建共现矩阵...")

    for index, row in df.iterrows():
        try:
            if pd.notna(row['classification']) and row['classification'].strip():
                classification_list = ast.literal_eval(row['classification'])
                valid_mechanisms = [mech for mech in classification_list if mech in mechanism2idx]

                if valid_mechanisms:
                    # 更新单个机制出现次数
                    for mech in valid_mechanisms:
                        mechanism_individual_count[mech] += 1

                    # 更新共现矩阵
                    indices = [mechanism2idx[mech] for mech in valid_mechanisms]
                    for i in indices:
                        for j in indices:
                            if i != j:
                                cooccurrence_matrix[i][j] += 1

        except (ValueError, SyntaxError, TypeError) as e:
            print(f"警告: 第 {index + 1} 行数据解析错误: {e}")
            continue

    return cooccurrence_matrix, mechanism2idx, mechanism_individual_count


def calculate_jaccard_similarity(cooccurrence_matrix, mechanism_individual_count, mechanisms):
    """
    正确计算Jaccard相似度矩阵
    """
    n = len(mechanisms)
    jaccard_matrix = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            if i != j:
                # 获取机制A和B的单独出现次数
                count_A = mechanism_individual_count[mechanisms[i]]
                count_B = mechanism_individual_count[mechanisms[j]]
                count_AB = cooccurrence_matrix[i][j]  # 共现次数

                # 计算Jaccard相似度: |A∩B| / |A∪B|
                if count_A + count_B - count_AB > 0:
                    jaccard_matrix[i][j] = count_AB / (count_A + count_B - count_AB)
                else:
                    jaccard_matrix[i][j] = 0.0
            else:
                jaccard_matrix[i][i] = 1.0  # 对角线设为1（自身相似度）

    return jaccard_matrix


def save_cooccurrence_matrix(cooccurrence_matrix, jaccard_matrix, mechanism_individual_count, mechanisms, output_file):
    """
    保存共现矩阵和Jaccard相似度矩阵到Excel文件
    """
    # 创建带有机制名称的DataFrame
    cooccurrence_df = pd.DataFrame(
        cooccurrence_matrix,
        index=mechanisms,
        columns=mechanisms
    )

    jaccard_df = pd.DataFrame(
        jaccard_matrix,
        index=mechanisms,
        columns=mechanisms
    )

    # 创建机制出现次数统计
    count_df = pd.DataFrame({
        '机制': mechanisms,
        '出现次数': [mechanism_individual_count[mech] for mech in mechanisms]
    })

    # 保存到Excel
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # 保存原始共现矩阵
        cooccurrence_df.to_excel(writer, sheet_name='原始共现矩阵')

        # 保存Jaccard相似度矩阵（保留3位小数）
        jaccard_df.round(3).to_excel(writer, sheet_name='Jaccard相似度矩阵')

        # 保存机制出现次数统计
        count_df.to_excel(writer, sheet_name='机制出现次数', index=False)

        # 保存统计信息
        stats_data = {
            '统计指标': [
                '总机制数量', '总文档数量', '总共现对数数量',
                '平均Jaccard相似度', '最大Jaccard相似度', '最小非零Jaccard相似度'
            ],
            '数值': [
                len(mechanisms),
                len(cooccurrence_matrix),
                np.sum(cooccurrence_matrix) // 2,
                np.mean(jaccard_matrix[jaccard_matrix > 0]) if np.any(jaccard_matrix > 0) else 0,
                np.max(jaccard_matrix),
                np.min(jaccard_matrix[jaccard_matrix > 0]) if np.any(jaccard_matrix > 0) else 0
            ]
        }
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_excel(writer, sheet_name='统计信息', index=False)

    print(f"共现矩阵已保存到: {output_file}")


def analyze_results(cooccurrence_matrix, jaccard_matrix, mechanisms, top_n=10):
    """
    分析结果并显示最高共现和最高相似度的机制对
    """
    print("\n=== 分析结果 ===")

    # 分析最高共现的机制对
    print(f"\n最高共现的机制对 (前{top_n}):")
    cooccurrence_pairs = []
    for i in range(len(mechanisms)):
        for j in range(i + 1, len(mechanisms)):
            if cooccurrence_matrix[i][j] > 0:
                cooccurrence_pairs.append((mechanisms[i], mechanisms[j], cooccurrence_matrix[i][j]))

    cooccurrence_pairs.sort(key=lambda x: x[2], reverse=True)
    for i, (mech1, mech2, count) in enumerate(cooccurrence_pairs[:top_n]):
        print(f"{i + 1}. {mech1} & {mech2}: {count}次")

    # 分析最高Jaccard相似度的机制对
    print(f"\n最高Jaccard相似度的机制对 (前{top_n}):")
    similarity_pairs = []
    for i in range(len(mechanisms)):
        for j in range(i + 1, len(mechanisms)):
            if jaccard_matrix[i][j] > 0:
                similarity_pairs.append((mechanisms[i], mechanisms[j], jaccard_matrix[i][j]))

    similarity_pairs.sort(key=lambda x: x[2], reverse=True)
    for i, (mech1, mech2, similarity) in enumerate(similarity_pairs[:top_n]):
        print(f"{i + 1}. {mech1} & {mech2}: {similarity:.3f}")


def process_excel_with_cooccurrence(input_file, output_file):
    """
    处理Excel文件并生成共现矩阵
    """
    try:
        # 读取Excel文件
        print(f"正在读取文件: {input_file}")
        df = pd.read_excel(input_file)

        if 'classification' not in df.columns:
            raise ValueError("Excel文件中必须包含'classification'列")

        # 构建共现矩阵和统计单个机制出现次数
        cooccurrence_matrix, mechanism2idx, mechanism_individual_count = build_cooccurrence_matrix(df, mechanisms)

        # 计算正确的Jaccard相似度矩阵
        jaccard_matrix = calculate_jaccard_similarity(cooccurrence_matrix, mechanism_individual_count, mechanisms)

        # 显示基本信息
        print(f"\n共现矩阵构建完成:")
        print(f"- 矩阵维度: {cooccurrence_matrix.shape}")
        print(f"- 总共现次数: {np.sum(cooccurrence_matrix) // 2}")
        print(f"- 非零Jaccard相似度对数量: {np.count_nonzero(jaccard_matrix) - len(mechanisms)}")

        # 分析结果
        analyze_results(cooccurrence_matrix, jaccard_matrix, mechanisms)

        # 保存结果
        save_cooccurrence_matrix(cooccurrence_matrix, jaccard_matrix, mechanism_individual_count, mechanisms,
                                 output_file)

        return cooccurrence_matrix, jaccard_matrix, mechanism_individual_count

    except Exception as e:
        print(f"处理文件时出错: {e}")
        return None, None, None


# 使用示例
if __name__ == "__main__":
    input_file = "../stage2/53_all_classification.xlsx"
    output_file = "mechanism_cooccurrence_analysis.xlsx"

    cooccurrence_matrix, jaccard_matrix, mechanism_count = process_excel_with_cooccurrence(input_file, output_file)

    if cooccurrence_matrix is not None:
        print(f"\n处理完成！结果已保存到: {output_file}")
        print("Excel文件中包含四个工作表:")
        print("1. 原始共现矩阵 - 机制对的共现次数")
        print("2. Jaccard相似度矩阵 - 标准化相似度(0-1)")
        print("3. 机制出现次数 - 每个机制单独出现的次数")
        print("4. 统计信息 - 各种统计指标")