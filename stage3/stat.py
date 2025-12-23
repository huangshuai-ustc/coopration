import pandas as pd
import ast

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

level = {
    "high": 3,
    "medium": 2,
    "low": 1
}

def process_excel_file(file_path):
    """
    读取Excel文件并处理mechanism等级信息

    Parameters:
    file_path (str): Excel文件路径

    Returns:
    pd.DataFrame: 处理后的DataFrame
    """
    # 读取Excel文件
    df = pd.read_excel(file_path)

    # 确保所需的列存在
    if 'classification' not in df.columns or 'levels' not in df.columns:
        raise ValueError("Excel文件中必须包含'classification'和'levels'列")

    # 为每个机制创建新列，初始值为'无'
    for mechanism in mechanisms:
        df[mechanism] = '无'

    # 处理每一行数据
    for index, row in df.iterrows():
        try:
            # 解析classification列（字符串形式的列表）
            if pd.notna(row['classification']) and row['classification'].strip():
                classification_list = ast.literal_eval(row['classification'])
            else:
                classification_list = []

            # 解析levels列（字符串形式的字典）
            if pd.notna(row['levels']) and row['levels'].strip():
                levels_dict = ast.literal_eval(row['levels'])
            else:
                levels_dict = {}

            # 为每个机制设置等级
            for mechanism in mechanisms:
                if mechanism in classification_list:
                    if mechanism in levels_dict:
                        df.at[index, mechanism] = level[levels_dict[mechanism]]
                    else:
                        # 如果在classification中但levels中没有，标记为'未知'
                        df.at[index, mechanism] = -1
                else:
                    # 不在classification中，保持'无'
                    df.at[index, mechanism] = 0

        except (ValueError, SyntaxError) as e:
            print(f"解析第 {index + 1} 行数据时出错: {e}")
            # 出错时将所有机制设为'解析错误'
            for mechanism in mechanisms:
                df.at[index, mechanism] = '解析错误'

    return df


# 使用示例
if __name__ == "__main__":
    # 替换为你的Excel文件路径
    input_file = "../stage2/53_all_classification.xlsx"
    output_file = "processed_file.xlsx"

    try:
        # 处理文件
        processed_df = process_excel_file(input_file)

        # 保存处理后的文件
        processed_df.to_excel(output_file, index=False)
        print(f"文件处理完成，已保存为: {output_file}")

        # 显示处理后的数据前几行
        print("\n处理后的数据前几行:")
        print(processed_df.head())

    except Exception as e:
        print(f"处理文件时出错: {e}")

