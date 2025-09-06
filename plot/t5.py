import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# 读取 Excel 文件
df = pd.read_excel("../stage3/processed_file.xlsx")
data = df.loc[:, "cognitive_offload":"psychological_expansion"]

# 定义群体及其列名
groups = {
    1: ["cognitive_offload", "cognitive_load_management", "information_processing_aid"],
    2: ["ambiguity_reduction", "uncertainty_communication", "situation_awareness"],
    3: ["bounded_rationality", "decision_heuristics", "anchoring_adjustment",
        "bias_mitigation_amplification", "error_probability_estimation"],
    4: ["preference_learning", "mind_perception", "cognition_emotion_interaction",
        "emotion_recognition", "ai_empathy_dynamics"],
    5: ["affective_transition", "stress_anxiety_regulation", "motivation_engagement",
        "psychological_safety", "self_efficacy", "agency_restoration", "moral_agency"],
    6: ["responsibility_allocation", "responsibility_gap", "ethical_dissonance",
        "ai_threat_perceptions", "negative_work_rumination", "approach_avoidance_dynamics",
        "psychological_expansion"],
    7: ["human_ai_interaction", "interaction_fluency", "social_presence",
        "anthropomorphism_effects", "human_machine_teaming", "cooperation_competition_dynamics",
        "delegated_decision_making"],
    8: ["coordination_ability", "team_shared_awareness", "feedback_loops",
        "task_crafting", "error_recovery_support"],
    9: ["trust_dynamics", "trust_calibration", "trust_miscalibration",
        "trust_repair_strategies", "algorithmic_attitudes", "automation_bias",
        "levels_of_autonomy", "adaptive_automation", "automation_transparency"],
    10: ["perceived_fairness", "value_alignment"]
}

# 计算每个群体在每行是否出现
group_presence = pd.DataFrame()
for g, cols in groups.items():
    group_presence[g] = (data[cols] != 0).any(axis=1).astype(int)

# 初始化链条统计字典
chain_counts = {}
chain_elements = {}

# 遍历每行
for _, row in group_presence.iterrows():
    present_groups = [g for g, val in row.items() if val == 1]
    # 遍历长度 2 到 len(present_groups) 的所有连续组合
    for L in range(2, len(present_groups)+1):
        for i in range(len(present_groups) - L + 1):
            chain = tuple(present_groups[i:i+L])
            chain_counts[chain] = chain_counts.get(chain, 0) + 1
            # 保存链条对应的具体列名
            chain_elements[chain] = [groups[g] for g in chain]

# 输出 Excel 文件
results_df = pd.DataFrame([
    {"群体链条": " → ".join(["Group"+str(g) for g in chain]),
     "支持数量": count,
     "具体内容": chain_elements[chain]}
    for chain, count in chain_counts.items()
])
results_df.to_excel("variable_length_group_chains_with_content.xlsx", index=False)

# 绘制网络图
G = nx.DiGraph()
for chain, count in chain_counts.items():
    for i in range(len(chain)-1):
        src, dst = chain[i], chain[i+1]
        if G.has_edge(src, dst):
            G[src][dst]['weight'] += count
        else:
            G.add_edge(src, dst, weight=count)

plt.figure(figsize=(12,8))
pos = nx.spring_layout(G, seed=42)
edges = G.edges()
weights = [G[u][v]['weight'] for u,v in edges]

nx.draw(G, pos, with_labels=True, node_size=1500, node_color='skyblue',
        edge_color='orange', width=[w/5 for w in weights], arrowsize=20)

plt.title("Variable Length Group Chains Network")
plt.show()
