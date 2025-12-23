import pandas as pd
import ast
import tqdm as tqdm
# import sentence_transformers.SentenceTransformer as SentenceTransformer
import sentence_transformers.util as util
from sentence_transformers import SentenceTransformer

# from sentence_transformers import SentenceTransformer, util


# ======== 一级主题 ========
classify_one = [
    'Human–AI Cognition & Decision Mechanisms',
    'Trust, Transparency & Explainability',
    'Emotion, Empathy & Social Acceptance',
    'Algorithm Perceptions, Attitudes & Psychological Mechanisms',
    'Human–AI Teaming & Control',
    'Human–AI Service & Psychological Mechanisms',
    'Human–AI Collaboration, Safety & Societal Governance'
]

# ======== 二级主题 ========
classify_two = {
    0: [
        "Information Processing & Situation Awareness",
        "Automation Reliance & Biases",
        "Decision Aids & Cognitive Offloading"
    ],
    1: [
        "Trust Formation & Calibration",
        "Explainability & Transparency",
        "Governance & Accountability"
    ],
    2: [
        "Emotional Reactions & Mental Workload",
        "Empathy & Social Interaction",
        "Social Acceptance & Resistance"
    ],
    3: [
        "Algorithmic Attitudes",
        "Cognitive & Decision Styles",
        "Mental Models & Biases"
    ],
    4: [
        "Collaboration & Coordination",
        "Role Allocation & Adaptation",
        "Control & Intervention"
    ],
    5: [
        "Experience & Interaction",
        "Augmentation & Immersion",
        "Psychological Mechanisms & Anthropomorphism"
    ],
    6: [
        "Trust & Safety",
        "Social Signals & Norms",
        "Governance & Ethics"
    ]
}

# ======== 三级主题 ========
classify_three = {
    (0, 0): [
        "Situation Awareness & Dynamic Representations",
        "Information Overload & Inadequacies",
        "Attention Allocation & Conflict Detection"
    ],
    (0, 1): [
        "Automation Bias: Omission & Commission",
        "Skill Degradation & Complacency",
        "Use, Disuse, Misuse"
    ],
    (0, 2): [
        "Human-in-the-Loop Supervision",
        "Cognitive Offloading & Decision Support",
        "Dynamic Allocation & Levels of Automation"
    ],

    (1, 0): [
        "Initial Trust & Revision",
        "Trust Repair & Recalibration",
        "Trust Inertia & Volatility"
    ],
    (1, 1): [
        "Decision Path Interpretability",
        "Uncertainty & Risk Communication",
        "Model Transparency & Auditability"
    ],
    (1, 2): [
        "Fairness & Responsibility Allocation",
        "Accountability & Safety Standards",
        "Policy-Making & Societal Trust"
    ],

    (2, 0): [
        "Emotions to Automation Errors",
        "Stress & Frustration",
        "Emotion-Driven Trust"
    ],
    (2, 1): [
        "Empathic Agents & Affective Communication",
        "Social Roles in HAI",
        "Long-Term Attachment & Parasocial Relations"
    ],
    (2, 2): [
        "Social Norms & Adoption",
        "Resistance & Motivational Factors",
        "Cross-Cultural Differences & Psychological Reactions"
    ],

    (3, 0): [
        "Algorithm Aversion vs. Appreciation",
        "Comparisons: Human vs. Algorithm",
        "Perceived Usefulness & Effectiveness"
    ],
    (3, 1): [
        "Heuristic vs. Analytical Styles",
        "Satisficing & Anchoring Bias",
        "Risk Perception & Trust"
    ],
    (3, 2): [
        "Mental Model Adjustments",
        "Biases & Misconceptions",
        "Spillover & Cross-Domain Effects"
    ],

    (4, 0): [
        "Team Management & Situation Awareness",
        "Ad-hoc Cooperation & Task Switching",
        "Error Prevention in Teams"
    ],
    (4, 1): [
        "Dynamic Role Allocation",
        "Function Allocation & Complementarity",
        "Team Adaptation & Transition"
    ],
    (4, 2): [
        "Attention & Control Conflicts",
        "Takeover & Supervisory Mechanisms",
        "Interventions & Feedback"
    ],

    (5, 0): [
        "Affective Interactions",
        "Communication Styles & Satisfaction",
        "Adoption & Continuance Intention"
    ],
    (5, 1): [
        "Immersive Experience & Engagement",
        "Social Inclusion & Perceptions",
        "Tech-Mediated Experience"
    ],
    (5, 2): [
        "Psychological Safety & Hedonic Value",
        "Anthropomorphism & Affinity Perception",
        "Parasocial Relations & Emotional Attachment"
    ],

    (6, 0): [
        "Trust Breakdown & Repair",
        "Reliance, Misuse & Resistance",
        "Perceived Safety & Risk Assessment"
    ],
    (6, 1): [
        "External Interfaces & Communication Mechanisms",
        "Social Norms & Behavioral Coordination",
        "Public Opinion & Social Acceptance"
    ],
    (6, 2): [
        "Algorithmic Management & Organizational Responsibility",
        "Privacy, Security & Ethical Discourses",
        "Responsible AI & Cross-Cultural Governance"
    ]
}


# ======== 读取 Excel ========
print("正在读取Excel文件...")
df = pd.read_excel("classify_result_by_xinyue.xlsx")

# 第14列转列表
col_idx = 0
# df.iloc[:, col_idx] = df.iloc[:, col_idx].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# ======== 加载模型 ========
print("正在加载模型...")
model = SentenceTransformer('../initialize/all-MiniLM-L6-v2')

# ======== 编码所有分类 ========
print("正在编码一级主题...")
emb_one = model.encode(classify_one, convert_to_tensor=True)

print("正在编码二级主题...")
emb_two = {i: model.encode(v, convert_to_tensor=True) for i, v in classify_two.items()}

print("正在编码三级主题...")
emb_three = {k: model.encode(v, convert_to_tensor=True) for k, v in classify_three.items()}

# ======== 处理每个短语 ========
results = []

for row_idx, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="处理进度"):
    phrase = row.iloc[col_idx]

    phrase_emb = model.encode(phrase, convert_to_tensor=True)

    # 一级
    sims_one = util.cos_sim(phrase_emb, emb_one)[0].cpu().tolist()
    best_one_idx = sims_one.index(max(sims_one))
    # best_one = classify_one[best_one_idx]

    # 二级
    sims_two = util.cos_sim(phrase_emb, emb_two[best_one_idx])[0].cpu().tolist()
    best_two_idx = sims_two.index(max(sims_two))
    # best_two = classify_two[best_one_idx][best_two_idx]

    # 三级
    sims_three = util.cos_sim(phrase_emb, emb_three[(best_one_idx, best_two_idx)])[0].cpu().tolist()
    best_three_idx = sims_three.index(max(sims_three))
    # best_three = classify_three[(best_one_idx, best_two_idx)][best_three_idx]

    results.append({
        "phrase": phrase,
        "excel_row": row_idx,
        "topic1_index": best_one_idx,
        "topic2_index": best_two_idx,
        "topic3_index": best_three_idx,
        "topic1_content": classify_one[best_one_idx],
        "topic2_content": classify_two[best_one_idx][best_two_idx],
        "topic3_content": classify_three[(best_two_idx, best_three_idx)][best_three_idx],
    })

# ======== 保存结果 ========
output_df = pd.DataFrame(results)
output_df.to_excel("classify_results_new.xlsx", index=False)
print("处理完成！结果已保存到 classify_results_new.xlsx")
