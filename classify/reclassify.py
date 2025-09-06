import json
import numpy as np
import openai
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import umap
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# ========== 分类定义 ==========
classify_one = [
    'Human-AI Cognitive Interaction',
    'Trust, Ethics, and Social Acceptance',
    'AI Effectiveness and Management Practices',
    'Individual and Psychological Mechanisms'
]

classify_two_0 = [
    "Trust Calibration in AI Systems", "Cognitive Offloading and Automation Bias",
    "Explainability and Interpretability of AI", "Decision Confidence and Meta-Cognition",
    "Emotional Reactions to AI Errors"
]
classify_two_1 = [
    "Human-AI Trust Formation and Breakdown", "Algorithm Aversion and Algorithm Appreciation",
    "Ethical Implications of AI Decision-Making", "Perceived Control and Agency in AI-Assisted Decisions",
    "Accountability and Transparency in AI Governance"
]
classify_two_2 = [
    "AI-Supported Operational Decision-Making", "Strategic vs. Tactical AI Use in Organizations",
    "Performance Trade-offs: Speed, Accuracy, and Cost", "Organizational Readiness and AI Adoption",
    "Human-in-the-Loop Decision Systems"
]
classify_two_3 = [
    "Personality Traits and AI Reliance", "Expertise, Domain Knowledge and AI Overtrust",
    "Risk Perception and AI Trust", "Framing Effects in AI Recommendations",
    "Social Influence and Norms in AI Use"
]

classify_two = {
    0: classify_two_0,
    1: classify_two_1,
    2: classify_two_2,
    3: classify_two_3
}

classify_three = { (0, 0): ["Initial Trust Formation Mechanisms", "Trajectory of Trust Dynamics", "Overtrust vs. Distrust", "Trust Repair Mechanisms", "Domain-Specific Effects", "User Engagement and Trust Adjustment", "Perceived Cost of Trust Misalignment", "Trust Prediction Models in Multi-Round Tasks", "Trustworthiness Indicators Design"], (0, 1): ["AI-Guided vs. Self-Driven Decision-Making", "Post-Offloading Cognitive Decline", "Susceptibility to Automation Bias", "Formation of Dependency Habits", "Risk-Driven Offloading", "Offloading and Feedback Loops", "Error Detection Impairment", "Task Switching and Cognitive Load Distribution", "Assistance vs. Replacement Boundary Awareness"], (0, 2): ["Types of Explanation Tools", "User Explanation Preferences", "Presentation Modes", "Information Overload from Over-Explanation", "Explainability vs. Accuracy Trade-Off", "Adaptive Explanation Mechanisms", "Trust Enhancement through Explanation", "Explanation-Induced Error Correction", "Shifts in Accountability Due to Explanation"], (0, 3): ["Self-Confidence vs. AI Confidence", "Uncertainty Presentation and Meta-Cognitive Regulation", "Alignment of AI Confidence with Human Judgment", "Enhancing the ‘I Know I Don’t Know‘ Capability", "Performance Feedback and Meta-Cognitive Growth", "Non-Linear Relationship Between Meta-Cognition and Performance", "Post-Decision Correction Behavior", "Individual Differences in Meta-Cognitive Capacity", "Co-Construction of Confidence"], (0, 4): ["Error Type and Emotional Mapping", "High Expectation – High Disappointment Effect", "Emotion as Mediator in Trust Breakdown", "Individual Emotion Regulation as a Buffer", "Efficacy of Post-Error Mitigation Strategies", "Attribution of Emotional Response", "Emotional Memory Effects", "Affective Contagion vs. Emotional Containment", "Emotional Fluctuation Across Interactions"], (1, 0): ["Sources of Initial Trust", "Trust Accumulation Mechanisms", "Abrupt Trust Violations", "User Reactions to Trust Breakdown", "Comparing Trust Repair Strategies", "Impact of Task Risk on Trust Tolerance", "Trust Rebound vs. Trust Transfer", "Multimodal Trust Signaling", "Cultural Differences in Trust Dynamics"], (1, 1): ["Psychological Basis of Algorithm Aversion", "Pathways to Algorithm Appreciation", "Error Sensitivity and Trust Tolerance", "Domain-Specific Acceptance", "Impact of Authority Endorsement", "Transparency as a Moderator of Acceptance", "User-Type Differences", "Control and Algorithmic Attitudes", "Social Norm Influence"], (1, 2): ["Detection and Reaction to Algorithmic Bias", "Perceptions of Fairness", "Privacy Concerns and Usage Intentions", "Individual Differences in Ethical Boundaries", "Transparency of Ethical Frameworks", "Moral Ambiguity in Complex Situations", "Cross-Cultural Ethical Conflicts", "Acceptance of Ethical Auditing Mechanisms", "Responsibility Attribution for Unethical AI Behavior"], (1, 3): ["Sense of Control in Decision-Making", "Interface Design and Perceived Authority", "Mandatory Recommendations vs. Optional Advice", "Autonomy and User Satisfaction", "Depth of AI Involvement", "Individual Differences in Desire for Control", "Perceptions of Role Allocation in Joint Decisions", "Customizability of System Feedback", "Collapse of Control and Disengagement"], (1, 4): ["Responsibility Attribution Preferences", "Link Between Accountability and Trust", "Anxiety from 'Black Box' Decisions", "Auditable Decision Mechanisms", "Acceptance of Shared Responsibility Models", "Optimal Transparency Thresholds", "Access vs. Comprehensibility", "Corporate Legal and Moral Responsibility", "Support for Algorithm Rating or Blacklist Systems"], (2, 0): ["AI-based Scheduling Optimization", "Inventory Forecasting with AI", "Predictive Maintenance using AI", "AI in Supply Chain Coordination", "Warehouse Operations Optimization", "AI-Driven Forecasting Accuracy", "Performance Benchmarks for AI in Operations", "AI for Resource Allocation", "Scalability of Operational AI Systems"], (2, 1): ["AI in Strategic Planning", "AI in Tactical Execution", "Temporal Framing of AI Recommendations", "Organizational Alignment of AI Objectives", "Risk Appetite Across Strategic vs. Tactical AI", "Speed-Accuracy Tradeoffs at Different Levels", "Inter-Departmental AI Coordination", "Adaptive Learning in Strategic AI", "Scenario Planning with AI Support"], (2, 2): ["Speed vs. Accuracy Dilemma", "AI Decision Cost Models", "Latency-Tolerant vs. Latency-Critical Applications", "Approximate Algorithms for Real-Time Use", "AI Performance Under Resource Constraints", "Accuracy Plateau and Diminishing Returns", "Model Complexity vs. Interpretability", "Cost-Aware Model Selection", "Sustainability and Energy Trade-offs"], (2, 3): ["Technological Infrastructure for AI", "Cultural Readiness and Mindset", "Workforce AI Literacy", "Leadership Support and Vision", "Cross-Functional AI Teams", "Data Governance and Availability", "Internal vs. External AI Development", "Adoption Metrics and Success Criteria", "Organizational Barriers to AI Use"], (2, 4): ["Joint Human-AI Decision Models", "Role Allocation Between Human and AI", "Transparency in Human-AI Collaboration", "Error Detection and Correction Mechanisms", "Trust Calibration in Shared Decisions", "Interface Design for Hybrid Systems", "Accountability Distribution Models", "Real-Time Coordination Protocols", "Measuring Effectiveness of Human-AI Teams"], (3, 0): ["Openness to Experience and AI Exploration", "Neuroticism and Risk-Averse AI Use", "Conscientiousness and Rule-Based AI Interaction", "Extraversion and Social AI Acceptance", "Agreeableness and Trust Propensity", "Locus of Control and AI Delegation", "Impulsivity and Automation Bias", "Trait Epistemic Curiosity and Explanation Seeking", "Personality-AI Fit Perception"], (3, 1): ["Calibration of Trust in Experts vs. Novices", "Expert Overreliance on Domain-Specific AI", "Layperson Vulnerability to Illusory AI Authority", "Metacognitive Monitoring in Experts", "Training-Induced Trust Regulation", "Decision Audit Behavior by Professionals", "Mismatch Between AI Output and Domain Mental Models", "Trust Anchoring in Past Performance", "Deference in Ambiguous or High-Complexity Tasks"], (3, 2): ["Perceived Task Criticality and AI Avoidance", "High-Stakes Risk Amplification of AI Errors", "Low-Risk Contexts and Automation Comfort", "Risk Aversion Personality and Delegation Behavior", "AI Risk Communication Framing", "Risk Normalization Through Exposure", "Insurance and Liability Framing in Risky AI Use", "Trust-Risk Asymmetry", "Individual vs. Shared Risk Perception"], (3, 3): ["Gain vs. Loss Framing of AI Advice", "Default Framing and Opt-In/Opt-Out Decisions", "Certainty vs. Probability Language", "Authority Tone vs. Collaborative Tone", "Personalization and Self-Relevance Framing", "Emotionally Charged Language Framing", "Explanation Framing: Causal vs. Statistical", "Temporal Framing in Recommendations", "Trust Framing through Source Disclosure"], (3, 4): ["Descriptive Norms: ‘What Others Are Doing‘", "Injunctive Norms: ‘What Others Approve‘", "Peer Influence on AI Trust", "Organizational Culture and AI Acceptance", "Authority and Supervisor Endorsement", "Social Identity and Group-Based AI Use", "Fear of Judgement for Non-Use or Overuse", "AI Usage as a Status Symbol", "Collective Trust Calibration Mechanisms"] }


# ========== 解析 key 到 层级文本 ==========
def decode_key(key: str):
    """
    key 形如 "(1,2,3)"，返回一级/二级/三级的 index 和 内容
    """
    idx_tuple = eval(key)  # 转成元组
    one_idx, two_idx, three_idx = idx_tuple

    one_title = classify_one[one_idx]
    two_title = classify_two[one_idx][two_idx]
    three_title = classify_three[(one_idx, two_idx)][three_idx]

    return {
        "one_idx": one_idx,
        "one_title": one_title,
        "two_idx": two_idx,
        "two_title": two_title,
        "three_idx": three_idx,
        "three_title": three_title
    }


# ========== 读取 JSON ==========
def load_large_clusters(json_file, threshold=100):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    large_clusters = {k: v for k, v in data.items() if len(v) > threshold}
    print(f"筛选出 {len(large_clusters)} 个 value 超过 {threshold} 的 key")
    return large_clusters


# ========== 评估聚类 ==========
def evaluate_clusters(embeddings, jpg_file_path, cluster_range=range(2, 21), random_state=42):
    sil_scores, ch_scores, db_scores = [], [], []

    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        sil = silhouette_score(embeddings, labels)
        ch = calinski_harabasz_score(embeddings, labels)
        db = davies_bouldin_score(embeddings, labels)

        sil_scores.append(sil)
        ch_scores.append(ch)
        db_scores.append(db)

        print(f"n_clusters={k} | Silhouette={sil:.3f} | CH={ch:.1f} | DBI={db:.3f}")

    # 归一化
    sil_norm = (sil_scores - np.min(sil_scores)) / (np.max(sil_scores) - np.min(sil_scores) + 1e-9)
    ch_norm = (ch_scores - np.min(ch_scores)) / (np.max(ch_scores) - np.min(ch_scores) + 1e-9)
    db_inv = 1 / np.array(db_scores)
    db_norm = (db_inv - np.min(db_inv)) / (np.max(db_inv) - np.min(db_inv) + 1e-9)

    combined = (sil_norm + ch_norm + db_norm) / 3
    best_k = cluster_range[np.argmax(combined)]
    print(f"\n推荐的最佳聚类数 (综合评分): {best_k}")

    # 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(cluster_range, sil_scores, "o-", label="Silhouette")
    plt.plot(cluster_range, ch_scores, "s-", label="Calinski-Harabasz")
    plt.plot(cluster_range, db_scores, "d-", label="Davies-Bouldin")
    plt.plot(cluster_range, combined, "^-", label="Combined (norm)")
    plt.legend()
    plt.xlabel("Number of clusters")
    plt.title("Clustering Evaluation Metrics")
    plt.savefig(jpg_file_path)
    plt.show()

    return best_k


# ========== 聚类 ==========
def cluster_with_k(embeddings, n_clusters, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    return labels


# ========== 总结短语 ==========
def summarize_clusters(phrases, labels, max_words=5):
    df = pd.DataFrame({"phrase": phrases, "cluster": labels})
    summaries = {}

    for c in sorted(df["cluster"].unique()):
        cluster_phrases = df[df["cluster"] == c]["phrase"].tolist()
        if not cluster_phrases:
            summaries[c] = "N/A"
            continue

        vectorizer = TfidfVectorizer(stop_words="english", max_features=50)
        tfidf_matrix = vectorizer.fit_transform(cluster_phrases)
        scores = np.asarray(tfidf_matrix.sum(axis=0)).ravel()
        terms = vectorizer.get_feature_names_out()

        top_idx = np.argsort(scores)[::-1][:max_words]
        summary = " ".join([terms[i] for i in top_idx])
        summaries[c] = summary

    return summaries


# ========== 可视化 ==========
def visualize_clusters(embeddings, labels, png_file_path, n_neighbors=15, min_dist=0.1, n_components=2):
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=42)
    reduced = reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="tab20", s=20, alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Clusters", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title("UMAP Clustering Visualization")
    plt.savefig(png_file_path, bbox_inches='tight')
    plt.show()


def rename_three_title(three_title, phrases, api_key):
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )
    prompt = f"""你是一个学术研究助手。
    以下是某个三级分类【{three_title}】下的若干短语：
    {phrases[:]}  # 取前 50 个代表

    请基于这些短语，总结该三级分类的核心主题，并给它起一个简短（不超过6个单词）的英文描述，直接返回该英文描述，无需额外内容。
    """
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    reply = response.choices[0].message.content.strip()
    return reply


# ========== 主程序 ==========
if __name__ == "__main__":
    json_file = "phrases_by_topic.json"

    # 1. 找出大簇
    large_clusters = load_large_clusters(json_file, threshold=100)

    print("正在加载句向量模型...")
    model = SentenceTransformer("../initialize/all-MiniLM-L6-v2")

    for key, items in large_clusters.items():
        print("\n" + "="*50)
        print(f"处理 Key {key}，共 {len(items)} 个短语")

        phrases = [p[0] for p in items]
        rows = [p[1] for p in items]

        embeddings = model.encode(phrases, convert_to_numpy=True, show_progress_bar=True)

        jpg_file_path = f"./reclassify_results/key_{key.replace(',', '_')}_Clustering_Evaluation_Metrics.png"
        best_k = evaluate_clusters(embeddings, jpg_file_path, cluster_range=range(2, 15))
        labels = cluster_with_k(embeddings, best_k)

        summaries = summarize_clusters(phrases, labels)
        cluster_summary_col = [summaries[c] for c in labels]

        # 解码层级信息
        level_info = decode_key(key)


        df = pd.DataFrame({
            "phrase": phrases,
            "excel_row": rows,
            "cluster": labels,
            "cluster_summary": cluster_summary_col,
            "one_idx": level_info["one_idx"],
            "one_title": level_info["one_title"],
            "two_idx": level_info["two_idx"],
            "two_title": level_info["two_title"],
            "three_idx": level_info["three_idx"],
            "three_title": level_info["three_title"]
        })

        # 用 DeepSeek 给三级标题重新命名
        api_key = "sk-73b1e9510c504a54a5eeabeadde2d51e"
        new_three_title = rename_three_title(level_info["three_title"], phrases, api_key)
        print(f"原三级标题: {level_info['three_title']} -> 新标题: {new_three_title}")

        df["three_title_renamed"] = new_three_title

        out_file = f"./reclassify_results/cluster_results_key_{key.replace(',', '_')}_k{best_k}.xlsx"
        df.to_excel(out_file, index=False)
        print(f"聚类结果已保存到 {out_file}")

        png_file_path = f"./reclassify_results/key_{key.replace(',', '_')}_UMAP_Clustering_Visualization.png"
        visualize_clusters(embeddings, labels, png_file_path)
