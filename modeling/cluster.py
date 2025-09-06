import pandas as pd
import ast
import numpy as np
from sklearn.cluster import KMeans
from umap import UMAP
from sentence_transformers import SentenceTransformer
import plotly.graph_objects as go
import matplotlib.pyplot as plt

NUM_CLUSTERS = 3  # 设置聚类数量

# 读取 CSV 文件并解析最后一列为短语列表
df = pd.read_csv("abstracts_with_phrases_filtered.csv")
phrase_lists = df['key_phrases'].apply(ast.literal_eval)


# 扁平化所有短语
texts = [phrase for sublist in phrase_lists for phrase in sublist]

# 加载 BERT 模型并计算句子嵌入
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts)

# 使用 UMAP 降维至 3D
embedding_3d = UMAP(n_neighbors=5, n_components=3, min_dist=0.3, random_state=42).fit_transform(embeddings)

# 使用 KMeans 进行聚类
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
labels = kmeans.fit_predict(embedding_3d)

# 提取每个聚类的代表性短语（最靠近质心）
cluster_names = {}
for cluster_id in range(NUM_CLUSTERS):
    idxs = np.where(labels == cluster_id)[0]
    cluster_points = embedding_3d[idxs]
    centroid = kmeans.cluster_centers_[cluster_id]
    distances = np.linalg.norm(cluster_points - centroid, axis=1)
    best_idx = idxs[np.argmin(distances)]
    cluster_names[cluster_id] = texts[best_idx]

# 可交互 3D 可视化（使用 Plotly）
colors = ['blue', 'red', 'green', 'yellow', 'black', 'orange', 'purple', 'cyan']
fig = go.Figure()

for cluster_id in range(NUM_CLUSTERS):
    indices = np.where(labels == cluster_id)[0]
    fig.add_trace(go.Scatter3d(
        x=embedding_3d[indices, 0],
        y=embedding_3d[indices, 1],
        z=embedding_3d[indices, 2],
        mode='markers',
        marker=dict(size=5, color=colors[cluster_id], opacity=0.8),
        name=cluster_names[cluster_id]
    ))

fig.update_layout(
    title="3D Scatter Plot of Sentence Embeddings (Interactive)",
    scene=dict(
        xaxis_title='x',
        yaxis_title='y',
        zaxis_title='z'
    ),
    legend=dict(title="Cluster Topics"),
    margin=dict(l=0, r=0, b=0, t=30)
)

# 保存为 HTML 文件
fig.write_html(f"./cluster_results/3d_embedding_plot_cluster_{NUM_CLUSTERS}.html")

# 在交互环境中显示（可选）
fig.show()

# ===== 可选：2D 降维并静态绘图（仍使用 matplotlib） =====

# 降维至 2D
embedding_2d = UMAP(n_neighbors=5, n_components=2, min_dist=0.3, random_state=42).fit_transform(embeddings)

# 聚类
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
labels = kmeans.fit_predict(embedding_2d)

# 提取主题
cluster_names = {}
for cluster_id in range(NUM_CLUSTERS):
    idxs = np.where(labels == cluster_id)[0]
    cluster_points = embedding_2d[idxs]
    centroid = kmeans.cluster_centers_[cluster_id]
    distances = np.linalg.norm(cluster_points - centroid, axis=1)
    best_idx = idxs[np.argmin(distances)]
    cluster_names[cluster_id] = texts[best_idx]

plt.figure(figsize=(10, 7))
for cluster_id in range(NUM_CLUSTERS):
    indices = np.where(labels == cluster_id)[0]
    plt.scatter(embedding_2d[indices, 0], embedding_2d[indices, 1],
                c=colors[cluster_id],
                label=cluster_names[cluster_id], alpha=0.8)

plt.title("2D Scatter Plot of Sentence Embeddings by UMAP")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"./cluster_results/2d_embedding_plot_cluster_{NUM_CLUSTERS}.png")
plt.show()

# 将扁平化的 labels 重新按每行原始 phrase_list 的结构组合起来
phrase_cluster_labels = []
i = 0
for phrase_list in phrase_lists:
    num_phrases = len(phrase_list)
    cluster_label_list = labels[i:i+num_phrases].tolist()
    phrase_cluster_labels.append(cluster_label_list)
    i += num_phrases

# 加到原始 DataFrame 中
df[f"cluster_labels_{NUM_CLUSTERS}"] = phrase_cluster_labels

# 保存新文件
df.to_csv("abstracts_with_phrases_filtered_classify.csv", index=False)
