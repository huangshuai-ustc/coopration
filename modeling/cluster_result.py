import pandas as pd
import ast

# 读取原始带聚类标签的文件
df = pd.read_csv("abstracts_with_phrases_filtered.csv")

NUM_CLUSTERS = 8
# 准备两个列表来存储展开的数据
all_phrases = []
all_labels = []

# 遍历每行，将 phrases 和 cluster_labels 逐个拆开
for phrases_str, labels_str in zip(df['key_phrases'], df[f"cluster_labels_{NUM_CLUSTERS}"]):
    phrases = ast.literal_eval(phrases_str)
    labels = ast.literal_eval(labels_str)

    if len(phrases) != len(labels):
        raise ValueError("每行的 phrases 和 cluster_labels 长度不一致！")

    all_phrases.extend(phrases)
    all_labels.extend(labels)

# 构建新 DataFrame
flattened_df = pd.DataFrame({
    "phrase": all_phrases,
    "cluster_label": all_labels
})

# 按 cluster_label 升序排列
flattened_df.sort_values(by="cluster_label", inplace=True)

# 保存为新 CSV 文件
flattened_df.to_csv(f"./cluster_results/flattened_phrases_with_clusters_{NUM_CLUSTERS}.csv", index=False)
