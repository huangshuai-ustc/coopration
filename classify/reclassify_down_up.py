import json

import pandas as pd
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from treelib import Tree
import matplotlib.pyplot as plt
import uuid
from collections import defaultdict

# ===========================
# 参数设置
# ===========================
file_path = "classify_result_by_xinyue.xlsx"       # 输入 Excel 文件名
output_file = "clustered_phrases_bottom2up.xlsx"  # 输出 Excel 文件名

# 每层的簇数，可以自己改
cluster_layers = [4, 5, 7]   # 固定簇数写 int

df = pd.read_excel(file_path)
phrases = df.iloc[:, 0].dropna().astype(str).tolist()
print(f"读取到 {len(phrases)} 个短语")

# ===========================
# 2. 向量化短语
# ===========================
model = SentenceTransformer('../initialize/all-MiniLM-L6-v2')
embeddings = model.encode(phrases, show_progress_bar=True)

# ===========================
# 3. 逐层聚类（自底向上）
# ===========================
linked = linkage(embeddings, method='ward')
labels_per_layer = []

for n_clusters in cluster_layers:
    layer_labels = fcluster(linked, t=int(n_clusters), criterion='maxclust')
    labels_per_layer.append(layer_labels)
    print(f"完成一层聚类，簇数={n_clusters}")


client = OpenAI(
    api_key="sk-73b1e9510c504a54a5eeabeadde2d51e",  # ⚠️ 替换为你自己的
    base_url="https://api.deepseek.com"
)


def req_v3(promptxxxxx):
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": promptxxxxx}]
        )
        reply = response.choices[0].message.content.strip()
        return reply
    except Exception as e:
        print("⚠️ DeepSeek API 调用失败:", e)
        return "Unknown"


def get_prompt(title, abstract, keywords, prompt_info):
    """ get_prompt """
    if not isinstance(keywords, str):
        keywords = ""
    prompt_tmp = prompt_info
    prompt_tmp = prompt_tmp.replace("<TITLE>", title)
    prompt_tmp = prompt_tmp.replace("<ABSTRACT>", abstract)
    prompt_tmp = prompt_tmp.replace("<KEYWORDS>", keywords)
    return prompt_tmp


def make_offline_input(title, abstract, keywords, promptxxx):
    prompt_infos = get_prompt(title, abstract, keywords, promptxxx)
    req_dict = {
        "request_body": {
            "system": "",
            "messages": [
                {
                    "role": "user",
                    "content": prompt_infos
                }
            ],
            "top_p": 0.8,
            "temperature": 1,
            "penalty_score": 1
        }
    }

    out_list = json.dumps(req_dict, ensure_ascii=False)
    return out_list


# ===========================
# 5. 构建树状结构 + 导出 Excel
# ===========================
tree = Tree()
tree.create_node("所有短语", "root")

records = []
cluster_to_phrases = defaultdict(list)

# 收集每一层类别对应的短语，并记录 parent
for i, phrase in enumerate(phrases):
    parent_id = "root"
    for layer_idx, layer_labels in enumerate(labels_per_layer):
        cluster_id = layer_labels[i]
        node_name = f"L{layer_idx+1}_C{cluster_id}"
        node_id = f"{node_name}_{layer_idx}"

        cluster_to_phrases[(layer_idx, cluster_id)].append((phrase, parent_id))
        parent_id = node_id  # 下一层的父节点就是当前层

    # 保存记录
    record = {"行号": i+1, "短语": phrase}
    for j, lbl in enumerate(labels_per_layer):
        record[f"第{j+1}层类别"] = f"L{j+1}_C{lbl[i]}"
    records.append(record)

# 创建树
cluster_desc_cache = {}
for (layer_idx, cluster_id), items in cluster_to_phrases.items():
    phrases_in_cluster = [p for p, _ in items]
    parent_id = items[0][1]   # 取第一个的 parent

    node_name = f"L{layer_idx+1}_C{cluster_id}"
    node_id = f"{node_name}_{layer_idx}"

    if node_id not in cluster_desc_cache:
        desc = req_v3(phrases_in_cluster)
        cluster_desc_cache[node_id] = desc
        tree.create_node(f"{node_name} ({desc})", node_id, parent=parent_id)

    # 每个类下的短语节点
    for p, _ in items:
        phrase_id = f"p_{uuid.uuid4().hex}"
        tree.create_node(p, phrase_id, parent=node_id)

print("\n===== 分类结果树 =====")
tree.show()

# 给 Excel 增加类别描述
for record in records:
    for j, lbl in enumerate(labels_per_layer):
        cluster_id = lbl[records.index(record)]
        node_id = f"L{j+1}_C{cluster_id}_{j}"
        desc = cluster_desc_cache.get(node_id, "")
        record[f"第{j+1}层描述"] = desc

output_df = pd.DataFrame(records)
output_df.to_excel(output_file, index=False)
print(f"✅ 结果已导出到 {output_file}")

# # ===========================
# # 6. 全局 Dendrogram 可视化
# # ===========================
# plt.figure(figsize=(10, 6))
# dendrogram(linked, labels=phrases, orientation='right', leaf_font_size=8)
# plt.title("短语自底向上层次聚类树 (Dendrogram)")
# plt.tight_layout()
# plt.show()
