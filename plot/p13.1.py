import pandas as pd
import networkx as nx
from ast import literal_eval
from pyvis.network import Network
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import math

# ========== 参数 ==========
file_path = "variable_length_group_chains_with_content.xlsx"
layout_mode = "line"  # 可选: "force", "star", "line"
# =========================

# 1. 读取 Excel
df = pd.read_excel(file_path)
df['具体内容'] = df['具体内容'].apply(literal_eval)

# 2. 构建有向图
G = nx.DiGraph()
node_weights = {}

for _, row in df.iterrows():
    count = row['支持数量']
    nested_phrases = row['具体内容']

    for sublist in nested_phrases:
        for phrase in sublist:
            node_weights[phrase] = node_weights.get(phrase, 0) + count
            G.add_node(phrase)

        for i in range(len(sublist) - 1):
            src = sublist[i]
            tgt = sublist[i + 1]
            if G.has_edge(src, tgt):
                G[src][tgt]['weight'] += count
            else:
                G.add_edge(src, tgt, weight=count)

# 3. 颜色映射
cmap = cm.get_cmap("coolwarm", 256)
norm = mcolors.Normalize(vmin=min(node_weights.values()), vmax=max(node_weights.values()))

def get_color(value):
    rgba = cmap(norm(value))
    return mcolors.to_hex(rgba)

# 4. pyvis 交互式网络
net = Network(height="1000px", width="100%", directed=True, notebook=False)

set1 = set()
if layout_mode == "line":
    # 每个链条单独一行
    x_offset = 200
    y_offset = 100
    y_counter = 0  # 连续行计数
    for nested_phrases in df['具体内容']:
        for sublist in nested_phrases:
            if tuple(sublist) not in set1:
                for col_idx, phrase in enumerate(sublist):
                    freq = node_weights[phrase]
                    net.add_node(
                        phrase,
                        size=15,
                        color=get_color(freq),
                        title=f"{phrase}<br>出现次数: {freq}",
                        physics=False,
                        x=col_idx * x_offset,
                        y=y_counter * y_offset
                    )
                y_counter += 1  # 每个子列表占一行
                set1.add(tuple(sublist))


elif layout_mode == "star":
    # 星形布局
    center_node = max(node_weights, key=node_weights.get)
    net.add_node(center_node, size=40, color="red",
                 title=f"{center_node}<br>出现次数: {node_weights[center_node]}",
                 physics=False, x=0, y=0)

    radius = 400
    for idx, (node, freq) in enumerate(node_weights.items()):
        if node == center_node:
            continue
        angle = 2 * math.pi * idx / (len(node_weights) - 1)
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        net.add_node(node, size=15, color=get_color(freq),
                     title=f"{node}<br>出现次数: {freq}",
                     physics=False, x=x, y=y)

else:
    # 默认力导向
    for node, freq in node_weights.items():
        net.add_node(node, size=15, color=get_color(freq),
                     title=f"{node}<br>出现次数: {freq}")

# 5. 添加边
for u, v, data in G.edges(data=True):
    net.add_edge(u, v,
                 value=data.get("weight", 1),
                 title=f"权重: {data.get('weight', 1)}",
                 smooth=False)

# 6. 输出
net.write_html("phrase_network.html")
print("✅ 已生成 phrase_network.html，打开即可交互查看")
