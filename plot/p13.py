import pandas as pd
import networkx as nx
from ast import literal_eval
from pyvis.network import Network
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# 1. 读取 Excel
file_path = "variable_length_group_chains_with_content.xlsx"
df = pd.read_excel(file_path)

# 第二列是出现次数，第三列是嵌套列表
df['具体内容'] = df['具体内容'].apply(literal_eval)

# 2. 构建有向图
G = nx.DiGraph()
node_weights = {}  # 统计每个节点出现的总次数

for _, row in df.iterrows():
    count = row['支持数量']
    nested_phrases = row['具体内容']

    for sublist in nested_phrases:
        for phrase in sublist:
            node_weights[phrase] = node_weights.get(phrase, 0) + count

        for i in range(len(sublist) - 1):
            src = sublist[i]
            tgt = sublist[i + 1]
            if G.has_edge(src, tgt):
                G[src][tgt]['weight'] += count
            else:
                G.add_edge(src, tgt, weight=count)

# 3. 颜色映射（节点出现次数 -> 颜色）
cmap = cm.get_cmap("coolwarm", 256)  # 蓝到红渐变
norm = mcolors.Normalize(vmin=min(node_weights.values()), vmax=max(node_weights.values()))

def get_color(value):
    rgba = cmap(norm(value))
    return mcolors.to_hex(rgba)

# 4. 用 pyvis 生成交互式网络
net = Network(height="800px", width="100%", directed=True, notebook=False)

# 添加节点（大小=度数，颜色=出现次数）
for node, freq in node_weights.items():
    net.add_node(
        node,
        size=10 + G.degree(node) * 2,
        color=get_color(freq),
        title=f"{node}<br>出现次数: {freq}"
    )

# 添加边（粗细=权重）
for u, v, data in G.edges(data=True):
    net.add_edge(
        u, v,
        value=data.get("weight", 1),
        title=f"权重: {data.get('weight', 1)}"
    )

# 5. 输出 HTML
net.write_html("phrase_network.html")
