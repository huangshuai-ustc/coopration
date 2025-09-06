import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from pathlib import Path

FILE_PATH = Path("../stage3/mechanism_cooccurrence_analysis.xlsx")

# 读取前两个子表
xls = pd.ExcelFile(FILE_PATH)
cooc = pd.read_excel(xls, sheet_name=0, index_col=0)
sim = pd.read_excel(xls, sheet_name=1, index_col=0)

# --- 共现矩阵热力图 ---
plt.figure(figsize=(10,8))
sns.heatmap(cooc, cmap="YlGnBu", annot=False)  # annot=True 可以显示具体数值
plt.title("Co-occurrence Matrix Heatmap")
plt.tight_layout()
plt.savefig("heatmap_cooccurrence.png", dpi=600)
plt.show()

# --- 相似度矩阵热力图 ---
plt.figure(figsize=(10,8))
sns.heatmap(sim, cmap="YlOrRd", annot=False)
plt.title("Similarity Matrix Heatmap")
plt.tight_layout()
plt.savefig("heatmap_similarity.png", dpi=600)
plt.show()

# --- 可选：带聚类的热力图（更清晰分群） ---
# 禁用默认 colorbar
cg = sns.clustermap(sim, cmap="YlOrRd", figsize=(12,12), row_cluster=True, col_cluster=True, cbar_pos=None)

# 隐藏左侧行 dendrogram
cg.ax_row_dendrogram.set_visible(False)

# 获取 heatmap Axes
ax = cg.ax_heatmap

# 手动添加 colorbar 到左侧
norm = Normalize(vmin=sim.min().min(), vmax=sim.max().max())
sm = cm.ScalarMappable(cmap="YlOrRd", norm=norm)
sm.set_array([])

# colorbar 放在左边
cbar_ax = cg.fig.add_axes([0.05, 0.2, 0.02, 0.6])  # [left, bottom, width, height]
cbar = cg.fig.colorbar(sm, cax=cbar_ax)
cbar.set_label("Similarity")

plt.savefig("clustermap_similarity_left_cbar.png", dpi=600, bbox_inches='tight')
plt.show()
