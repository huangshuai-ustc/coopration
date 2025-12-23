import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# ====================== 1. 模拟数据（替换为你的实际数据） ======================
data = {
    1969: [0, 9, 0, 0, 1, 0, 0],
    1971: [0, 8, 0, 0, 0, 0, 0],
    1973: [0, 3, 0, 0, 0, 0, 0],
    1981: [0, 7, 0, 0, 0, 0, 0],
    1982: [0, 18, 0, 0, 0, 2, 0],
    1985: [0, 3, 1, 0, 1, 1, 0],
    1986: [0, 16, 0, 0, 2, 2, 0],
    1987: [0, 4, 0, 0, 0, 0, 0],
    1990: [0, 8, 0, 0, 1, 0, 0],
    1991: [0, 12, 0, 0, 0, 0, 0],
    1993: [0, 7, 0, 0, 2, 0, 0],
    1994: [0, 7, 0, 0, 0, 1, 0],
    1995: [0, 6, 0, 0, 0, 1, 0],
    1996: [0, 27, 0, 0, 3, 3, 0],
    1997: [0, 23, 1, 0, 1, 1, 0],
    1999: [0, 20, 0, 1, 8, 2, 0],
    2000: [2, 66, 3, 0, 4, 0, 1],
    2001: [1, 30, 3, 0, 3, 0, 0],
    2002: [0, 35, 0, 0, 4, 0, 0],
    2003: [0, 1, 1, 0, 1, 0, 0],
    2004: [0, 35, 2, 0, 4, 0, 0],
    2005: [0, 15, 1, 0, 1, 0, 0],
    2006: [0, 15, 0, 1, 4, 0, 0],
    2007: [0, 9, 1, 0, 1, 0, 0],
    2008: [0, 12, 0, 1, 0, 1, 0],
    2009: [0, 29, 0, 0, 3, 0, 0],
    2010: [0, 27, 3, 0, 2, 2, 1],
    2011: [0, 19, 1, 0, 3, 2, 0],
    2012: [0, 29, 0, 0, 4, 1, 0],
    2013: [0, 24, 3, 1, 3, 3, 0],
    2014: [0, 47, 0, 0, 7, 7, 1],
    2015: [1, 57, 0, 1, 7, 3, 1],
    2016: [0, 150, 1, 2, 9, 5, 1],
    2017: [1, 124, 0, 2, 17, 3, 0],
    2018: [2, 157, 9, 2, 25, 8, 5],
    2019: [1, 279, 10, 6, 22, 5, 5],
    2020: [2, 265, 11, 3, 29, 9, 1],
    2021: [2, 406, 6, 5, 52, 16, 3],
    2022: [3, 450, 12, 7, 70, 18, 5],
    2023: [4, 936, 33, 21, 107, 31, 8],
    2024: [5, 1661, 54, 26, 163, 55, 21],
    2025: [5, 268, 17, 0, 25, 4, 1]
}

# ====================== 2. 数据预处理 ======================
# 转换为 DataFrame
df = pd.DataFrame.from_dict(data, orient='index')
df.columns = [
    'Human–AI Cognition & Decision Mechanisms',
    'Trust, Transparency & Explainability',
    'Emotion, Empathy & Social Acceptance',
    'Algorithm Perceptions, Attitudes & Psychological Mechanisms',
    'Human–AI Teaming & Control',
    'Human–AI Service & Psychological Mechanisms',
    'Human–AI Collaboration, Safety & Societal Governance'
]
df.index.name = 'Year'

# 补全缺失年份（可选）
all_years = range(1969, 2026)
df = df.reindex(all_years, fill_value=0)

# 数据标准化（可选：解决维度量纲差异）
scaler = MinMaxScaler()
df_normalized = pd.DataFrame(
    scaler.fit_transform(df),
    index=df.index,
    columns=df.columns
)

# ====================== 3. 基础热力图 ======================
plt.figure(figsize=(14, 8))
sns.heatmap(
    df_normalized.T,  # 转置：维度在行，年份在列
    cmap='YlOrRd',    # 颜色映射
    # annot=True,       # 显示数值
    # fmt=".2f",        # 数值格式（标准化后用浮点数）
    linewidths=0.5,   # 单元格边框
    cbar_kws={'label': 'Normalized Value'}
)
plt.title("Heatmap of 7 Topic 1 Over Years", pad=20)
plt.xlabel("Year")
plt.ylabel("Dimensions")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('Heatmap_7_Topic1_Over_Years.png', dpi=600)
plt.show()

# ====================== 4. 进阶可视化：堆叠子图 ======================
fig, axes = plt.subplots(7, 1, figsize=(14, 16), sharex=True)
for i, dim in enumerate(df.columns):
    sns.heatmap(
        df[[dim]].T,  # 每个维度单独成图
        ax=axes[i],
        cmap='YlOrRd',
        annot=True,
        fmt="d",
        cbar=False,
        linewidths=0.5
    )
    axes[i].set_title(f"Dimension {i+1}: {dim}")
    axes[i].set_yticks([])  # 隐藏 Y 轴标签

plt.suptitle("Trend of Each Dimension Over Years", y=1.02)
plt.tight_layout()
plt.savefig('Heatmap_7_Topic1_Over_Years_single.png', dpi=600)
plt.show()

# ====================== 5. 交互式热力图（Plotly） ======================
try:
    import plotly.graph_objects as go
    fig = go.Figure(
        data=go.Heatmap(
            z=df_normalized.values,  # 标准化后的数据
            x=df.index,
            y=df.columns,
            colorscale='Viridis',
            colorbar=dict(title="Normalized Value")
        )
    )
    fig.update_layout(
        title="Interactive Heatmap (Hover to See Values)",
        xaxis_title="Year",
        yaxis_title="Dimensions",
        width=800,
        height=500
    )
    fig.show()
except ImportError:
    print("Plotly not installed. Run `pip install plotly` to enable interactive plots.")