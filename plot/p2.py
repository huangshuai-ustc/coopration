import matplotlib.pyplot as plt
import numpy as np
import textwrap

# 数据准备
categories = [
    "Human–AI Cognition & Decision Mechanisms",
    "Trust, Transparency & Explainability",
    "Emotion, Empathy & Social Acceptance",
    "Algorithm Perceptions, Attitudes & Psychological Mechanisms",
    "Human–AI Teaming & Control",
    "Human–AI Service & Psychological Mechanisms",
    "Human–AI Collaboration, Safety & Societal Governance"
]
quantities = [29, 5324, 173, 79, 589, 186, 54]

# 设置高清分辨率和图形尺寸
plt.figure(figsize=(14, 8), dpi=300)

# 使用 matplotlib 内置样式（替代 seaborn）
plt.style.use('ggplot')  # 也可用 'default', 'bmh', 'fivethirtyeight' 等

# 创建横板柱状图（带颜色渐变）
colors = plt.cm.Blues(np.array(quantities) / max(quantities))
bars = plt.barh(
    y=np.arange(len(categories)),
    width=quantities,
    color=colors,
    edgecolor='gray',
    linewidth=0.7
)

# 优化y轴标签（自动换行长文本）
plt.yticks(
    np.arange(len(categories)),
    [textwrap.fill(cat, 40) for cat in categories],  # 每行最多40字符
    fontsize=9,
    ha='right'
)

# 添加网格和标签
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.xlabel('Number of topic 1', fontsize=12, labelpad=10)
plt.title('Quantity statistics for Topic 1', fontsize=14, pad=20)

# 智能数值标签
for bar in bars:
    width = bar.get_width()
    if width > 1000:
        offset = width * 0.02
        color = 'white'
    else:
        offset = width + 50
        color = 'black'
    plt.text(
        offset,
        bar.get_y() + bar.get_height() / 2,
        f'{width:,}',
        va='center',
        fontsize=8,
        color=color
    )

# 调整边距
plt.subplots_adjust(left=0.35, right=0.95, top=0.9, bottom=0.1)

# 保存或显示
plt.savefig('Quantity_statistics_for_Topic_1.png', bbox_inches='tight', dpi=300)  # 可选保存
plt.show()