import matplotlib.pyplot as plt
import numpy as np
import textwrap

# 数据准备（类别和数值）
categories = [
    "Explainability & Transparency",
    "Collaboration & Coordination",
    "Governance & Accountability",
    "Trust Formation & Calibration",
    "Augmentation & Immersion",
    "Empathy & Social Interaction",
    "Control & Intervention",
    "Role Allocation & Adaptation",
    "Psychological Mechanisms & Anthropomorphism",
    "Emotional Reactions & Mental Workload",
    "Experience & Interaction",
    "Cognitive & Decision Styles",
    "Decision Aids & Cognitive Offloading",
    "Social Signals & Norms",
    "Information Processing & Situation Awareness",
    "Algorithmic Attitudes",
    "Governance & Ethics",
    "Mental Models & Biases",
    "Social Acceptance & Resistance",
    "Automation Reliance & Biases"
]
quantities = [3960, 77, 896, 468, 89, 29, 462, 50, 73, 140, 24, 53, 10, 48, 18, 11, 6, 15, 4, 1]

# 按数值降序排序（可选）
sorted_data = sorted(zip(categories, quantities), key=lambda x: -x[1])
categories, quantities = zip(*sorted_data)

# 设置图形
plt.figure(figsize=(12, 10), dpi=300)
plt.style.use('ggplot')  # 使用内置样式
plt.rcParams['font.sans-serif'] = ['Arial']  # 统一字体

# 颜色渐变（根据数值大小）
colors = plt.cm.viridis(np.array(quantities) / max(quantities))

# 绘制横向条形图
bars = plt.barh(
    y=np.arange(len(categories)),
    width=quantities,
    color=colors,
    edgecolor='gray',
    linewidth=0.5
)

# 优化y轴标签（自动换行 + 右对齐）
plt.yticks(
    np.arange(len(categories)),
    [textwrap.fill(cat, 30) for cat in categories],  # 每行最多30字符
    fontsize=8,
    ha='right'
)

# 添加网格和标题
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.xlabel('Number of topic 2', fontsize=12, labelpad=10)
plt.title('Quantity statistics for Topic 2', fontsize=14, pad=20)

# 智能数值标签
for bar in bars:
    width = bar.get_width()
    if width > 1000:
        offset = width * 0.01
        color = 'white'
    else:
        offset = width + 30
        color = 'black'
    plt.text(
        offset,
        bar.get_y() + bar.get_height() / 2,
        f'{int(width):,}',  # 显示整数
        va='center',
        fontsize=7,
        color=color
    )

# 调整边距（避免标签被截断）
plt.subplots_adjust(left=0.3, right=0.95, top=0.95, bottom=0.05)

# 保存图片（可选）
plt.savefig('Quantity_statistics_for_Topic_2.png', bbox_inches='tight', dpi=300)
plt.show()