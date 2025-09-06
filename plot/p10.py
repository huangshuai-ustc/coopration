import pandas as pd
import matplotlib.pyplot as plt

# 1. 读取Excel文件
file_path = "../stage3/processed_file.xlsx"  # 替换为你的Excel文件路径
df = pd.read_excel(file_path, usecols="R:BR")  # 读取R列到BR列（共53列）

# 2. 统计每列中0、1、2的数量
stats = pd.DataFrame()
for col in df.columns:
    counts = df[col].value_counts().reindex([1, 2, 3], fill_value=0)  # 确保0/1/2都统计（即使缺失）
    stats[col] = counts

# 转置数据：行=列名（R,S,...），列=类别（0,1,2）
stats = stats.T
stats.columns = ["low", "medium", "high"]  # 重命名列

# 3. 绘制横向堆叠条形图
fig, ax = plt.subplots(figsize=(13, 13))  # 宽度8，高度13（可调整）
stats.plot(
    kind="barh",
    stacked=True,
    color=["#4C72B0", "#55A868", "#C44E52"],
    edgecolor="black",
    ax=ax
)

# 图表美化
ax.set_title("Statistical Chart of Frequency of Research Direction and Viewpoints", pad=20, fontsize=12)
ax.set_xlabel("Count", labelpad=10)
ax.set_ylabel("viewpoint", labelpad=10)
ax.set_yticklabels(ax.get_yticklabels(), ha="right", fontsize=10)  # 字号改小
ax.legend(title="Category", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()

# 4. 保存或显示图表
plt.savefig("Statistical_Chart_of_Frequency_of_Research_Direction_and_Viewpoints.png", dpi=600, bbox_inches="tight")
plt.show()