import pandas as pd

# 读取分类结果
df = pd.read_excel("classify_results_new.xlsx")

# 一级标题数量
df["topic1_count"] = df.groupby("topic1_index")["phrase"].transform("count")

# 二级标题数量（分组条件：一级+二级）
df["topic2_count"] = df.groupby(["topic1_index", "topic2_index"])["phrase"].transform("count")

# 三级标题数量（分组条件：一级+二级+三级）
df["topic3_count"] = df.groupby(["topic1_index", "topic2_index", "topic3_index"])["phrase"].transform("count")

# 调整列顺序
new_order = [
    "phrase",
    "excel_row",
    "topic1_index",
    "topic1_count",  # 一级标题数量
    "topic2_index",
    "topic2_count",  # 二级标题数量
    "topic3_index",
    "topic3_count",  # 三级标题数量
    "topic1_content",
    "topic2_content",
    "topic3_content"
]
df = df[new_order]

# 保存结果
df.to_excel("classify_results_new_with_counts.xlsx", index=False)
print("已保存到 classify_results_with_counts.xlsx")
