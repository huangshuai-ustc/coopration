import pandas as pd
import json
from collections import defaultdict

# 读取Excel
df = pd.read_excel("classify_result_by_xinyue.xlsx")

# 生成分组字典
grouped_dict = defaultdict(list)

for (t1, t2, t3), group in df.groupby(["topic1_index", "topic2_index", "topic3_index"]):
    key = f"{t1},{t2},{t3}"
    phrases = group["phrase"].tolist()
    excel_rows = group["excel_row"].tolist()
    for i in range(len(phrases)):
        grouped_dict[key].append([phrases[i].strip(), excel_rows[i]])


for key, value in grouped_dict.items():
    if len(value) > 100:
        print(f"Key {key} has {len(value)} items, which is more than 100.")

# 保存为一行 JSON
with open("phrases_by_topic_by_xinyue.json", "w", encoding="utf-8") as f:
    json.dump(grouped_dict, f, ensure_ascii=False, separators=(',', ':'))

print("已生成 phrases_by_topic.json (一行格式)")
