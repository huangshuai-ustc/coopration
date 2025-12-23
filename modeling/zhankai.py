import pandas as pd
from ast import literal_eval

# 配置路径
input_csv = "phrases_with_deepseek_analysis(batch).csv"       # 替换为你的文件名
output_csv = "phrases_with_deepseek_analysis(batch)_zhankai.csv"      # 输出文件名

# 读取 CSV 文件
df = pd.read_csv(input_csv)

# 获取倒数第二列名
target_column = df.columns[-2]

# 解析列表并展开
expanded_rows = []
for item in df[target_column]:
    try:
        items = literal_eval(item)
        if isinstance(items, list):
            expanded_rows.extend(items)
    except:
        continue  # 跳过解析失败的行

# 创建新 DataFrame 并保存
expanded_df = pd.DataFrame(expanded_rows, columns=["expanded_item"])
expanded_df.to_csv(output_csv, index=False)
print(f"✅ 已保存展开后的结果到：{output_csv}")
