import pandas as pd

# 读取两个 CSV 文件
df1 = pd.read_csv("ok_data_enhancement.csv")
df2 = pd.read_csv("not_ok.csv")
df1['label'] = 1
df2['label'] = 0

# 合并并打乱顺序
df = pd.concat([df1, df2], ignore_index=True).sample(frac=1, random_state=42)

# 计算分割点（80%）
split_index = int(len(df) * 0.8)

# 划分数据集
df_train = df.iloc[:split_index]
df_test = df.iloc[split_index:]

# 保存为两个 CSV 文件
df_train.to_csv("../train.csv", index=False)
df_test.to_csv("../eval.csv", index=False)

print(f"✅ 合并并划分完成：训练集 {len(df_train)} 行，测试集 {len(df_test)} 行")
