import pandas as pd

# 读取三个 CSV 文件
df1 = pd.read_csv("no_AI.csv")
df2 = pd.read_csv("no_OM.csv")
df3 = pd.read_csv("no_psychology.csv")

# 获取每个文件的长度
n1, n2, n3 = len(df1), len(df2), len(df3)
total = n1 + n2 + n3

# 计算抽取数量（按比例）
target_total = 650
s1 = round(n1 / total * target_total)
s2 = round(n2 / total * target_total)
s3 = target_total - s1 - s2

# 随机采样
sample1 = df1.sample(n=s1, random_state=42)
sample2 = df2.sample(n=s2, random_state=42)
sample3 = df3.sample(n=s3, random_state=42)

# 合并并打乱顺序
result = pd.concat([sample1, sample2, sample3]).sample(frac=1, random_state=42)

# 保存结果
result.to_csv("not_ok.csv", index=False)

print(f"✅ 随机抽取完成：{s1} + {s2} + {s3} = {s1 + s2 + s3} 行，已保存为 not_ok.csv")

