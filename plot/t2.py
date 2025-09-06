import pandas as pd
from collections import defaultdict

# 读取第一个Excel文件（假设文件名为file1.xlsx，短语在第一列）
file1_path = 'matched_by_row_number.xlsx'
df1 = pd.read_excel(file1_path)  # 假设没有表头，第一列命名为'phrase'

default_array = lambda: [0] * 7
year_topic = defaultdict(default_array)

classify_one = [
    'Human–AI Cognition & Decision Mechanisms',
    'Trust, Transparency & Explainability',
    'Emotion, Empathy & Social Acceptance',
    'Algorithm Perceptions, Attitudes & Psychological Mechanisms',
    'Human–AI Teaming & Control',
    'Human–AI Service & Psychological Mechanisms',
    'Human–AI Collaboration, Safety & Societal Governance'
]

for idx, row in df1.iterrows():
    year = row.iloc[11]  # 第二列（索引1）
    topic1_content = row.iloc[8]  # 第二列（索引1）
    for i in range(7):
        if topic1_content == classify_one[i]:
            year_topic[year][i] += 1

for k, v in year_topic.items():
    print(f"{k}: {v}")

