import pandas as pd
from collections import defaultdict as defaultdict

file_path = "../classify/classify_results_new_with_counts.xlsx"
file_path2 = "../classify/classify_results_new.xlsx"

count = defaultdict(set)
df = pd.read_excel(file_path)
# for idx, row in df.iterrows():
#     key = (row['topic1_index'], row['topic2_index'], row['topic3_index'])
#     count[key] += 1

# for idx, row in df.iterrows():
#     key = row['topic2_content']
#     count[key] = row['topic2_count']

for idx, row in df.iterrows():
    key = row['excel_row']
    count[key].add(row['topic1_content'])

t = 0
for key, value in count.items():
    t -= -1
    print(f"{key}: {list(value)}")

print(t)