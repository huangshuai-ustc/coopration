import pandas as pd

# 读取第一个Excel文件（假设第二列是 'phrase'）
file1_path = 'matched_by_row_number.xlsx'
df1 = pd.read_excel(file1_path)
phrases = df1.iloc[:, 1]  # 第二列（索引1）

# 读取第二个Excel文件
file2_path = '../classify/auto_filtered_phrases_output_0.82.xlsx'
df2 = pd.read_excel(file2_path)
target_column = df2.iloc[:, 3]  # 第四列（索引3）

# 提取第二个文件的行号和对应第四列的值（行号从1开始）
max_row = len(df2)
results = []

for phrase in phrases:
    try:
        row_num = int(phrase) + 1  # 尝试将短语转换为行号
        if 1 <= row_num <= max_row:
            value = df2.iloc[row_num - 1, 3]  # 第四列（索引3）
            results.append(value)
        else:
            results.append("Out of range")
    except ValueError:
        results.append("Invalid row number")

# 将结果添加到第一个DataFrame
df1['matched_value'] = results

# 保存到新Excel文件
output_path = 'matched_by_row_number.xlsx'
df1.to_excel(output_path, index=False)

print(f"匹配完成，结果已保存到 {output_path}")