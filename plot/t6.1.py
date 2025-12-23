import pandas as pd
import numpy as np
from collections import defaultdict


def process_excel_files(file1_path, file2_path):
    # 读取第一个Excel文件
    df1 = pd.read_excel(file1_path, header=None)

    # 读取第二个Excel文件
    df2 = pd.read_excel(file2_path)

    # 获取第一个Excel的行属性和列属性
    row_attributes = df1.iloc[1:4, 0].tolist()  # 第2-4行，第1列
    col_attributes = df1.iloc[0, 1:4].tolist()  # 第1行，第2-4列

    # 构建短语到属性的映射字典
    phrase_to_attributes = {}

    # 遍历第一个Excel的3x3区域（第2-4行，第2-4列）
    for i in range(1, 4):  # 行索引 1,2,3
        for j in range(1, 4):  # 列索引 1,2,3
            cell_content = str(df1.iloc[i, j])
            if cell_content and cell_content != 'nan':
                phrases = [p.strip() for p in cell_content.split(',') if p.strip()]
                row_attr = row_attributes[i - 1]  # 对应行属性
                col_attr = col_attributes[j - 1]  # 对应列属性

                for phrase in phrases:
                    phrase_to_attributes[phrase] = (row_attr, col_attr)

    # 处理第二个Excel文件 - 汇总所有行的结果
    # 使用字典来汇总每个短语的总出现次数
    phrase_summary = defaultdict(lambda: {'row_attr': None, 'col_attr': None, 'total_count': 0})

    # 获取R列到BR列的列名（共53列）
    phrase_columns = df2.columns[df2.columns.get_loc('cognitive_offload'):df2.columns.get_loc('psychological_expansion') + 1]

    # 遍历每一行
    for index, row in df2.iterrows():
        # 遍历每个短语列
        for col_name in phrase_columns:
            value = row[col_name]
            # 检查值是否为1,2,3（存在）
            if value in [1, 2, 3]:
                phrase = col_name  # 列名就是短语名称
                occurrence = value  # 出现次数

                # 查找短语对应的属性
                if phrase in phrase_to_attributes:
                    row_attr, col_attr = phrase_to_attributes[phrase]
                    # 如果是第一次遇到这个短语，记录属性
                    if phrase_summary[phrase]['row_attr'] is None:
                        phrase_summary[phrase]['row_attr'] = row_attr
                        phrase_summary[phrase]['col_attr'] = col_attr
                    # 累加出现次数
                    phrase_summary[phrase]['total_count'] += occurrence
                else:
                    print(f"警告: 短语 '{phrase}' 在第一个Excel中未找到")

    # 转换为结果列表
    results = []
    for phrase, data in phrase_summary.items():
        if data['row_attr'] is not None:  # 只包含找到属性的短语
            results.append((phrase, data['row_attr'], data['col_attr'], data['total_count']))

    return results


# 使用示例
if __name__ == "__main__":
    file1_path = "../stage3/九宫格.xlsx"
    file2_path = "../stage3/processed_file.xlsx"

    results = process_excel_files(file1_path, file2_path)

    # 输出汇总结果
    print("汇总结果（所有行的总计）:")
    print("-" * 60)
    for result in results:
        print(f"短语: {result[0]:<15} | 行属性: {result[1]:<10} | 列属性: {result[2]:<10} | 总出现次数: {result[3]}")

    # 统计信息
    print(f"\n总计: 共找到 {len(results)} 个短语")
    print(f"总出现次数: {sum(result[3] for result in results)}")

    # 保存到Excel文件
    result_df = pd.DataFrame(results, columns=['短语', '行属性', '列属性', '总出现次数'])
    result_df.to_excel("汇总处理结果.xlsx", index=False)
    print("\n结果已保存到 '汇总处理结果.xlsx'")