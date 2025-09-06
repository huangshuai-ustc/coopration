import matplotlib.pyplot as plt

# 数据
years = [1999, 2006, 2008, 2013, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
values = [1, 1, 1, 1, 1, 2, 2, 2, 6, 3, 5, 7, 21, 26]

# 创建图表
plt.figure(figsize=(14, 7))  # 调整画布大小
plt.plot(years, values, marker='o', linestyle='-', color='blue', linewidth=2, markersize=8, label='Value Trend')

# 添加标题和标签
plt.title('Algorithm Perceptions, Attitudes & Psychological Mechanisms\nTrend of Values (1985-2025)', fontsize=16, pad=20)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Value', fontsize=12)

# 优化X轴刻度（显示所有年份）
plt.xticks(years, rotation=45)  # 旋转45度避免重叠
plt.grid(axis='y', linestyle='--', alpha=0.7)  # 添加水平网格线

# 标注关键点（如最小值、最大值）
min_year, min_val = min(zip(years, values), key=lambda x: x[1])
max_year, max_val = max(zip(years, values), key=lambda x: x[1])
plt.annotate(f'Min: {min_val} ({min_year})',
             xy=(min_year, min_val),
             xytext=(min_year, min_val + 1),
             arrowprops=dict(arrowstyle='->'), fontsize=10)
plt.annotate(f'Max: {max_val} ({max_year})',
             xy=(max_year, max_val),
             xytext=(max_year, max_val + 2),
             arrowprops=dict(arrowstyle='->'), fontsize=10)

# 添加图例
plt.legend(loc='upper left')

# 自动调整布局
plt.tight_layout()

plt.savefig('Algorithm_Perceptions_Attitudes_Psychological_Mechanisms_Trend.png', dpi=600)
# 显示图表
plt.show()