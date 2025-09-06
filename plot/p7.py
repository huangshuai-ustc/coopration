import matplotlib.pyplot as plt

# 数据
years = [1985, 1997, 2000, 2001, 2003, 2004, 2005, 2007, 2010, 2011, 2013, 2016, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
values = [1, 1, 3, 3, 1, 2, 1, 1, 3, 1, 3, 1, 9, 10, 11, 6, 12, 33, 54, 17]

# 创建图表
plt.figure(figsize=(14, 7))  # 调整画布大小
plt.plot(years, values, marker='o', linestyle='-', color='blue', linewidth=2, markersize=8, label='Value Trend')

# 添加标题和标签
plt.title('Emotion, Empathy & Social Acceptance\nTrend of Values (1985-2025)', fontsize=16, pad=20)
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

plt.savefig('Emotion_Empathy_Social_Acceptance_Trend.png', dpi=600)
# 显示图表
plt.show()