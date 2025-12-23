import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 数据
years = [
    1969, 1971, 1973, 1981, 1982, 1985, 1986, 1987, 1990, 1991, 1993, 1994, 1995, 1996, 1997, 1999, 2000,
    2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017,
    2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025
]
values = [
    9, 8, 3, 7, 18, 3, 16, 4, 8, 12, 7, 7, 6, 27, 23, 20, 66, 30, 35, 1, 35, 15, 15, 9, 12, 29, 27, 19,
    29, 24, 47, 57, 150, 124, 157, 279, 265, 406, 450, 936, 1661, 268
]

# 创建图表
plt.figure(figsize=(14, 7))  # 调整画布大小
plt.plot(years, values, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)

# 添加标题和标签
plt.title('Trust Transparency Explainability\nTrend of Values Over Years (1969-2025)', fontsize=16, pad=20)
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
             xytext=(min_year, min_val + 0.5),
             arrowprops=dict(arrowstyle='->'), fontsize=10)
plt.annotate(f'Max: {max_val} ({max_year})',
             xy=(max_year, max_val),
             xytext=(max_year, max_val + 0.5),
             arrowprops=dict(arrowstyle='->'), fontsize=10)

# 添加图例
plt.legend(loc='upper left')

# 自动调整布局
plt.tight_layout()
plt.savefig('Trust_Transparency_Explainability_Trend.png', dpi=600)
# 显示图表
plt.show()