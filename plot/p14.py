import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_53_balls_visualization(results_file_path):
    # 读取数据
    result_df = pd.read_excel(results_file_path)
    row_attributes = sorted(result_df['行属性'].unique())
    col_attributes = sorted(result_df['列属性'].unique())

    # 网格中心
    grid_centers = {}
    for i, row_attr in enumerate(row_attributes):
        for j, col_attr in enumerate(col_attributes):
            grid_centers[(row_attr, col_attr)] = (j + 0.5, 2.5 - i)

    # 初始化短语信息
    phrase_positions = []
    for _, row in result_df.iterrows():
        phrase_positions.append({
            'phrase': row['短语'],
            'row_attr': row['行属性'],
            'col_attr': row['列属性'],
            'count': row['总出现次数'],
            'x': 0,
            'y': 0
        })

    # 排布小球，严格避免重叠
    phrase_positions = arrange_circles_strict(phrase_positions, grid_centers)

    # 创建交互式HTML
    create_interactive_html(phrase_positions, row_attributes, col_attributes)


def arrange_circles_strict(phrase_positions, grid_centers, max_iter=1000, grid_half=0.35):
    """
    严格圆碰撞检测，确保每个球完全不重叠
    grid_half: 网格半宽
    """
    max_count = max(p['count'] for p in phrase_positions)

    # 半径根据频次计算，最大不超过网格大小
    for item in phrase_positions:
        item['radius'] = grid_half * (0.25 + 0.75 * item['count'] / max_count)

    # 初始化位置
    for item in phrase_positions:
        cx, cy = grid_centers[(item['row_attr'], item['col_attr'])]
        r = item['radius']
        item['x'] = cx + np.random.uniform(-grid_half + r, grid_half - r)
        item['y'] = cy + np.random.uniform(-grid_half + r, grid_half - r)

    # 迭代推开重叠
    for _ in range(max_iter):
        moved = False
        for (row_attr, col_attr) in grid_centers.keys():
            items = [p for p in phrase_positions if p['row_attr'] == row_attr and p['col_attr'] == col_attr]
            for i, item in enumerate(items):
                for j in range(i):
                    other = items[j]
                    dx = item['x'] - other['x']
                    dy = item['y'] - other['y']
                    dist = np.sqrt(dx ** 2 + dy ** 2)
                    min_dist = item['radius'] + other['radius']
                    if dist < min_dist:
                        if dist == 0:
                            dx = np.random.uniform(-0.01, 0.01)
                            dy = np.random.uniform(-0.01, 0.01)
                            dist = np.sqrt(dx ** 2 + dy ** 2)
                        factor = (min_dist - dist) / dist / 2
                        item['x'] += dx * factor
                        item['y'] += dy * factor
                        other['x'] -= dx * factor
                        other['y'] -= dy * factor
                        moved = True
        if not moved:
            break

    # 限制在网格内
    for item in phrase_positions:
        cx, cy = grid_centers[(item['row_attr'], item['col_attr'])]
        r = item['radius']
        item['x'] = np.clip(item['x'], cx - grid_half + r, cx + grid_half - r)
        item['y'] = np.clip(item['y'], cy - grid_half + r, cy + grid_half - r)
        del item['radius']

    return phrase_positions


def create_interactive_html(phrase_positions, row_attributes, col_attributes):
    fig = make_subplots(rows=1, cols=1)

    all_counts = [item['count'] for item in phrase_positions]
    min_count = min(all_counts) if all_counts else 1
    max_count = max(all_counts) if all_counts else 1

    unique_combinations = sorted(set((item['row_attr'], item['col_attr']) for item in phrase_positions))

    # 改进颜色：深蓝到红色渐变，更醒目
    colorscale = ['#4169E1', '#1E90FF', '#00CED1', '#32CD32', '#FFD700', '#FF8C00', '#FF4500', '#FF0000']
    color_map = {combo: colorscale[i % len(colorscale)] for i, combo in enumerate(unique_combinations)}

    for item in phrase_positions:
        count = item['count']
        combo = (item['row_attr'], item['col_attr'])
        color = color_map[combo]
        size = 10 + 50 * (count - min_count) / (max_count - min_count + 1e-10)

        fig.add_trace(go.Scatter(
            x=[item['x']], y=[item['y']], mode='markers+text',
            marker=dict(size=size, color=color, line=dict(color='black', width=1)),
            text=[str(count)],  # 所有小球都显示次数
            textposition='middle center',
            textfont=dict(color='black', size=10, family="Arial Black"),
            name=f"{item['phrase']}",
            texttemplate='%{text}',
            hovertext=f"短语: {item['phrase']}<br>行属性: {item['row_attr']}<br>列属性: {item['col_attr']}<br>次数: {count}",
            hoverinfo='text',
            showlegend=False
        ))

    # 网格线
    for i in range(4):
        fig.add_shape(type="line", x0=i, y0=0, x1=i, y1=3, line=dict(color="gray", width=1, dash="dot"))
        fig.add_shape(type="line", x0=0, y0=i, x1=3, y1=i, line=dict(color="gray", width=1, dash="dot"))

    # 行列标签
    for i, row_attr in enumerate(row_attributes):
        fig.add_annotation(x=-0.2, y=2.5 - i + 0.5, text=row_attr, showarrow=False, font=dict(size=12, color='black'),
                           xanchor='right', yanchor='middle')
    for j, col_attr in enumerate(col_attributes):
        fig.add_annotation(x=j + 0.5, y=3.2, text=col_attr, showarrow=False, font=dict(size=12, color='black'),
                           xanchor='center', yanchor='bottom')

    # 图例
    for combo in unique_combinations:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=15, color=color_map[combo], line=dict(color='black', width=1)),
            name=f"{combo[0]} - {combo[1]}"
        ))

    fig.update_layout(
        title=dict(text='53 phrase frequency distribution visualization<br><sub>Ball size represents frequency, color represents attribute combination</sub>', x=0.5, font=dict(size=20)),
        xaxis=dict(range=[-0.5, 3.5], showgrid=False, zeroline=False, showticklabels=False, title_text='Level of Analysis'),
        yaxis=dict(range=[-0.5, 3.5], showgrid=False, zeroline=False, showticklabels=False, title_text='Meachaism Type'),
        width=1000, height=900,
        legend=dict(title=dict(text='combination', font=dict(size=14)), itemsizing='constant', font=dict(size=10),
                    x=1.05, y=0.5, xanchor='left', yanchor='middle'),
        plot_bgcolor='white', paper_bgcolor='white'
    )

    total_phrases = len(phrase_positions)
    total_count = sum(item['count'] for item in phrase_positions)
    fig.add_annotation(x=3.8, y=2.8, text=f"总短语数: {total_phrases}", showarrow=False, font=dict(size=12),
                       xanchor='left')
    fig.add_annotation(x=3.8, y=2.6, text=f"总出现次数: {total_count}", showarrow=False, font=dict(size=12),
                       xanchor='left')
    fig.add_annotation(x=3.8, y=2.4, text=f"频次范围: {min_count}-{max_count}", showarrow=False, font=dict(size=12),
                       xanchor='left')

    fig.write_html("53个短语可视化_交互式.html")
    print("HTML文件已保存为 '53个短语可视化_交互式.html'")
    print(f"总短语数: {total_phrases}，总出现次数: {total_count}，频次范围: {min_count}-{max_count}")


if __name__ == "__main__":
    results_file_path = "汇总处理结果.xlsx"
    create_53_balls_visualization(results_file_path)
