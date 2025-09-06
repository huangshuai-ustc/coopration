import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_53_balls_visualization(results_file_path):
    # 读取数据
    result_df = pd.read_excel(results_file_path)
    row_attributes = sorted(result_df['行属性'].unique())
    col_attributes = sorted(result_df['列属性'].unique())

    # 建立网格中心: 3x3（按你原逻辑）
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
            'x': 0.0,
            'y': 0.0
        })

    # 排布小球（避免初始重叠）
    phrase_positions = arrange_circles_strict(phrase_positions, grid_centers)

    # 生成可拖拽的交互式HTML
    create_interactive_html(phrase_positions, row_attributes, col_attributes)


def arrange_circles_strict(phrase_positions, grid_centers, max_iter=1000, grid_half=0.35):
    """严格圆碰撞检测，避免小球重叠"""
    max_count = max(p['count'] for p in phrase_positions)

    for item in phrase_positions:
        item['radius'] = grid_half * (0.25 + 0.75 * item['count'] / max_count)

    # 初始放置在各自单元格内
    for item in phrase_positions:
        cx, cy = grid_centers[(item['row_attr'], item['col_attr'])]
        r = item['radius']
        item['x'] = cx + np.random.uniform(-grid_half + r, grid_half - r)
        item['y'] = cy + np.random.uniform(-grid_half + r, grid_half - r)

    # 迭代推开重叠（仅在同一单元格内）
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

    # 限制在各自单元格内
    for item in phrase_positions:
        cx, cy = grid_centers[(item['row_attr'], item['col_attr'])]
        r = item['radius']
        grid_half = 0.5  # 单元格半边长
        item['x'] = np.clip(item['x'], cx - grid_half + r, cx + grid_half - r)
        item['y'] = np.clip(item['y'], cy - grid_half + r, cy + grid_half - r)
        del item['radius']

    return phrase_positions


def create_interactive_html(phrase_positions, row_attributes, col_attributes):
    # 准备单一 trace 的逐点数组
    xs, ys, sizes, colors, texts, hovertexts, customdata = [], [], [], [], [], [], []
    all_counts = [p['count'] for p in phrase_positions]
    min_count = min(all_counts) if all_counts else 1
    max_count = max(all_counts) if all_counts else 1

    # 颜色映射：按 (row_attr, col_attr)
    unique_combos = sorted(set((p['row_attr'], p['col_attr']) for p in phrase_positions))
    colorscale = ['#4169E1', '#1E90FF', '#00CED1', '#32CD32', '#FFD700', '#FF8C00', '#FF4500', '#FF0000']
    color_map = {combo: colorscale[i % len(colorscale)] for i, combo in enumerate(unique_combos)}

    # 预计算每个点所在单元格边界（默认锁定单元格时使用）
    # x in [j, j+1]；y in [2 - i, 3 - i]，其中 i 是 row 索引（从 0 开始，自上而下）
    row_index = {ra: i for i, ra in enumerate(row_attributes)}
    col_index = {ca: j for j, ca in enumerate(col_attributes)}

    for p in phrase_positions:
        xs.append(float(p['x']))
        ys.append(float(p['y']))
        size = 10 + 50 * (p['count'] - min_count) / (max_count - min_count + 1e-10)
        sizes.append(size)
        col = color_map[(p['row_attr'], p['col_attr'])]
        colors.append(col)
        texts.append(str(p['count']))
        hovertexts.append(f"短语: {p['phrase']}<br>行属性: {p['row_attr']}<br>列属性: {p['col_attr']}<br>次数: {p['count']}")

        i = row_index[p['row_attr']]
        j = col_index[p['col_attr']]
        xmin, xmax = float(j), float(j + 1)
        ymin, ymax = float(2 - i), float(3 - i)

        # customdata: [phrase, row_attr, col_attr, count, xmin, xmax, ymin, ymax]
        customdata.append([p['phrase'], p['row_attr'], p['col_attr'], int(p['count']),
                           xmin, xmax, ymin, ymax])

    fig = make_subplots(rows=1, cols=1)

    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode='markers+text',
        marker=dict(size=sizes, color=colors, line=dict(color='black', width=1)),
        text=texts, textposition='middle center',
        textfont=dict(color='black', size=10, family="Arial Black"),
        hovertext=hovertexts, hoverinfo='text',
        customdata=customdata,
        name="phrases",
        showlegend=False,
        cliponaxis=True
    ))

    # 画 3x3 网格
    for i in range(4):
        fig.add_shape(type="line", x0=i, y0=0, x1=i, y1=3, line=dict(color="gray", width=1, dash="dot"))
        fig.add_shape(type="line", x0=0, y0=i, x1=3, y1=i, line=dict(color="gray", width=1, dash="dot"))

    # 行/列标签
    for i, row_attr in enumerate(row_attributes):
        fig.add_annotation(x=-0.2, y=2.5 - i + 0.5, text=row_attr, showarrow=False,
                           font=dict(size=12, color='black'), xanchor='right', yanchor='middle')
    for j, col_attr in enumerate(col_attributes):
        fig.add_annotation(x=j + 0.5, y=3.2, text=col_attr, showarrow=False,
                           font=dict(size=12, color='black'), xanchor='center', yanchor='bottom')

    # 颜色图例（按组合生成虚拟点）
    for combo in unique_combos:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=15, color=color_map[combo], line=dict(color='black', width=1)),
            name=f"{combo[0]} - {combo[1]}"
        ))

    fig.update_layout(
        title=dict(text='53 phrase frequency distribution visualization<br><sub>Ball size represents frequency, color represents attribute combination</sub>',
                   x=0.5, font=dict(size=20)),
        xaxis=dict(range=[-0.5, 3.5], showgrid=False, zeroline=False, showticklabels=False,
                   title_text='Level of Analysis'),
        yaxis=dict(range=[-0.5, 3.5], showgrid=False, zeroline=False, showticklabels=False,
                   title_text='Mechanism Type'),
        width=1000, height=900,
        legend=dict(title=dict(text='combination', font=dict(size=14)), itemsizing='constant', font=dict(size=10),
                    x=1.05, y=0.5, xanchor='left', yanchor='middle'),
        plot_bgcolor='white', paper_bgcolor='white',
        uirevision=True  # 防止拖动过程中视图被重置
    )

    html_file = "53个短语可视化_可拖动.html"
    fig.write_html(html_file, include_plotlyjs='cdn', full_html=True)

    # 注入前端拖拽 & 导出 CSV 的 JS/CSS
    drag_js = r"""
<script>
document.addEventListener("DOMContentLoaded", function() {
    const gd = document.querySelector('.js-plotly-plot');

    // UI：右上角控制条
    const bar = document.createElement('div');
    bar.style.position = 'fixed';
    bar.style.top = '12px';
    bar.style.right = '12px';
    bar.style.zIndex = 9999;
    bar.style.display = 'flex';
    bar.style.gap = '8px';

    const lockBtn = document.createElement('button');
    lockBtn.textContent = '锁定单元格';
    lockBtn.style.padding = '6px 10px';
    lockBtn.style.border = '1px solid #ccc';
    lockBtn.style.background = '#fff';
    lockBtn.style.cursor = 'pointer';
    lockBtn.style.borderRadius = '6px';

    const saveBtn = document.createElement('button');
    saveBtn.textContent = '保存CSV';
    saveBtn.style.padding = '6px 10px';
    saveBtn.style.border = '1px solid #ccc';
    saveBtn.style.background = '#fff';
    saveBtn.style.cursor = 'pointer';
    saveBtn.style.borderRadius = '6px';

    document.body.appendChild(bar);
    bar.appendChild(lockBtn);
    bar.appendChild(saveBtn);

    let lockToCell = true;
    lockBtn.onclick = () => {
        lockToCell = !lockToCell;
        lockBtn.textContent = lockToCell ? '锁定单元格' : '自由拖拽';
    };

    function clamp(v, a, b) { return Math.min(Math.max(v, a), b); }

    let hoverIdx = null;  // 当前悬停点 index
    let dragging = false;
    let dragIdx = null;

    // 用 hover 获取点索引
    gd.on('plotly_hover', function(ev){
        if (ev.points && ev.points.length) {
            hoverIdx = ev.points[0].pointIndex;
        }
    });
    gd.on('plotly_unhover', function(){
        hoverIdx = null;
    });

    // 鼠标按下开始拖拽（若悬停在点上）
    gd.addEventListener('mousedown', function(e){
        if (hoverIdx !== null) {
            dragging = true;
            dragIdx = hoverIdx;
            // 禁止浏览器选择文本，减少干扰
            document.body.style.userSelect = 'none';
        }
    });

    // 鼠标移动时更新该点坐标
    document.addEventListener('mousemove', function(e){
        if (!dragging || dragIdx === null) return;

        const fl = gd._fullLayout;
        const xaxis = fl.xaxis;
        const yaxis = fl.yaxis;

        // 计算相对绘图区的像素
        const rect = gd.getBoundingClientRect();
        const plotOffset = fl._plots.xy; // {x, y, width, height}
        const px = e.clientX - rect.left - plotOffset.x;
        const py = e.clientY - rect.top  - plotOffset.y;

        // 像素 -> 数据坐标
        const xVal = xaxis.p2l(px);
        const yVal = yaxis.p2l(py);

        // 读取数据数组并修改对应点
        const t = 0; // 我们把所有点放在 trace 0
        const xArr = gd.data[t].x.slice();
        const yArr = gd.data[t].y.slice();

        let newX = xVal;
        let newY = yVal;

        // 可选：限制在原单元格内（从 customdata 取范围）
        if (lockToCell) {
            const cd = gd.data[t].customdata[dragIdx];
            const xmin = +cd[4], xmax = +cd[5], ymin = +cd[6], ymax = +cd[7];
            newX = clamp(newX, xmin, xmax);
            newY = clamp(newY, ymin, ymax);
        } else {
            // 否则限制在坐标轴范围内
            const xmin = fl.xaxis.range[0], xmax = fl.xaxis.range[1];
            const ymin = fl.yaxis.range[0], ymax = fl.yaxis.range[1];
            newX = clamp(newX, Math.min(xmin,xmax), Math.max(xmin,xmax));
            newY = clamp(newY, Math.min(ymin,ymax), Math.max(ymin,ymax));
        }

        xArr[dragIdx] = newX;
        yArr[dragIdx] = newY;

        // 更新整条 trace（一次只改一个点的值）
        Plotly.restyle(gd, {x: [xArr], y: [yArr]}, [t]);
    });

    // 鼠标抬起结束拖拽
    document.addEventListener('mouseup', function(){
        if (dragging) {
            dragging = false;
            dragIdx = null;
            document.body.style.userSelect = '';
        }
    });

    // 导出 CSV：phrase,row_attr,col_attr,count,x,y
    saveBtn.onclick = function(){
        const t = 0;
        const data = gd.data[t];
        const rows = [['短语','行属性','列属性','总出现次数','x','y']];
        for (let i = 0; i < data.x.length; i++){
            const cd = data.customdata[i];
            rows.push([cd[0], cd[1], cd[2], cd[3], data.x[i], data.y[i]]);
        }
        const csv = rows.map(r => r.map(v => {
            const s = (v===null || v===undefined) ? '' : String(v);
            // 简单转义
            return /[",\n]/.test(s) ? '"' + s.replace(/"/g,'""') + '"' : s;
        }).join(',')).join('\n');

        const blob = new Blob([csv], {type: 'text/csv;charset=utf-8;'});
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = '拖拽后坐标.csv';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    };
});
</script>
"""

    with open(html_file, "r", encoding="utf-8") as f:
        html_content = f.read()

    # 注入到 </body> 之前
    html_content = html_content.replace("</body>", drag_js + "\n</body>")

    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"HTML 文件已保存为: {html_file}")


if __name__ == "__main__":
    results_file_path = "汇总处理结果.xlsx"
    create_53_balls_visualization(results_file_path)
