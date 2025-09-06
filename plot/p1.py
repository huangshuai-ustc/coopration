from graphviz import Digraph

def create_compact_tree_with_five_layers():
    tree = Digraph(
        name="Compact Human-AI Tree (5 Layers)",
        format="png",
        graph_attr={
            "rankdir": "TB",
            "size": "8,8",  # 适当增大画布以容纳五层
            "nodesep": "0.2",
            "ranksep": "0.4",
            "dpi": "600",
            "splines": "ortho"
        },
        node_attr={
            "shape": "box",
            "fontname": "Arial",
            "style": "rounded,filled",
            "fontsize": "8",  # 更小字体适应五层
            "penwidth": "0.5",
            "margin": "0.1,0.05"
        },
        edge_attr={"arrowsize": "0.4", "penwidth": "0.5"}
    )

    # 配色方案
    colors = {
        1: "#FF6B6B",  # 根节点
        2: "#4ECDC4",  # 第二层
        3: "#FFD166",  # 第三层
        4: "#A8E6CF",  # 第四层
        5: "#DCD0FF"   # 第五层
    }

    # 第一层：根节点
    tree.node("Root", label="Artificial Intelligence and Cognitive Psychology Perspectives on Decision Sciences in Operations Research", fillcolor=colors[1], color="transparent")

    # 第二层：7个分支（前3个详细，后4个省略+数量标注）
    all_branches = [
        ('Human–AI Cognition\nDecision Mechanisms', 3),  # 子节点数量=3
        ('Trust\nTransparency\nExplainability', 3),
        ('Emotion\nEmpathy\nSocial Acceptance', 3),
        ('Collaboration\nTeam Dynamics', 3),  # 省略2个子节点
        ('Ethics\nBias & Fairness', 3),       # 省略4个子节点
        ('Usability\nUser Experience', 3),    # 省略3个子节点
        ('Autonomy\nControl & Governance', 3) # 省略2个子节点
    ]

    # 详细展示的前3个分支
    for i, (topic, _) in enumerate(all_branches[:3], 1):
        node_id = f"B{i}"
        tree.node(node_id, label=topic, fillcolor=colors[2])
        tree.edge("Root", node_id, color="#999999")

    # 省略的后4个分支（标注数量，无连线延伸）
    for i, (topic, count) in enumerate(all_branches[3:], 4):
        node_id = f"B{i}"
        label = f'"... [{count}]"'  # 用引号包裹省略号
        tree.node(node_id, label=label, fillcolor=colors[2], shape="plaintext")  # 无边框
        tree.edge("Root", node_id, color="#999999", style="dashed")  # 虚线连接

    # 第三层：仅展开前3个分支的子节点
    third_layer_nodes = {
        "B1": ["Information Processing\nSituation Awareness", "Automation Reliance\nBiases", "Decision Aids\nCognitive Offloading"],
        # "B2": ["Trust Formation\nCalibration", "Explainability\nTransparency", "Governance\nAccountability"],
        # "B3": ["Emotional Reactions\nMental Workload", "Empathy\nModeling", "Social\nNorms"]
    }

    for parent, subtopics in third_layer_nodes.items():
        for idx, subtopic in enumerate(subtopics, 1):
            node_id = f"{parent}_C{idx}"
            tree.node(node_id, label=subtopic, fillcolor=colors[3])
            tree.edge(parent, node_id, color="#BBBBBB")

    # 第四层：仅展开B1_C1的子节点（其他用省略号）
    fourth_layer_nodes = {
        "B1_C1": ["Situation Awareness\nDynamic Representations", "Information Overload\nInadequacies", "Attention Allocation\nConflict Detection"],
        "B1_C2": ["...", "...", "..."],
        # "B1_C3": ["...", "...", "..."]
    }

    for parent, subtopics in fourth_layer_nodes.items():
        for idx, subtopic in enumerate(subtopics, 1):
            node_id = f"{parent}_D{idx}"
            if parent == "B1_C1" and idx <= 3:
                tree.node(node_id, label=subtopic, fillcolor=colors[4])
                tree.edge(parent, node_id, color="#CCCCCC")
            else:
                tree.node(node_id, label="...", fillcolor=colors[4], shape="plaintext")
                tree.edge(parent, node_id, color="#CCCCCC", style="dashed")

    # 第五层：仅展开B1_C1_D1的子节点（示例）
    tree.node("B1_C1_D1_E1", label="step man machine", fillcolor=colors[5])
    tree.edge("B1_C1_D1", "B1_C1_D1_E1", color="#DDDDDD")
    tree.node("B1_C1_D1_E2", label="...", fillcolor=colors[5], shape="plaintext")
    tree.edge("B1_C1_D1", "B1_C1_D1_E2", color="#DDDDDD", style="dashed")

    # 保存为高清图片
    tree.render("classification_tree", view=True, cleanup=True)

if __name__ == "__main__":
    create_compact_tree_with_five_layers()