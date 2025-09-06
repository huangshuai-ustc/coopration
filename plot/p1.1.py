from graphviz import Digraph

def create_compact_tree_with_five_layers():
    tree = Digraph(
        name="Compact Human-AI Tree (5 Layers)",
        format="png",
        graph_attr={
            "rankdir": "TB",
            "size": "12,12",
            "nodesep": "0.2",
            "ranksep": "0.5",
            "dpi": "600",
            "splines": "ortho"
        },
        node_attr={
            "shape": "box",
            "fontname": "Arial",
            "style": "rounded,filled",
            "fontsize": "8",
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
    tree.node("Root", label="Artificial Intelligence and Cognitive Psychology Perspectives on Decision Sciences in Operations Research", fillcolor="white", color="transparent")

    # 第二层：7个分支
    all_branches = [
        ('Trust\nTransparency\nExplainability', 3),
        ('Human–AI Cognition\nDecision Mechanisms', 3),
        ('Emotion\nEmpathy\nSocial Acceptance', 3),
        ('Collaboration\nTeam Dynamics', 3),
        ('Ethics\nBias & Fairness', 3),
        ('Usability\nUser Experience', 3),
        ('Autonomy\nControl & Governance', 3)
    ]

    # 展示前3个分支
    for i, (topic, _) in enumerate(all_branches[:3], 1):
        node_id = f"B{i}"
        tree.node(node_id, label=topic, fillcolor="white")
        tree.edge("Root", node_id, color="#999999")

    # 后4个分支显示省略号
    for i, (topic, count) in enumerate(all_branches[3:], 4):
        node_id = f"B{i}"
        label = f"... [{count}]"
        tree.node(node_id, label=label, fillcolor="white", shape="plaintext")
        tree.edge("Root", node_id, color="#999999", style="dashed")

    # 第三层：B1 子节点
    third_layer_nodes = {
        "B1": ["Trust Formation\nCalibration", "Explainability\nTransparency", "Governance\nAccountability"],
    }
    for parent, subtopics in third_layer_nodes.items():
        for idx, subtopic in enumerate(subtopics, 1):
            node_id = f"{parent}_C{idx}"
            tree.node(node_id, label=subtopic, fillcolor="white")
            tree.edge(parent, node_id, color="#BBBBBB")

    # 第四层：B1_C1 子节点（B1_C1_D2 下伸出省略号）
    fourth_layer_nodes = {
        "B1_C1": ["Decision Path\nInterpretability", "Uncertainty\nRisk Communication", "Model Transparency\nAuditability"],
        "B1_C2": ["...", "...", "..."],
        "B1_C3": ["...", "...", "..."]
    }
    for parent, subtopics in fourth_layer_nodes.items():
        for idx, subtopic in enumerate(subtopics, 1):
            node_id = f"{parent}_D{idx}"
            if parent == "B1_C1" and idx == 2:
                # 保留原内容
                tree.node(node_id, label=subtopic, fillcolor="white")
                tree.edge(parent, node_id, color="#CCCCCC")
                # 加一个伸出的省略号节点
                more_node_id = f"{node_id}_more"
                tree.node(more_node_id, label="...", shape="plaintext", fillcolor='white')
                tree.edge(node_id, more_node_id, color="#CCCCCC", style="dashed")
            elif parent == "B1_C1":
                tree.node(node_id, label=subtopic, fillcolor="white")
                tree.edge(parent, node_id, color="#CCCCCC")
            else:
                tree.node(node_id, label="...", fillcolor="white", shape="plaintext")
                tree.edge(parent, node_id, color="#CCCCCC", style="dashed")

    # 第五层：短语向右排列
    fifth_layer_phrases = [
        'appraisal potential technique',
        'decision aids training procedures',
        'training disuse rate decreased',
        'roles design creation operation maintenance',
        'calibration driving functions',
        'calibration transparency conditions',
        'concierges dialog interface embodiment',
        'communication design recommender systems',
        'integration rooted team centric concepts',
        'uncalibrated trust'
    ]

    # 创建一个 invisible 节点占位让短语向右排列
    tree.node("right_anchor", label="", shape="plaintext", width="0", height="0")
    tree.edge("B1_C1_D1", "right_anchor", style="invis")

    for idx, phrase in enumerate(fifth_layer_phrases, 1):
        node_id = f"E{idx}"
        tree.node(node_id, label=phrase, fillcolor='white')
        tree.edge("B1_C1_D1", node_id, color="#CCCCCC")

    # 保存为高清图片
    tree.render("classification_tree_right", view=True, cleanup=True)

if __name__ == "__main__":
    create_compact_tree_with_five_layers()
