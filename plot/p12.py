from graphviz import Digraph

def create_full_tree_with_five_layers():
    tree = Digraph(
        name="Full Human-AI Tree (5 Layers)",
        format="png",
        graph_attr={
            "rankdir": "TB",
            # "size": "8,8",
            "nodesep": "0.2",
            "ranksep": "0.4",
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
    tree.node("Root", label="Human-AI\nInteraction", fillcolor=colors[1], color="transparent")

    # 第二层：7个分支，全展开
    all_branches = [
        'Human–AI Cognition\nDecision Mechanisms',
        'Trust\nTransparency\nExplainability',
        'Emotion\nEmpathy\nSocial Acceptance',
        'Collaboration\nTeam Dynamics',
        'Ethics\nBias & Fairness',
        'Usability\nUser Experience',
        'Autonomy\nControl & Governance'
    ]

    for i, topic in enumerate(all_branches, 1):
        node_id = f"B{i}"
        tree.node(node_id, label=topic, fillcolor=colors[2])
        tree.edge("Root", node_id, color="#999999")

    # 第三层：定义为列表，便于替换
    third_layer_nodes = {
        "B1": ["Info Processing", "Automation Reliance", "Decision Aids"],
        "B2": ["Trust Formation", "Explainability", "Accountability"],
        "B3": ["Emotional Reactions", "Empathy", "Social Norms"],
        "B4": ["Collaboration Models", "Team Roles", "Shared Goals"],
        "B5": ["Bias Mitigation", "Fairness", "Ethics Rules"],
        "B6": ["Usability Testing", "UX Design", "Accessibility"],
        "B7": ["Autonomy Levels", "Control Mechanisms", "Governance"]
    }

    for parent, subtopics in third_layer_nodes.items():
        for j, subtopic in enumerate(subtopics, 1):
            node_id = f"{parent}_C{j}"
            tree.node(node_id, label=subtopic, fillcolor=colors[3])
            tree.edge(parent, node_id, color="#BBBBBB")

    # 第四层：每个第三层节点都展开三个
    fourth_layer_nodes = {}
    for parent, subtopics in third_layer_nodes.items():
        for j, _ in enumerate(subtopics, 1):
            child_id = f"{parent}_C{j}"
            fourth_layer_nodes[child_id] = [
                f"{child_id} - Detail1",
                f"{child_id} - Detail2",
                f"{child_id} - Detail3"
            ]

    for parent, subtopics in fourth_layer_nodes.items():
        for k, subtopic in enumerate(subtopics, 1):
            node_id = f"{parent}_D{k}"
            tree.node(node_id, label=subtopic, fillcolor=colors[4])
            tree.edge(parent, node_id, color="#CCCCCC")

    # 第五层：示例，仅展开 B1_C1_D1
    tree.node("B1_C1_D1_E1", label="Step Man Machine", fillcolor=colors[5])
    tree.edge("B1_C1_D1", "B1_C1_D1_E1", color="#DDDDDD")
    for e in range(2, 4):
        node_id = f"B1_C1_D1_E{e}"
        tree.node(node_id, label=f"... {e}", fillcolor=colors[5])
        tree.edge("B1_C1_D1", node_id, color="#DDDDDD", style="dashed")

    # 保存为高清图片
    tree.render("full_classification_tree", view=True, cleanup=True)

if __name__ == "__main__":
    create_full_tree_with_five_layers()
