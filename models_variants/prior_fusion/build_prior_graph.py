"""从 Excel 构建先验知识图并映射到 51 变量索引"""
import pandas as pd
import numpy as np
import torch, os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "Prior Knowledge Graph")


def build_prior_graph(column_names):
    """
    读取三个 Excel 文件，映射节点名到 column_names 的索引，
    返回 prior_edge_index [2, E_prior] 和 prior_weights [E_prior]
    """
    # 读取
    nodes_df   = pd.read_excel(os.path.join(DATA_DIR, "Nodes.xlsx"))
    control_df = pd.read_excel(os.path.join(DATA_DIR, "Control Edges.xlsx"))
    process_df = pd.read_excel(os.path.join(DATA_DIR, "Process Edges.xlsx"))

    # 构建节点名 → CSV 列索引的映射
    name_to_idx = {}
    for node_name in nodes_df.iloc[:, 0].values:
        node_name = str(node_name).strip()
        # 在 column_names 中查找匹配
        for i, col in enumerate(column_names):
            if col.upper() == node_name.upper():
                name_to_idx[node_name] = i
                break

    print(f"Prior nodes matched: {len(name_to_idx)} / {len(nodes_df)}")
    print(f"  Matched: {list(name_to_idx.keys())}")

    # 合并两条边表
    all_edges = []
    for df in [control_df, process_df]:
        for _, row in df.iterrows():
            src_name = str(row["Source"]).strip()
            tgt_name = str(row["Target"]).strip()
            weight   = float(row["Weight"])
            if src_name in name_to_idx and tgt_name in name_to_idx:
                all_edges.append((name_to_idx[src_name], name_to_idx[tgt_name], weight))
            else:
                print(f"  Skip: {src_name} → {tgt_name} (unmatched)")

    print(f"Prior edges created: {len(all_edges)}")

    # 转为 tensor
    if len(all_edges) == 0:
        return torch.empty(2, 0, dtype=torch.long), torch.empty(0)

    src = [e[0] for e in all_edges]
    dst = [e[1] for e in all_edges]
    wgt = [e[2] for e in all_edges]

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    weights    = torch.tensor(wgt, dtype=torch.float32)

    # 添加自环（权重=1.0）
    n = len(column_names)
    self_loops = torch.stack([torch.arange(n), torch.arange(n)], dim=0)
    self_wgts  = torch.ones(n)
    edge_index = torch.cat([edge_index, self_loops], dim=1)
    weights    = torch.cat([weights, self_wgts])

    return edge_index, weights
