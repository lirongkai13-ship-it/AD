# CLAUDE.md

本文件为 Claude Code（claude.ai/code）在此仓库中工作时提供指导。

## 项目概述

SWaT 水处理工业控制系统异常检测基线，使用 **GATv2 + TCN + GRU** 双分支架构。模型联合训练预测头（预测未来时间步）和重构头（自编码输入窗口），然后通过 IQR 归一化的逐变量误差取 top-k 聚合来评分异常。

## 常用命令

```bash
# 训练
python train.py --config config.yaml

# 评估（需要训练生成的 best_model.pt）
python evaluate.py --config config.yaml

# 可视化（需要评估生成的 .npy 文件）
python plot_score.py --config config.yaml
```

输出目录由 config.yaml 中的 `output.save_dir` 决定，默认：`./outputs/swat_normal_train_merged_test/`。

## 模型架构

```
输入 [B, 60, 51]  ──转置──▶  [B, 51, 60]   每个变量=图节点, 60步时序特征
                                      │
   GATv2Block (2层, heads=2→1)        │  残差连接 + LayerNorm
     Layer 1: in=60 → out=48×2=96     │  基于 Pearson 相关图的边
     Layer 2: in=96 → out=48 (均值)   │
                                      ▼
                                 [B, 51, 48]   每个变量的空间编码
                                      │ 转置
   TCN (2个Block, dilation=1→2)      ▼
     沿变量维做因果卷积                [B, 48, 51] → [B, 48, 51]
                                      │ 转置
   GRU (1层, 单向)                   ▼
     处理变量序列                     [B, 51, 48] → h_last [B, 48]
                                      │
                   ┌──────────────────┴──────────────────┐
                   ▼                                     ▼
   Pred Head:  Linear→ReLU→Dropout→Linear      Recon Head: 相同结构
   → [B, 51]  (预测 t+1 时刻)                  → [B, 60, 51] (重构窗口)
```

**异常评分流程**（`evaluate.py` 中实现）：
1. 逐变量误差：`λ·|pred - y| + (1-λ)·mean(|recon - y|, dim=时间维)`
2. IQR 归一化（参数仅在验证集正常数据上拟合）
3. Top-k 聚合（k=5）→ 每个样本的异常分数
4. 阈值 = 验证集正常分数在 q 分位数处的值（q=0.995）

## 关键设计决策

- **不依赖 torch_geometric**：`ManualGATv2Layer` 用纯 PyTorch 实现 GATv2 注意力，用 Python 循环做逐节点 softmax 和逐边消息聚合（~50 节点、~400 边时可接受）。
- **严格防数据泄露**：StandardScaler、Pearson 图和 IQR 参数**仅在训练集正常数据上拟合**（不用验证集，不用 merged 测试集）。
- **惰性窗口生成**：`SWaTDynamicWindowDataset` 在 `__getitem__` 中通过 numpy 切片动态生成滑动窗口，内存占用 O(T×N) 而非 O(M×W×N)。旧的 `SWaTWindowDataset`（提前物化所有窗口）和 `make_windows()` 仍保留作为工具函数，但主流程已不再使用。
- **数据划分**：`normal.csv` → 按时间顺序 80/20 划分训练/验证集（不打乱）。`merged.csv`（正常+攻击）仅用于最终测试。
- **图构建**：在训练集正常数据上计算 Pearson 相关系数，阈值 0.3，始终包含自环。阈值过高导致无边时，退回到仅保留自环。

## 配置参数说明 (config.yaml)

| 分类 | 关键参数 | 说明 |
|------|---------|------|
| `data` | `window_size: 60`, `stride: 10`, `horizon: 1` | stride=10 加速训练，完整精度用 1 |
| `data` | `corr_threshold: 0.3`, `label_mode: "future"` | future 模式使分数与攻击点精确对齐 |
| `model` | `hidden_dim: 48`, `gat_heads: 2`, `gru_hidden: 48`, `tcn_channels: 48` | ~209K 参数（为速度缩减） |
| `train` | `batch_size: 256`, `epochs: 15`, `lr: 0.001` | ReduceLROnPlateau (patience=3), EarlyStopping (patience=5) |
| `score` | `topk: 5`, `threshold_quantile: 0.995` | 取误差最大的 top-5 变量聚合为异常分数 |
| `output` | `save_dir` | 所有输出（模型、指标、图表）保存位置 |

## Dataset 变体

`data_loader.py` 中支持两种 Dataset 类：
- `SWaTDynamicWindowDataset`（当前使用）：惰性窗口生成，内存 O(T×N)
- `SWaTWindowDataset`（遗留）：提前物化所有窗口，内存 O(M×W×N)，仅适合小数据集
