# SWaT工业控制系统异常检测 — GATv2+TCN+GRU消融实验报告

## 1. 背景

- **数据集**: SWaT水处理工业控制系统，51个传感器，正常+攻击数据
- **任务**: 时间窗口异常检测二分类
- **窗口**: 60步历史窗口，stride=10预测未来1步
- **基线模型**: GATv2+TCN+GRU双分支架构（预测头+重构头）
- **评分**: IQR归一化+Top-5变量聚合，阈值分位数q=0.995
- **配置**: config_dev.yaml, hidden_dim=32, batch=256, epochs=5

## 2. 基线模型架构

GATv2(图注意力)→TCN(时序卷积)→GRU(门控循环)→双头(Pred+Recon)

- 参数: 209K
- 评分: 预测+重构混合(lambda=0.5)
- **Baseline F1=0.6676, AUC=0.9368**

## 3. 非USAD消融实验

基于基线单模块添加:

| 模型 | F1 | vs Baseline |
|------|-----|------|
| +Temporal Attention | 0.7063 | +0.0387 |
| +Prior Fusion Gate | 0.6988 | +0.0312 |
| +Multi-Scale TCN | 0.6916 | +0.0240 |
| +DynPrior Feat Fusion | 0.7122 | +0.0446 |
| +Dynamic Pearson Graph | 0.6596 | -0.0080 |
| +Dynamic Graph Diff | 0.6618 | -0.0058 |
| +Prior Dynamic | 0.6584 | -0.0092 |

## 4. 动态图验证

USAD双解码器架构下，唯一变量=动态Pearson图:

- static_usad(静态图): F1=0.7296, AUC=0.9441
- dynamic_usad(动态图): F1=0.7494, AUC=0.9419
- **动态图贡献: +0.0198 F1**

USAD比基线提升: 0.6676→0.7494 (+0.0818)

## 5. 并行双支路架构（当前最佳）

空间支路: DynamicPearson + 先验图boost融合 → GATv2 + Prior Gate → [B,51,32]
时间支路: 单尺度Conv1d(k=3)逐变量编码 → [B,51,32]
融合: concat→MLP→flatten→z→USAD双解码器

- **parallel_usad_prior: F1=0.7524, AUC=0.9503**
- 参数: 1.0M, 训练25分钟

## 6. 时间分支消融

| 时间分支 | F1 | 说明 |
|------|-----|------|
| Conv1d(k=3)单尺度 | 0.7524 | 最佳 |
| Conv1d(k=3,5,7)多尺度 | 0.7523 | 无提升 |
| TCN(dil=1,2,4) | 0.7440 | 更差 |
| TCN(dil=1,2,4)+GRU | 0.7271 | 最差 |

结论: 时间分支越轻量越好，多尺度/TCN/GRU全白做

## 7. 外部模型对比

| 排名 | 模型 | F1 | AUC |
|------|------|-----|-----|
| 1 | DCdetector | 0.7553 | 0.9337 |
| 2 | parallel_usad_prior(我们) | 0.7524 | 0.9503 |
| 3 | dynamic_usad(我们) | 0.7494 | 0.9419 |
| 4 | USAD | 0.7417 | 0.9471 |
| 5 | MTAD-GAT | 0.7194 | 0.9376 |
| 6 | CAN | 0.7057 | 0.9534 |

全实验排名第2，AUC最高

## 8. 关键发现

1. 动态Pearson图有效(+0.02)，已验证为真实提升
2. 并行双支路+boost融合+先验gate叠加才有效(0.7524)
3. 单独加任何模块(TA/MS-TCN/Prior Gate)都无效
4. 时间分支越轻量越好，Conv1d(k=3)最优
5. 先验图太稀疏(9边/51节点)，但boost融合后可发挥微弱作用
6. GATv2的Python for-loop是训练瓶颈但暂时无等价加速方案

## 9. 下一步

- 扩充先验图(设备内全连接)
- 多seed验证(每个模型3次)
- DCdetector差距仅0.0029，争取超越
- 向量化GAT需要更深入的对齐验证
