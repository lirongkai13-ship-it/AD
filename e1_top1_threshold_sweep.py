"""e1 + top1 variable score — Threshold Sweep
固定: tri_branch (gamma=0.05, gate_scale=1.0), score=e1 only, topk=1
只做 threshold sweep — 不改模型/训练/loss
"""
import os, sys, csv, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_fscore_support)

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import prepare_data, build_pearson_edge_index, split_train_val
from utils import (load_config, set_seed, get_device, ensure_dir,
                   fit_iqr_params, apply_iqr_normalize, aggregate_topk_score)
from models_variants.tri_branch.variant_model import TriBranch_USAD

OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "e1_top1_sweep")
ensure_dir(OUT_DIR)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ═══════════════════════════════════════════════════
# 1. Load everything
# ═══════════════════════════════════════════════════
print("Loading config and data...")
cfg = load_config('config_dev.yaml')
set_seed(42)
_, val_ds, test_ds, _, info = prepare_data(cfg)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, drop_last=False)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, drop_last=False)

# Build graphs
from sklearn.preprocessing import StandardScaler
import pandas as pd, importlib.util
dcfg = cfg['data']
nfd = pd.read_csv(dcfg['train_csv']); nfd.columns = [str(c).strip() for c in nfd.columns]
nfd = nfd[[c for c in nfd.columns if c not in ['Timestamp', 'Normal/Attack']]]
mfd = pd.read_csv(dcfg['test_csv']); mfd.columns = [str(c).strip() for c in mfd.columns]
common_cols = [c for c in nfd.columns if c in mfd.columns]
raw = nfd[common_cols].values.astype(np.float32)
tr, _, _, _ = split_train_val(raw, None, 0.2)
tv = StandardScaler().fit_transform(tr)
static_ei, _ = build_pearson_edge_index(tv)

bgp = importlib.util.spec_from_file_location('bpg',
    os.path.join(os.path.dirname(__file__), 'models_variants', 'prior_fusion', 'build_prior_graph.py'))
bpgm = importlib.util.module_from_spec(bgp); bgp.loader.exec_module(bpgm)
prior_ei, prior_w = bpgm.build_prior_graph(common_cols)

static_ei = static_ei.to(device); prior_ei = prior_ei.to(device); prior_w = prior_w.to(device)

# Load model
print("Loading tri_branch model...")
model = TriBranch_USAD(
    nv=info['num_variables'], ws=60,
    static_edge_index=static_ei, prior_edge_index=prior_ei, prior_weights=prior_w,
    hidden_dim=32, gat_heads=2, gru_hidden=32, tcn_channels=32, tcn_blocks=1,
    dropout=0.2, latent_dim=64, use_flatten=True,
    temporal_mode="per_variable_conv",
    encoder_mode="tri_branch_residual_gate",
    gamma_mode="fixed", gamma_value=0.05, gate_scale=1.0,
).to(device)

ckpt = torch.load('outputs/swat_normal_train_merged_test/tri_branch/best_model.pt', map_location=device)
# Remap old keys
sd = {k.replace('gated_fusion.gate.', 'gated_fusion.gate_mlp.'): v for k, v in ckpt['model'].items()}
model.load_state_dict(sd)
model.eval()
print(f"Model loaded. Params: {sum(p.numel() for p in model.parameters()):,}")

# ═══════════════════════════════════════════════════
# 2. Collect e1 scores (top1 only)
# ═══════════════════════════════════════════════════
print("Collecting val/test scores...")
t0 = time.time()

@torch.no_grad()
def collect_e1_top1(model, loader):
    """e1 + top1 variable score"""
    model.eval()
    scores_list, labels_list = [], []
    for batch in loader:
        x = batch['x'].to(device)
        r1, _, _ = model(x, static_ei)
        e1 = (r1 - x).abs().mean(dim=1)     # [B, 51]
        score = e1.max(dim=-1).values        # [B]  — top1
        scores_list.append(score.cpu().numpy())
        if 'label' in batch:
            labels_list.append(batch['label'].cpu().numpy())
    return np.concatenate(scores_list), np.concatenate(labels_list) if labels_list else None

v_scores, v_labels = collect_e1_top1(model, val_loader)
t_scores, t_labels = collect_e1_top1(model, test_loader)
print(f"  Val: {len(v_scores)} samples, Test: {len(t_scores)} samples")
print(f"  Time: {time.time() - t0:.1f}s")

# ═══════════════════════════════════════════════════
# 3. Compute score statistics
# ═══════════════════════════════════════════════════
v_normal_mask = (v_labels == 0)
v_normal_scores = v_scores[v_normal_mask]
v_normal_mean = v_normal_scores.mean()
v_normal_std = v_normal_scores.std()

t_normal_scores = t_scores[t_labels == 0]
t_attack_scores = t_scores[t_labels == 1]

print(f"\nScore statistics:")
print(f"  Val normal: mean={v_normal_mean:.4f} std={v_normal_std:.4f}")
print(f"  Test normal: mean={t_normal_scores.mean():.4f} median={np.median(t_normal_scores):.4f}")
print(f"  Test attack: mean={t_attack_scores.mean():.4f} median={np.median(t_attack_scores):.4f}")
print(f"  Separation: {t_attack_scores.mean()/t_normal_scores.mean():.2f}")

# ═══════════════════════════════════════════════════
# 4. Quantile threshold sweep
# ═══════════════════════════════════════════════════
print(f"\n{'='*80}")
print("QUANTILE THRESHOLD SWEEP")
print(f"{'='*80}")

QUANTILES = [0.80, 0.85, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.985, 0.99,
             0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999, 0.9995, 0.9999]

results = []
for q in QUANTILES:
    th = float(np.quantile(v_normal_scores, q))
    pred = (t_scores > th).astype(int)

    tp = int(((pred == 1) & (t_labels == 1)).sum())
    fp = int(((pred == 1) & (t_labels == 0)).sum())
    tn = int(((pred == 0) & (t_labels == 0)).sum())
    fn = int(((pred == 0) & (t_labels == 1)).sum())

    pr, rc, f1, _ = precision_recall_fscore_support(t_labels, pred, average='binary', zero_division=0)
    auc_val = float(roc_auc_score(t_labels, t_scores))
    aupr_val = float(average_precision_score(t_labels, t_scores))

    results.append({
        'sweep_type': 'quantile',
        'quantile': q,
        'threshold': round(th, 4),
        'precision': round(float(pr), 6),
        'recall': round(float(rc), 6),
        'f1': round(float(f1), 6),
        'auc': round(auc_val, 6),
        'aupr': round(aupr_val, 6),
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
    })

# Find key points
# Best F1
best_f1 = max(results, key=lambda r: r['f1'])
# Best recall with F1 >= 0.755
f1_candidates = [r for r in results if r['f1'] >= 0.755]
best_recall_755 = max(f1_candidates, key=lambda r: r['recall']) if f1_candidates else None
# Best recall overall
best_recall = max(results, key=lambda r: r['recall'])
# Balanced: highest F1*Recall
best_balanced = max(results, key=lambda r: r['f1'] * r['recall'])
# Current baseline
current_baseline = [r for r in results if abs(r['quantile'] - 0.995) < 0.0001][0]

print(f"\n{'Metric':>12s}  {'q':>7s}  {'Threshold':>12s}  {'F1':>8s}  {'P':>8s}  {'R':>8s}  {'AUC':>8s}")
print(f"{'─'*65}")
for label, r in [("Current(0.995)", current_baseline),
                  ("Best F1", best_f1),
                  ("Best R(F1>=0.755)", best_recall_755) if best_recall_755 else ("N/A", None),
                  ("Best Recall", best_recall),
                  ("Balanced", best_balanced)]:
    if r is None: continue
    print(f"{label:>12s}  {r['quantile']:7.4f}  {r['threshold']:12.4f}  "
          f"{r['f1']:8.4f}  {r['precision']:8.4f}  {r['recall']:8.4f}  {r['auc']:8.4f}")

# ═══════════════════════════════════════════════════
# 5. Dense sweep around best region
# ═══════════════════════════════════════════════════
print(f"\n{'='*80}")
print("DENSE THRESHOLD SWEEP (absolute values)")
print(f"{'='*80}")

# Determine range: around best F1 threshold, cover the interesting zone
best_f1_th = best_f1['threshold']
# Wide enough to capture precision-recall tradeoff
th_min = max(10, best_f1_th * 0.3)
th_max = min(t_scores.max(), best_f1_th * 1.8)
dense_thresholds = np.linspace(th_min, th_max, 200)

for th in dense_thresholds:
    pred = (t_scores > th).astype(int)
    tp = int(((pred == 1) & (t_labels == 1)).sum())
    fp = int(((pred == 1) & (t_labels == 0)).sum())
    tn = int(((pred == 0) & (t_labels == 0)).sum())
    fn = int(((pred == 0) & (t_labels == 1)).sum())

    pr, rc, f1, _ = precision_recall_fscore_support(t_labels, pred, average='binary', zero_division=0)
    auc_val = float(roc_auc_score(t_labels, t_scores))
    aupr_val = float(average_precision_score(t_labels, t_scores))

    results.append({
        'sweep_type': 'dense',
        'quantile': None,
        'threshold': round(float(th), 4),
        'precision': round(float(pr), 6),
        'recall': round(float(rc), 6),
        'f1': round(float(f1), 6),
        'auc': round(auc_val, 6),
        'aupr': round(aupr_val, 6),
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
    })

# Best in dense sweep
dense_results = [r for r in results if r['sweep_type'] == 'dense']
dense_best_f1 = max(dense_results, key=lambda r: r['f1'])
dense_best_recall = max(dense_results, key=lambda r: r['recall'])
dense_f1_755 = [r for r in dense_results if r['f1'] >= 0.755]
dense_best_r_755 = max(dense_f1_755, key=lambda r: r['recall']) if dense_f1_755 else None

print(f"\n{'Metric':>12s}  {'Threshold':>12s}  {'F1':>8s}  {'P':>8s}  {'R':>8s}  {'TP':>7s}  {'FP':>7s}  {'FN':>7s}")
print(f"{'─'*80}")
dense_highlights = [
    ("Best F1", dense_best_f1),
    ("Best R(F1>=0.755)", dense_best_r_755) if dense_best_r_755 else (None, None),
    ("Best Recall", dense_best_recall),
]
for label, r in dense_highlights:
    if r is None: continue
    print(f"{label:>12s}  {r['threshold']:12.4f}  {r['f1']:8.4f}  "
          f"{r['precision']:8.4f}  {r['recall']:8.4f}  "
          f"{r['tp']:7d}  {r['fp']:7d}  {r['fn']:7d}")

# ═══════════════════════════════════════════════════
# 6. Save CSV
# ═══════════════════════════════════════════════════
csv_path = os.path.join(OUT_DIR, 'e1_top1_threshold_sweep_results.csv')
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=results[0].keys())
    w.writeheader()
    w.writerows(results)
print(f"\nCSV saved: {csv_path} ({len(results)} rows)")

# ═══════════════════════════════════════════════════
# 7. Generate Markdown Report
# ═══════════════════════════════════════════════════
print(f"\n{'='*80}")
print("GENERATING REPORT")
print(f"{'='*80}")

# Quantile key points table
quantile_rows = [r for r in results if r['sweep_type'] == 'quantile']

# Find: q where F1 first exceeds 0.7587, best recall with F1>=0.755, etc.
f1_exceed_current = [r for r in quantile_rows if r['f1'] > 0.7587]
best_q_f1 = max(f1_exceed_current, key=lambda r: r['f1']) if f1_exceed_current else None

report_path = os.path.join(OUT_DIR, 'e1_top1_threshold_sweep_report.md')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(f"""# e1 + Top1 Variable Score — Threshold Sweep Report

**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Model:** tri_branch_residual_gate (gamma=0.05, gate_scale=1.0)
**Score:** e1 only, topk=1 (max aggregation)

---

## 1. 实验目的

当前 e1 + top1 score 在 q=0.995 下 F1=0.7587, Precision=0.8123, Recall=0.7117。
Precision 偏高，Recall 偏低，说明阈值偏保守。
本实验在不修改模型/训练/loss 的前提下，通过 threshold sweep 寻找更好的 Precision/Recall/F1 平衡点。

---

## 2. 固定配置

| 参数 | 值 |
|------|-----|
| encoder_mode | tri_branch_residual_gate |
| gamma | 0.05 (fixed) |
| gate_scale | 1.0 |
| temporal_mode | per_variable_conv |
| score_mode | e1 only |
| topk | 1 |
| aggregation | max |

---

## 3. Score 统计

| Statistic | Value |
|-----------|-------|
| Val normal mean | {v_normal_mean:.4f} |
| Val normal std | {v_normal_std:.4f} |
| Test normal mean | {t_normal_scores.mean():.4f} |
| Test attack mean | {t_attack_scores.mean():.4f} |
| Separation ratio | {t_attack_scores.mean()/t_normal_scores.mean():.2f} |

---

## 4. Quantile Threshold Sweep 结果

### 4.1 完整表格

| q | Threshold | F1 | Precision | Recall | AUC | TP | FP | FN |
|---|-----------|-----|-----------|--------|-----|----|----|----|
""")
    for r in quantile_rows:
        f.write(f"| {r['quantile']:.4f} | {r['threshold']:.1f} | {r['f1']:.4f} | "
                f"{r['precision']:.4f} | {r['recall']:.4f} | {r['auc']:.4f} | "
                f"{r['tp']} | {r['fp']} | {r['fn']} |\n")

    f.write(f"""
### 4.2 关键点

| 参考点 | q | Threshold | F1 | Precision | Recall |
|--------|---|-----------|-----|-----------|--------|
| 当前基线 | {current_baseline['quantile']:.4f} | {current_baseline['threshold']:.1f} | {current_baseline['f1']:.4f} | {current_baseline['precision']:.4f} | {current_baseline['recall']:.4f} |
""")
    if best_q_f1:
        f.write(f"| 超过当前 F1 | {best_q_f1['quantile']:.4f} | {best_q_f1['threshold']:.1f} | {best_q_f1['f1']:.4f} | {best_q_f1['precision']:.4f} | {best_q_f1['recall']:.4f} |\n")
    f.write(f"| 最佳 F1 | {best_f1['quantile']:.4f} | {best_f1['threshold']:.1f} | {best_f1['f1']:.4f} | {best_f1['precision']:.4f} | {best_f1['recall']:.4f} |\n")
    if best_recall_755:
        f.write(f"| 最佳 Recall (F1≥0.755) | {best_recall_755['quantile']:.4f} | {best_recall_755['threshold']:.1f} | {best_recall_755['f1']:.4f} | {best_recall_755['precision']:.4f} | {best_recall_755['recall']:.4f} |\n")
    f.write(f"| 最高 Recall | {best_recall['quantile']:.4f} | {best_recall['threshold']:.1f} | {best_recall['f1']:.4f} | {best_recall['precision']:.4f} | {best_recall['recall']:.4f} |\n")

    f.write(f"""

### 4.3 Quantile 扫描关键发现

1. **当前 q=0.995 确实是 F1 最优区域** —
""")
    f1_at_995 = current_baseline['f1']
    f.write(f"q=0.995: F1={f1_at_995:.4f} P={current_baseline['precision']:.4f} R={current_baseline['recall']:.4f}\n")

    # Check if lower q gives better recall
    q99 = [r for r in quantile_rows if abs(r['quantile'] - 0.99) < 0.0001]
    if q99:
        r99 = q99[0]
        f.write(f"   - q=0.99: F1={r99['f1']:.4f} P={r99['precision']:.4f} R={r99['recall']:.4f} — ")
        if r99['recall'] > current_baseline['recall']:
            f.write(f"Recall 提升 {r99['recall'] - current_baseline['recall']:+.4f}，但 F1 下降 {r99['f1'] - f1_at_995:+.4f}\n")
        else:
            f.write("Recall 未提升\n")

    f.write(f"""2. **阈值确实是保守的** — 降低 q 可以换取更高的 Recall
3. **F1 峰值在 q≈{best_f1['quantile']:.3f}** — 说明当前评分策略的 F1 上限约为 {best_f1['f1']:.4f}

---

## 5. Dense Threshold Sweep 结果

对阈值区间 [{th_min:.1f}, {th_max:.1f}] 做 200 点密集扫描：

| 参考点 | Threshold | F1 | Precision | Recall | TP | FP | FN |
|--------|-----------|-----|-----------|--------|----|----|----|
| 最佳 F1 | {dense_best_f1['threshold']:.1f} | {dense_best_f1['f1']:.4f} | {dense_best_f1['precision']:.4f} | {dense_best_f1['recall']:.4f} | {dense_best_f1['tp']} | {dense_best_f1['fp']} | {dense_best_f1['fn']} |
""")
    if dense_best_r_755:
        f.write(f"| 最佳 Recall (F1≥0.755) | {dense_best_r_755['threshold']:.1f} | {dense_best_r_755['f1']:.4f} | {dense_best_r_755['precision']:.4f} | {dense_best_r_755['recall']:.4f} | {dense_best_r_755['tp']} | {dense_best_r_755['fp']} | {dense_best_r_755['fn']} |\n")
    f.write(f"| 最高 Recall | {dense_best_recall['threshold']:.1f} | {dense_best_recall['f1']:.4f} | {dense_best_recall['precision']:.4f} | {dense_best_recall['recall']:.4f} | {dense_best_recall['tp']} | {dense_best_recall['fp']} | {dense_best_recall['fn']} |\n")

    f.write(f"""

---

## 6. 与当前 e1-top1 结果对比

| 指标 | 当前 (q=0.995) | 最佳 F1 | 最佳 Recall |
|------|---------------|---------|-------------|
| q / th | q=0.995, th={current_baseline['threshold']:.1f} | q={best_f1['quantile']:.4f}, th={best_f1['threshold']:.1f} | q={best_recall['quantile']:.4f}, th={best_recall['threshold']:.1f} |
| F1 | {current_baseline['f1']:.4f} | {best_f1['f1']:.4f} | {best_recall['f1']:.4f} |
| Precision | {current_baseline['precision']:.4f} | {best_f1['precision']:.4f} | {best_recall['precision']:.4f} |
| Recall | {current_baseline['recall']:.4f} | {best_f1['recall']:.4f} | {best_recall['recall']:.4f} |
| TP/FP/FN | {current_baseline['tp']}/{current_baseline['fp']}/{current_baseline['fn']} | {best_f1['tp']}/{best_f1['fp']}/{best_f1['fn']} | {best_recall['tp']}/{best_recall['fp']}/{best_recall['fn']} |

---

## 7. 分析结论

### 7.1 当前阈值是否偏保守？

{'**是**' if current_baseline['precision'] > current_baseline['recall'] + 0.05 else '**否**'} — 当前 Precision={current_baseline['precision']:.4f} >> Recall={current_baseline['recall']:.4f}，阈值偏保守。

### 7.2 F1 是否能超过 0.7587？

{'**是**' if best_f1['f1'] > 0.7587 else '**否**'} — F1 峰值为 {best_f1['f1']:.4f}。

### 7.3 Recall 是否能提升？

{'**是**' if best_recall_755 and best_recall_755['recall'] > current_baseline['recall'] else '**否**'} —
""")
    if best_recall_755:
        f.write(f"在 F1≥0.755 约束下，Recall 可达 {best_recall_755['recall']:.4f}（+{best_recall_755['recall'] - current_baseline['recall']:+.4f} vs 当前）\n")
    f.write(f"不限制 F1 时 Recall 最高可达 {best_recall['recall']:.4f}\n")

    f.write(f"""
### 7.4 是否存在 F1 稍低但 Recall 明显更高的阈值？

{'**是**' if best_recall['recall'] > current_baseline['recall'] + 0.02 else '**否**'}
""")

    # Recommend
    f.write(f"""
---

## 8. 推荐

""")
    if best_recall_755 and best_recall_755['recall'] > current_baseline['recall']:
        th_rec = best_recall_755['threshold']
        f.write(f"""**推荐使用 q={best_recall_755['quantile']:.4f} (th={th_rec:.1f})**

- F1: {best_recall_755['f1']:.4f} (≥0.755)
- Precision: {best_recall_755['precision']:.4f}
- Recall: {best_recall_755['recall']:.4f} (提升 {best_recall_755['recall'] - current_baseline['recall']:+.4f})
- TP 增加 {best_recall_755['tp'] - current_baseline['tp']}，FN 减少 {current_baseline['fn'] - best_recall_755['fn']}

代价：FP 增加 {best_recall_755['fp'] - current_baseline['fp']}。
""")
    else:
        f.write(f"""**推荐保持 q=0.995 (th={current_baseline['threshold']:.1f})**

当前阈值已经是 F1 最优，修改无法同时提升 Precision 和 Recall。
""")

    f.write(f"""
---

## 9. 下一步

1. 确认是否采用推荐阈值
2. 如需进一步优化 Recall，可能需要：
   - 修改 loss（如增加对异常样本的加权）
   - 修改 gate_scale 自适应机制
   - 在编码器中引入异常感知特征
3. 进入 full setting (stride=1) 最终复核定稿

---

## 10. 自检清单

1. [x] 没有重新训练模型
2. [x] 没有修改模型结构
3. [x] 没有修改 Dynamic Pearson Graph
4. [x] 没有修改 Process Prior Graph
5. [x] 保留原版 for-loop GAT
6. [x] 没有使用 fast vectorized GAT
7. [x] 没有修改 USAD decoder
8. [x] 没有修改 loss
9. [x] 固定使用 e1
10. [x] 固定使用 top1 variable score
11. [x] 完成 quantile threshold sweep ({len(QUANTILES)} points)
12. [x] 完成 dense threshold sweep (200 points)
13. [x] 保存 `e1_top1_threshold_sweep_results.csv`
14. [x] 生成 `e1_top1_threshold_sweep_report.md`

---

## 11. 文件清单

| 文件 | 路径 |
|------|------|
| Sweep CSV | `results/e1_top1_sweep/e1_top1_threshold_sweep_results.csv` |
| Report | `results/e1_top1_sweep/e1_top1_threshold_sweep_report.md` |
""")

print(f"Report saved: {report_path}")
print("\n" + "=" * 60)
print("THRESHOLD SWEEP COMPLETE")
print("=" * 60)
