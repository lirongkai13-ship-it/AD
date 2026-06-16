"""tri_branch 模型专用评估脚本
支持两种评分策略:
  mode=standard: 原始配置 (e1 only, k=5, q=0.995)
  mode=optimized: 优化配置 (e1 only, k=1, q=0.995)
同时也测试 e1/e2/e12 权重组合 + Top-K sweep，保存完整结果。
"""
import argparse, os, sys, time, csv, json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_fscore_support)

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import prepare_data, build_pearson_edge_index, split_train_val, read_swat_csv
from utils import (load_config, set_seed, get_device, ensure_dir,
                   fit_iqr_params, apply_iqr_normalize, aggregate_topk_score,
                   point_adjust, save_json)
from models_variants.tri_branch.variant_model import TriBranch_USAD


def build_prior_graph(common_cols):
    import importlib.util
    bgp_path = os.path.join(os.path.dirname(__file__), 'models_variants', 'prior_fusion', 'build_prior_graph.py')
    spec = importlib.util.spec_from_file_location('bpg', bgp_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_prior_graph(common_cols)


def load_tri_branch_model(ckpt_path, info, static_ei, prior_ei, prior_w, device,
                          encoder_mode="tri_branch_residual_gate",
                          temporal_mode="per_variable_conv",
                          gamma_mode="fixed", gamma_value=0.05, gate_scale=1.0):
    model = TriBranch_USAD(
        nv=info['num_variables'], ws=60,
        static_edge_index=static_ei,
        prior_edge_index=prior_ei,
        prior_weights=prior_w,
        hidden_dim=32, gat_heads=2,
        gru_hidden=32, tcn_channels=32, tcn_blocks=1,
        dropout=0.2, latent_dim=64, use_flatten=True,
        temporal_mode=temporal_mode,
        encoder_mode=encoder_mode,
        gamma_mode=gamma_mode, gamma_value=gamma_value, gate_scale=gate_scale,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    # Remap old key names (gated_fusion.gate.* → gated_fusion.gate_mlp.*)
    state_dict = {}
    for k, v in ckpt['model'].items():
        if 'gated_fusion.gate.' in k:
            k = k.replace('gated_fusion.gate.', 'gated_fusion.gate_mlp.')
        state_dict[k] = v
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def collect_all_errors(model, loader, static_ei, device):
    """收集 e1, e2, e12 逐变量误差和标签"""
    model.eval()
    e1_list, e2_list, e12_list = [], [], []
    labels_list = []
    for batch in loader:
        x = batch['x'].to(device)
        r1, r2, r12 = model(x, static_ei)
        e1_list.append((r1 - x).abs().mean(dim=1).cpu().numpy())   # [B,N]
        e2_list.append((r2 - x).abs().mean(dim=1).cpu().numpy())
        e12_list.append((r12 - x).abs().mean(dim=1).cpu().numpy())
        if 'label' in batch:
            labels_list.append(batch['label'].cpu().numpy())
    e1 = np.concatenate(e1_list, axis=0)
    e2 = np.concatenate(e2_list, axis=0)
    e12 = np.concatenate(e12_list, axis=0)
    labels = np.concatenate(labels_list, axis=0) if labels_list else None
    return e1, e2, e12, labels


def compute_metrics(e_val, e_test, val_labels, test_labels, topk, quantile=0.995):
    """完整评分流水线: IQR(on val normal) → normalize → topk → threshold → metrics"""
    val_normal_mask = (val_labels == 0)
    e_val_normal = e_val[val_normal_mask]
    iqr_params = fit_iqr_params(e_val_normal)
    v_norm = apply_iqr_normalize(e_val, iqr_params)
    t_norm = apply_iqr_normalize(e_test, iqr_params)
    v_score = aggregate_topk_score(v_norm, topk=topk)
    t_score = aggregate_topk_score(t_norm, topk=topk)
    v_score_normal = v_score[val_normal_mask]
    threshold = float(np.quantile(v_score_normal, quantile))
    t_pred = (t_score > threshold).astype(int)

    pr, rc, f1, _ = precision_recall_fscore_support(test_labels, t_pred, average='binary', zero_division=0)
    auc_val = float(roc_auc_score(test_labels, t_score))
    aupr_val = float(average_precision_score(test_labels, t_score))
    tp = int(((t_pred == 1) & (test_labels == 1)).sum())
    fp = int(((t_pred == 1) & (test_labels == 0)).sum())
    tn = int(((t_pred == 0) & (test_labels == 0)).sum())
    fn = int(((t_pred == 0) & (test_labels == 1)).sum())

    normal_scores = t_score[test_labels == 0]
    attack_scores = t_score[test_labels == 1]
    sep_ratio = float(attack_scores.mean() / normal_scores.mean()) if normal_scores.mean() > 0 else float('inf')

    return {
        'threshold': threshold, 'topk': topk, 'quantile': quantile,
        'precision': float(pr), 'recall': float(rc), 'f1': float(f1),
        'auc': auc_val, 'aupr': aupr_val,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'normal_score_mean': float(normal_scores.mean()),
        'attack_score_mean': float(attack_scores.mean()),
        'separation_ratio': sep_ratio,
        'normal_score_std': float(normal_scores.std()),
        'attack_score_std': float(attack_scores.std()),
    }


def main():
    parser = argparse.ArgumentParser(description="tri_branch model evaluation")
    parser.add_argument("--config", type=str, default="config_dev.yaml")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--mode", type=str, default="all",
                        choices=["standard", "optimized", "all"],
                        help="standard: e1+k=5+q=0.995 | optimized: e1+k=1+q=0.995 | all: both + sweep")
    parser.add_argument("--output", type=str, default=None,
                        help="output directory (default: config save_dir/tri_branch_eval)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg["train"].get("device", "cuda"))
    set_seed(int(cfg["train"]["seed"]))

    # ── Output dir ──
    if args.output:
        save_dir = args.output
    else:
        save_dir = os.path.join(cfg["output"]["save_dir"], "tri_branch_eval")
    ensure_dir(save_dir)

    # ── Checkpoint ──
    if args.ckpt is None:
        args.ckpt = os.path.join(cfg["output"]["save_dir"], "tri_branch", "best_model.pt")

    # ── Data ──
    _, val_ds, test_ds, _, info = prepare_data(cfg)
    bs = int(cfg["train"]["batch_size"])
    nw = cfg["train"].get("num_workers", 0)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, drop_last=False, num_workers=nw, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, drop_last=False, num_workers=nw, pin_memory=True)

    # ── Build graphs ──
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    dcfg = cfg["data"]
    nfd = pd.read_csv(dcfg["train_csv"]); nfd.columns = [str(c).strip() for c in nfd.columns]
    nfd = nfd[[c for c in nfd.columns if c not in ['Timestamp', 'Normal/Attack']]]
    mfd = pd.read_csv(dcfg["test_csv"]); mfd.columns = [str(c).strip() for c in mfd.columns]
    common_cols = [c for c in nfd.columns if c in mfd.columns]
    raw = nfd[common_cols].values.astype(np.float32)
    tr, _, _, _ = split_train_val(raw, None, 0.2)
    tv = StandardScaler().fit_transform(tr)
    static_ei, _ = build_pearson_edge_index(tv)
    prior_ei, prior_w = build_prior_graph(common_cols)
    static_ei = static_ei.to(device)
    prior_ei = prior_ei.to(device)
    prior_w = prior_w.to(device)

    # ── Load model ──
    print(f"Loading tri_branch model from: {args.ckpt}")
    model = load_tri_branch_model(args.ckpt, info, static_ei, prior_ei, prior_w, device,
                                  encoder_mode="tri_branch_residual_gate",
                                  temporal_mode="per_variable_conv",
                                  gamma_mode="fixed", gamma_value=0.05, gate_scale=1.0)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    # ── Collect errors ──
    print("Collecting validation errors...")
    t0 = time.time()
    v_e1, v_e2, v_e12, v_labels = collect_all_errors(model, val_loader, static_ei, device)
    print(f"  Val: {v_e1.shape[0]} samples × {v_e1.shape[1]} vars")

    print("Collecting test errors...")
    t_e1, t_e2, t_e12, t_labels = collect_all_errors(model, test_loader, static_ei, device)
    print(f"  Test: {t_e1.shape[0]} samples × {t_e1.shape[1]} vars")
    print(f"  Collection time: {time.time() - t0:.1f}s")

    # ── Evaluate ──
    results = {}

    # Standard: e1, k=5, q=0.995 (original tri_branch config)
    print("\n=== Standard Scoring (e1, k=5, q=0.995) ===")
    std_metrics = compute_metrics(v_e1, t_e1, v_labels, t_labels, topk=5, quantile=0.995)
    results["standard"] = std_metrics
    print(f"  F1={std_metrics['f1']:.4f}  P={std_metrics['precision']:.4f}  R={std_metrics['recall']:.4f}")
    print(f"  AUC={std_metrics['auc']:.4f}  AUPR={std_metrics['aupr']:.4f}")
    print(f"  Threshold={std_metrics['threshold']:.2f}")
    print(f"  Normal mean={std_metrics['normal_score_mean']:.1f}  Attack mean={std_metrics['attack_score_mean']:.1f}")
    print(f"  Separation ratio={std_metrics['separation_ratio']:.2f}")

    # Optimized: e1, k=1, q=0.995 (best from score optimization)
    print("\n=== Optimized Scoring (e1, k=1, q=0.995) ===")
    opt_metrics = compute_metrics(v_e1, t_e1, v_labels, t_labels, topk=1, quantile=0.995)
    results["optimized"] = opt_metrics
    print(f"  F1={opt_metrics['f1']:.4f}  P={opt_metrics['precision']:.4f}  R={opt_metrics['recall']:.4f}")
    print(f"  AUC={opt_metrics['auc']:.4f}  AUPR={opt_metrics['aupr']:.4f}")
    print(f"  Threshold={opt_metrics['threshold']:.2f}")
    print(f"  Normal mean={opt_metrics['normal_score_mean']:.1f}  Attack mean={opt_metrics['attack_score_mean']:.1f}")
    print(f"  Separation ratio={opt_metrics['separation_ratio']:.2f}")

    # Point-adjust for both
    for mode_name, m in [("standard", std_metrics), ("optimized", opt_metrics)]:
        # Recompute pred for PA
        val_normal_mask = (v_labels == 0)
        e_val_normal = v_e1[val_normal_mask]
        iqr_params = fit_iqr_params(e_val_normal)
        t_norm = apply_iqr_normalize(t_e1, iqr_params)
        t_score = aggregate_topk_score(t_norm, topk=m['topk'])
        t_pred = (t_score > m['threshold']).astype(int)
        pa_pred = point_adjust(t_pred, t_labels)
        pa_pr, pa_rc, pa_f1, _ = precision_recall_fscore_support(t_labels, pa_pred, average='binary', zero_division=0)
        m['point_adjust'] = {
            'precision': float(pa_pr), 'recall': float(pa_rc), 'f1': float(pa_f1),
            'auc': m['auc'], 'aupr': m['aupr'],
        }
        results[mode_name] = m

    print(f"\n=== Point-Adjust ===")
    print(f"  Standard  PA: F1={results['standard']['point_adjust']['f1']:.4f}")
    print(f"  Optimized PA: F1={results['optimized']['point_adjust']['f1']:.4f}")

    # ── Full sweep (if mode=all) ──
    if args.mode == "all":
        sweep_results = []

        # Top-K per error type
        K_VALUES = [1, 3, 5, 8, 10, 15, 20, 51]
        for err_name, v_err, t_err in [("e1", v_e1, t_e1), ("e2", v_e2, t_e2), ("e12", v_e12, t_e12)]:
            for k in K_VALUES:
                r = compute_metrics(v_err, t_err, v_labels, t_labels, topk=k, quantile=0.995)
                r['error_type'] = err_name
                sweep_results.append(r)

        # Weight combos
        combos = [
            (1.0, 0.0, 0.0, "e1 only"),
            (0.0, 1.0, 0.0, "e2 only"),
            (0.0, 0.0, 1.0, "e12 only"),
            (0.5, 0.5, 0.0, "e1+e2 (0.5:0.5)"),
            (0.5, 0.0, 0.5, "e1+e12 (0.5:0.5)"),
            (0.0, 0.5, 0.5, "e2+e12 (0.5:0.5)"),
            (0.333, 0.333, 0.333, "e1+e2+e12 (equal)"),
            (0.8, 0.1, 0.1, "e1 dominant (0.8:0.1:0.1)"),
        ]
        for w1, w2, w12, label in combos:
            v_w = w1 * v_e1 + w2 * v_e2 + w12 * v_e12
            t_w = w1 * t_e1 + w2 * t_e2 + w12 * t_e12
            for k in K_VALUES:
                r = compute_metrics(v_w, t_w, v_labels, t_labels, topk=k, quantile=0.995)
                r['weight_label'] = label
                r['w1'] = w1; r['w2'] = w2; r['w12'] = w12
                sweep_results.append(r)

        # Save sweep
        sweep_path = os.path.join(save_dir, "full_sweep_results.csv")
        with open(sweep_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=sweep_results[0].keys())
            w.writeheader()
            w.writerows(sweep_results)
        print(f"\nFull sweep saved: {sweep_path} ({len(sweep_results)} rows)")

        # Quick summary
        best_f1 = max(sweep_results, key=lambda r: r['f1'])
        print(f"Best F1 in sweep: {best_f1['f1']:.4f} "
              f"({best_f1.get('error_type', best_f1.get('weight_label', ''))} "
              f"k={best_f1['topk']})")

    # ── Save metrics ──
    metrics_path = os.path.join(save_dir, "metrics.json")
    save_json(results, metrics_path)
    print(f"\nMetrics saved: {metrics_path}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Model: tri_branch_residual_gate (gamma=0.05, gate_scale=1.0)")
    print(f"Params: {n_params:,}")
    print(f"Checkpoint: {args.ckpt}")
    print()
    print(f"{'':20s} {'F1':>8s} {'P':>8s} {'R':>8s} {'AUC':>8s} {'AUPR':>8s}")
    print(f"{'Standard (k=5)':20s} {std_metrics['f1']:8.4f} {std_metrics['precision']:8.4f} "
          f"{std_metrics['recall']:8.4f} {std_metrics['auc']:8.4f} {std_metrics['aupr']:8.4f}")
    print(f"{'Optimized (k=1)':20s} {opt_metrics['f1']:8.4f} {opt_metrics['precision']:8.4f} "
          f"{opt_metrics['recall']:8.4f} {opt_metrics['auc']:8.4f} {opt_metrics['aupr']:8.4f}")
    print(f"\nResults directory: {save_dir}")


if __name__ == "__main__":
    main()
