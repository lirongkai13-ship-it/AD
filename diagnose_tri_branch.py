"""tri_branch 诊断分析: 为什么Recall/AUC下降?"""
import sys, os, json, numpy as np, torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, auc
from sklearn.metrics import precision_recall_fscore_support
from collections import defaultdict

sys.path.insert(0, '.')
from data_loader import prepare_data
from utils import load_config, set_seed, get_device, fit_iqr_params, apply_iqr_normalize, aggregate_topk_score
from models_variants.tri_branch.variant_model import TriBranch_USAD
from models_variants.parallel_usad_prior.variant_model import ParallelPrior_USAD

OUT_DIR = 'results/diagnosis'
os.makedirs(OUT_DIR, exist_ok=True)
device = 'cuda'

cfg = load_config('config_dev.yaml')
set_seed(42)
_, val_ds, test_ds, _, info = prepare_data(cfg)
val_loader = DataLoader(val_ds, 256, shuffle=False)
test_loader = DataLoader(test_ds, 256, shuffle=False)

# Build graphs
from data_loader import build_pearson_edge_index, split_train_val, read_swat_csv
from sklearn.preprocessing import StandardScaler
import pandas as pd, importlib.util
dcfg = cfg['data']
nfd = pd.read_csv(dcfg['train_csv']); nfd.columns=[str(c).strip() for c in nfd.columns]
nfd=nfd[[c for c in nfd.columns if c not in ['Timestamp','Normal/Attack']]]
mfd = pd.read_csv(dcfg['test_csv']); mfd.columns=[str(c).strip() for c in mfd.columns]
common_cols = [c for c in nfd.columns if c in mfd.columns]
raw = nfd[common_cols].values.astype(np.float32)
tr,_,_,_=split_train_val(raw,None,0.2)
tv = StandardScaler().fit_transform(tr)
static_ei, _ = build_pearson_edge_index(tv)

bgp = importlib.util.spec_from_file_location('bpg','models_variants/prior_fusion/build_prior_graph.py')
bpgm = importlib.util.module_from_spec(bgp); bgp.loader.exec_module(bpgm)
prior_ei, prior_w = bpgm.build_prior_graph(common_cols)

static_ei = static_ei.to(device); prior_ei = prior_ei.to(device); prior_w = prior_w.to(device)

# Load models
def load_model(cls, ckpt_path, **kw):
    m = cls(info['num_variables'], 60, static_ei, prior_ei, prior_w, 32, 2, 32, 32, 1, 0.2, **kw).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    m.load_state_dict(ckpt['model']); m.eval()
    return m

prior_model = load_model(ParallelPrior_USAD, 'outputs/swat_normal_train_merged_test/parallel_usad_prior/best_model.pt')
tri_model = load_model(TriBranch_USAD, 'outputs/swat_normal_train_merged_test/tri_branch/best_model.pt',
                       encoder_mode='tri_branch_residual_gate')

# Collect scores
def collect_scores(model, loader):
    errors, labels = [], []
    for batch in loader:
        x = batch['x'].to(device)
        with torch.no_grad():
            r = model.forward_eval(x, static_ei)
        errors.append((r - x).abs().cpu().numpy())
        if 'label' in batch: labels.append(batch['label'].cpu().numpy())
    return np.concatenate(errors, 0), np.concatenate(labels, 0)

print("Collecting val/test scores...")
v_prior, _ = collect_scores(prior_model, val_loader)
v_tri, _ = collect_scores(tri_model, val_loader)
t_prior, t_labels = collect_scores(prior_model, test_loader)
t_tri, _ = collect_scores(tri_model, test_loader)

# Standard scoring pipeline
def compute_score(errors, iqr_params=None, topk=5):
    if iqr_params is None:
        iqr_params = fit_iqr_params(errors)
    norm = apply_iqr_normalize(errors, iqr_params)
    return aggregate_topk_score(norm, topk=topk), iqr_params

scores_prior_val, iqr_prior = compute_score(v_prior)
scores_tri_val, iqr_tri = compute_score(v_tri)
scores_prior_test, _ = compute_score(t_prior, iqr_prior)
scores_tri_test, _ = compute_score(t_tri, iqr_tri)

th_prior = float(np.quantile(scores_prior_val, 0.995))
th_tri = float(np.quantile(scores_tri_val, 0.995))

pred_prior = (scores_prior_test > th_prior).astype(int)
pred_tri = (scores_tri_test > th_tri).astype(int)

# ════════════ 1. Score Distribution ════════════
print("\n=== Score Distribution ===")
stats = []
for name, scores, lbls in [('prior',scores_prior_test,t_labels),('tri',scores_tri_test,t_labels)]:
    for label_name, mask in [('normal',lbls==0),('attack',lbls==1)]:
        s = scores[mask]
        row = {'model':name,'label':label_name,'mean':np.mean(s),'std':np.std(s),
               'min':np.min(s),'max':np.max(s)}
        for q in [50,75,90,95,97,99]:
            row[f'p{q}'] = np.percentile(s,q)
        stats.append(row)
        print(f"  {name} {label_name}: mean={np.mean(s):.2f} std={np.std(s):.2f} p95={np.percentile(s,95):.2f}")

import csv
with open(f'{OUT_DIR}/score_distribution_stats.csv','w',newline='') as f:
    w = csv.DictWriter(f,fieldnames=stats[0].keys()); w.writeheader(); w.writerows(stats)

# ════════════ 2. Score Histogram (simplified text) ════════════
print("\n=== Score Bins ===")
for name, scores in [('prior',scores_prior_test),('tri',scores_tri_test)]:
    for lbl, mask in [('normal',t_labels==0),('attack',t_labels==1)]:
        s=scores[mask]
        print(f"  {name} {lbl}: min={s.min():.1f} max={s.max():.1f} median={np.median(s):.1f}")

# ════════════ 3. ROC/PR curves ════════════
print("\n=== ROC/PR ===")
roc_prior = roc_auc_score(t_labels, scores_prior_test)
roc_tri = roc_auc_score(t_labels, scores_tri_test)
pr_prior = average_precision_score(t_labels, scores_prior_test)
pr_tri = average_precision_score(t_labels, scores_tri_test)
print(f"  prior: AUROC={roc_prior:.4f} AUPR={pr_prior:.4f}")
print(f"  tri:   AUROC={roc_tri:.4f} AUPR={pr_tri:.4f}")

with open(f'{OUT_DIR}/curve_metrics.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['model','AUROC','AUPR','best_F1','best_threshold','precision_at_best','recall_at_best'])
    w.writeheader()
    for name,scores in [('prior',scores_prior_test),('tri',scores_tri_test)]:
        fpr,tpr,_=roc_curve(t_labels,scores)
        prec,rec,thr=precision_recall_curve(t_labels,scores)
        fs=2*prec*rec/(prec+rec+1e-8)
        bi=np.argmax(fs)
        w.writerow({'model':name,'AUROC':roc_auc_score(t_labels,scores),
                    'AUPR':average_precision_score(t_labels,scores),
                    'best_F1':fs[bi],'best_threshold':thr[bi] if bi<len(thr) else 0,
                    'precision_at_best':prec[bi],'recall_at_best':rec[bi]})

# ════════════ 4. Threshold Sweep ════════════
print("\n=== Threshold Sweep (tri_branch) ===")
with open(f'{OUT_DIR}/threshold_sweep_tri_branch.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['threshold','F1','Precision','Recall','FP','FN','TP','TN'])
    w.writeheader()
    for q in [90,92,94,95,96,97,98,99,99.5,99.9]:
        th = float(np.quantile(scores_tri_test, q/100))
        p = (scores_tri_test > th).astype(int)
        tp=((p==1)&(t_labels==1)).sum(); tn=((p==0)&(t_labels==0)).sum()
        fp=((p==1)&(t_labels==0)).sum(); fn=((p==0)&(t_labels==1)).sum()
        prec_v = tp/(tp+fp+1e-8); rec_v = tp/(tp+fn+1e-8)
        f1_v = 2*prec_v*rec_v/(prec_v+rec_v+1e-8)
        print(f"  q={q:5.1f}% th={th:.1f} F1={f1_v:.4f} P={prec_v:.4f} R={rec_v:.4f}")
        w.writerow({'threshold':th,'F1':f1_v,'Precision':prec_v,'Recall':rec_v,'FP':fp,'FN':fn,'TP':tp,'TN':tn})

# ════════════ 5. Top-K Variable Score ════════════
print("\n=== Top-K Variable Score ===")
with open(f'{OUT_DIR}/topk_score_results.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['k','F1','Precision','Recall','AUC','AUPR','best_threshold'])
    w.writeheader()
    for k in [1,3,5,8,10,15,20,51]:
        # Compute top-k scores per model on val+test
        v_norm_prior = apply_iqr_normalize(v_prior, iqr_prior)
        v_norm_tri = apply_iqr_normalize(v_tri, iqr_tri)

        t_norm_prior = apply_iqr_normalize(t_prior, iqr_prior)
        t_norm_tri = apply_iqr_normalize(t_tri, iqr_tri)

        s_v_prior = aggregate_topk_score(v_norm_prior, k)
        s_t_prior = aggregate_topk_score(t_norm_prior, k)
        th_k_prior = float(np.quantile(s_v_prior, 0.995))
        p_k_prior = (s_t_prior > th_k_prior).astype(int)
        pr,rc,f1,_ = precision_recall_fscore_support(t_labels,p_k_prior,average='binary',zero_division=0)
        w.writerow({'k':k,'F1':f1,'Precision':pr,'Recall':rc,
                    'AUC':roc_auc_score(t_labels,s_t_prior),
                    'AUPR':average_precision_score(t_labels,s_t_prior),
                    'best_threshold':th_k_prior})

        s_v_tri = aggregate_topk_score(v_norm_tri, k)
        s_t_tri = aggregate_topk_score(t_norm_tri, k)
        th_k_tri = float(np.quantile(s_v_tri, 0.995))
        p_k_tri = (s_t_tri > th_k_tri).astype(int)
        pr2,rc2,f12,_ = precision_recall_fscore_support(t_labels,p_k_tri,average='binary',zero_division=0)
        print(f"  k={k:2d}: prior F1={f1:.4f} P={pr:.4f} R={rc:.4f} | tri F1={f12:.4f} P={pr2:.4f} R={rc2:.4f}")

# ════════════ 6. Attack Segment Recall ════════════
print("\n=== Attack Segment Recall ===")
segments = []
start = None
for i in range(len(t_labels)):
    if t_labels[i]==1 and start is None: start=i
    elif t_labels[i]==0 and start is not None:
        segments.append((start,i-1,i-start)); start=None
if start is not None: segments.append((start,len(t_labels)-1,len(t_labels)-start))

short=0; medium=0; long=0
s_r={'prior':{'short':[],'medium':[],'long':[]},'tri':{'short':[],'medium':[],'long':[]}}
for sid,(st,en,length) in enumerate(segments):
    cat = 'short' if length<100 else ('medium' if length<1000 else 'long')
    for name,pred in [('prior',pred_prior),('tri',pred_tri)]:
        seg_pred=pred[st:en+1]
        seg_true=t_labels[st:en+1]
        detected = int(seg_pred.sum()>0)
        seg_rec = seg_pred.sum()/len(seg_pred) if len(seg_pred)>0 else 0
        if cat=='short': short+=1
        elif cat=='medium': medium+=1
        else: long+=1
        s_r[name][cat].append(detected)

print(f"  Total segments: {len(segments)} (short<100: {short}, medium<1000: {medium}, long>=1000: {long})")
for name in ['prior','tri']:
    for cat in ['short','medium','long']:
        vals=s_r[name][cat]
        if vals:
            print(f"  {name} {cat}: detection_rate={np.mean(vals):.3f} ({sum(vals)}/{len(vals)})")

# ════════════ 7. Gate Analysis ════════════
print("\n=== Gate Analysis ===")
# Need to run encoder with gate extraction
gate_normal=[]; gate_attack=[]
with torch.no_grad():
    for batch in DataLoader(test_ds, 256, shuffle=False):
        x = batch['x'].to(device)
        lbl = batch['label'].cpu().numpy()
        # Run encoder manually to extract gate values
        with torch.no_grad():
            # We can't easily extract gate mid-forward, so just note this
            pass
        break
print("  (Gate extraction requires model internal hook - skipping for now)")

# ════════════ Summary ════════════
print("\n" + "="*60)
print("DIAGNOSIS SUMMARY")
print("="*60)
print(f"  threshold: prior={th_prior:.1f} vs tri={th_tri:.1f} (tri is {th_tri/th_prior*100-100:+.0f}% higher)")
print(f"  F1:        prior={0.7524} vs tri=0.7546")
print(f"  Precision: prior=0.7661 vs tri=0.7957  (tri higher)")
print(f"  Recall:    prior=0.7392 vs tri=0.7176  (tri LOWER)")
print(f"  AUC:       prior=0.9503 vs tri=0.9374  (tri LOWER)")

# Score distribution analysis
normal_prior = scores_prior_test[t_labels==0]
attack_prior = scores_prior_test[t_labels==1]
normal_tri = scores_tri_test[t_labels==0]
attack_tri = scores_tri_test[t_labels==1]

print(f"\n  Normal score mean: prior={normal_prior.mean():.1f} vs tri={normal_tri.mean():.1f}")
print(f"  Attack score mean: prior={attack_prior.mean():.1f} vs tri={attack_tri.mean():.1f}")
print(f"  Separation ratio (attack_mean/normal_mean): prior={attack_prior.mean()/normal_prior.mean():.2f} vs tri={attack_tri.mean()/normal_tri.mean():.2f}")
print(f"\n  Threshold vs normal mean: prior={th_prior/normal_prior.mean():.2f} vs tri={th_tri/normal_tri.mean():.2f}")

print(f"\nFull report -> {OUT_DIR}/")
