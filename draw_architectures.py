"""生成所有模型架构图 (纯英文，无表情符号)"""
import sys, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "model_diagrams")
os.makedirs(OUT_DIR, exist_ok=True)

C = {
    'input': '#E8F5E9', 'gat': '#BBDEFB', 'tcn': '#FFE0B2',
    'gru': '#F8BBD0', 'pred': '#CE93D8', 'recon': '#80CBC4',
    'decoder1': '#80CBC4', 'decoder2': '#4DB6AC', 'prior': '#FFCC80',
    'time_attn': '#A5D6A7', 'dyn_graph': '#90CAF9', 'special': '#EF9A9A',
    'fusion': '#FFF176',
}
BOX_W, BOX_H = 2.0, 0.55
GAP_Y = 0.85

def box(ax, x, y, w, h, text, color, fs=8, bold=False):
    b = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle='round,pad=0.08',
                        facecolor=color, edgecolor='#37474F', linewidth=1.2)
    ax.add_patch(b)
    ax.text(x, y, text, ha='center', va='center', fontsize=fs,
            fontweight='bold' if bold else 'normal', family='sans-serif')

def draw_model(name, layers, save_path):
    n = len(layers)
    fig, ax = plt.subplots(1, 1, figsize=(10, max(6, n * 1.1 + 2)))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, n * 1.1 + 2)
    ax.axis('off')
    ax.text(6, n * 1.1 + 1.5, name, ha='center', fontsize=14, fontweight='bold')
    y = n * 1.1 + 0.3
    for (label, color, w_override) in layers:
        w = w_override if w_override else BOX_W
        box(ax, 6, y, w, BOX_H, label, color, fs=7)
        y -= 1.05
    out = os.path.join(OUT_DIR, save_path)
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f'  [OK] {save_path}')

# ====== Our Models ======
draw_model('GATv2+TCN+GRU (Baseline)', [
    ('Input [B,60,51]', C['input'], None),
    ('GATv2 Block (2 layers, heads=2->1, residual+LN)', C['gat'], 3.5),
    ('TCN (2 Blocks, dilation=1->2, kernel=3)', C['tcn'], 3.0),
    ('GRU (1-layer, unidirectional)', C['gru'], None),
    ('Pred Head: Lin->ReLU->Drop->Lin [B,51]\n||  Recon Head: Lin->ReLU->Drop->Lin [B,60,51]', C['pred'], 5.5),
], '01_baseline.png')

draw_model('GATv2 + Temporal Attention + TCN + GRU', [
    ('Input [B,60,51]', C['input'], None),
    ('Temporal Attention [NEW] (Multi-Head Self-Attn on time dim)', C['time_attn'], 3.8),
    ('GATv2 Block', C['gat'], None),
    ('TCN Block', C['tcn'], None),
    ('GRU', C['gru'], None),
    ('Pred Head  ||  Recon Head', C['pred'], 3.0),
], '02_temporal_attn.png')

draw_model('GATv2 + Prior Knowledge Fusion + TCN + GRU', [
    ('Input [B,60,51]', C['input'], None),
    ('GATv2 Block -> h_spatial', C['gat'], None),
    ('Prior Node Embed [NEW] (Excel->Graph->Embed) -> h_prior\n  ||  Gate: h=sigmoid*spatial + (1-sigmoid)*prior', C['fusion'], 5.5),
    ('TCN Block', C['tcn'], None),
    ('GRU', C['gru'], None),
    ('Pred Head  ||  Recon Head', C['pred'], 3.0),
], '03_prior_fusion.png')

draw_model('GATv2 + Dynamic Pearson Graph + TCN + GRU', [
    ('Input [B,60,51]', C['input'], None),
    ('Dynamic Pearson [NEW] (per-batch corr -> edges)  +  Static Edges', C['dyn_graph'], 4.5),
    ('GATv2 Block (with dynamic edges)', C['gat'], 2.8),
    ('TCN Block', C['tcn'], None),
    ('GRU', C['gru'], None),
    ('Pred Head  ||  Recon Head', C['pred'], 3.0),
], '04_dynamic_graph.png')

draw_model('GATv2 + Multi-Scale TCN + GRU', [
    ('Input [B,60,51]', C['input'], None),
    ('GATv2 Block', C['gat'], None),
    ('Multi-Scale TCN [NEW] (kernel=3,5,7 parallel branches)', C['tcn'], 4.0),
    ('GRU', C['gru'], None),
    ('Pred Head  ||  Recon Head', C['pred'], 3.0),
], '05_ms_tcn.png')

draw_model('USAD Dual-Decoder (GATv2+TCN+GRU backbone)', [
    ('Input [B,60,51]', C['input'], None),
    ('GATv2 + TCN + GRU Encoder', C['gru'], None),
    ('Decoder1 [NEW] -> r1  ||  Decoder2 [NEW] -> r2', C['decoder1'], 4.5),
    ('r1 -> re-encode -> Decoder2 -> r12 [2nd pass]', C['special'], 4.5),
    ('Loss = MSE(r1,x) + MSE(r2,x) + 0.5*MSE(r12,x)', C['pred'], 4.5),
], '06_usad_dual.png')

draw_model('Dynamic Pearson + USAD Dual-Decoder', [
    ('Input [B,60,51]', C['input'], None),
    ('Dynamic Pearson -> edges + GATv2 Block', C['dyn_graph'], 3.5),
    ('TCN + GRU Encoder', C['gru'], 2.5),
    ('Decoder1 -> r1  ||  Decoder2 -> r2', C['decoder1'], 3.5),
    ('r1 -> re-encode -> Decoder2 -> r12', C['special'], 3.5),
], '07_dynamic_usad.png')

draw_model('Dynamic Pearson + Prior Fusion + USAD (Our Full Model)', [
    ('Input [B,60,51]', C['input'], None),
    ('Dynamic Pearson -> edges  +  Prior Node Embed -> h_prior', C['dyn_graph'], 5.0),
    ('GATv2 -> Gate(h_spatial, h_prior) -> fused h', C['fusion'], 4.5),
    ('TCN + GRU Encoder', C['gru'], 2.5),
    ('Decoder1 -> r1  ||  Decoder2 -> r2', C['decoder1'], 3.5),
    ('r1 -> re-encode -> Decoder2 -> r12', C['special'], 3.5),
], '08_dyn_usad_prior.png')

# ====== External Comparison Models ======
EXTERNAL = [
    ('09_lstm_ae', 'LSTM-AE', [
        ('LSTM Encoder (2-layer)', C['gru'], None),
        ('Latent Bottleneck', C['prior'], None),
        ('LSTM Decoder (2-layer)', C['gru'], None),
        ('Reconstruction [B,60,51]', C['recon'], None)]),
    ('10_dagmm', 'DAGMM', [
        ('LSTM Encoder -> Latent z', C['gru'], None),
        ('Estimation Net (GMM params)', C['pred'], None),
        ('Joint Energy: Recon + Prob', C['special'], None)]),
    ('11_usad', 'USAD', [
        ('LSTM Encoder', C['gru'], None),
        ('Latent Space', C['prior'], None),
        ('Decoder1 + Decoder2', C['decoder1'], None),
        ('Adversarial Training (2-phase)', C['special'], None)]),
    ('12_mtad_gat', 'MTAD-GAT', [
        ('1D Conv + GAT (per-variable)', C['gat'], None),
        ('GRU (temporal modeling)', C['gru'], None),
        ('VAE: mu, sigma -> z', C['pred'], None),
        ('Reconstruction + Forecasting', C['recon'], None)]),
    ('13_mad_gan', 'MAD-GAN', [
        ('LSTM Generator', C['gru'], None),
        ('LSTM Discriminator', C['special'], None),
        ('DR-Score = lambda*Rec + (1-lambda)*Disc', C['pred'], None)]),
    ('14_anotrans', 'Anomaly Transformer', [
        ('Position Encoding', C['input'], None),
        ('Anomaly-Attention (Prior + Series)', C['time_attn'], None),
        ('Feed-Forward + LayerNorm', C['tcn'], None),
        ('Minimax: Recon + Association Discrepancy', C['special'], None)]),
    ('15_tranad', 'TranAD', [
        ('Transformer Encoder', C['time_attn'], None),
        ('Window Encoder + Focus Score', C['prior'], None),
        ('Decoder1 + Decoder2', C['decoder1'], None),
        ('Self-conditioning Distillation', C['special'], None)]),
    ('16_timesnet', 'TimesNet', [
        ('TimesBlock (FFT -> 2D Conv)', C['prior'], None),
        ('Multi-period Decomposition', C['tcn'], None),
        ('Feed-Forward + Residual', C['input'], None),
        ('Reconstruction', C['recon'], None)]),
    ('17_dcdetector', 'DCdetector', [
        ('Patch Embedding', C['input'], None),
        ('Dual Attention (Patch + Channel)', C['time_attn'], None),
        ('Contrastive: Permuted vs Normal', C['special'], None),
        ('Anomaly Score = Contrastive Divergence', C['pred'], None)]),
    ('18_gdn', 'GDN', [
        ('Sensor Embedding + Graph Structure', C['gat'], None),
        ('Attention-Based Forecasting', C['time_attn'], None),
        ('Deviation Score + Graph Pruning', C['special'], None)]),
    ('19_can', 'CAN', [
        ('Conv1D Feature Encoder', C['tcn'], None),
        ('Channel-Aware Graph (learned)', C['gat'], None),
        ('Multi-Head Temporal Attention', C['time_attn'], None),
        ('Reconstruction + Graph Regularization', C['recon'], None)]),
    ('20_gcn', 'GCN-TCN-GRU', [
        ('GCN (Mean Aggregation, 2-layer)', C['gat'], None),
        ('TCN (along variable dim)', C['tcn'], None),
        ('GRU -> Pred + Recon Heads', C['gru'], None),
        ('IQR Top-K Scoring', C['pred'], None)]),
]

for fname, title, layers in EXTERNAL:
    draw_model(title, layers, f'{fname}.png')

print(f'\nDone! {len(os.listdir(OUT_DIR))} diagrams saved to {OUT_DIR}')
