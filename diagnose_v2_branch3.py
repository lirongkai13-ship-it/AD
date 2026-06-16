"""Branch 3 diagnostic: verify v2 model's upgraded Branch 3 is actually training."""
import sys, os, copy
import numpy as np
import torch, torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import prepare_data, build_pearson_edge_index, split_train_val, read_swat_csv
from utils import load_config, set_seed, get_device
from torch.utils.data import DataLoader
import importlib.util, pandas as pd
from sklearn.preprocessing import StandardScaler

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg = load_config('config_dev.yaml')
set_seed(42)

# ── Build graphs ──
dcfg = cfg['data']
nfd = pd.read_csv(dcfg['train_csv']); nfd.columns = [str(c).strip() for c in nfd.columns]
nfd = nfd[[c for c in nfd.columns if c not in ['Timestamp', 'Normal/Attack']]]
mfd = pd.read_csv(dcfg['test_csv']); mfd.columns = [str(c).strip() for c in mfd.columns]
common_cols = [c for c in nfd.columns if c in mfd.columns]
raw = nfd[common_cols].values.astype(np.float32)
tr, _, _, _ = split_train_val(raw, None, 0.2)
tv = StandardScaler().fit_transform(tr)
static_ei, _ = build_pearson_edge_index(tv)
bgp = importlib.util.spec_from_file_location('bpg','models_variants/prior_fusion/build_prior_graph.py')
bpgm = importlib.util.module_from_spec(bgp); bgp.loader.exec_module(bpgm)
prior_ei, prior_w = bpgm.build_prior_graph(common_cols)
static_ei = static_ei.to(device); prior_ei = prior_ei.to(device); prior_w = prior_w.to(device)

print("=" * 70)
print("DIAGNOSIS: v2 Branch 3 verification")
print("=" * 70)

# ── Load v2 model ──
from models_variants.tri_branch_v2.variant_model import TriBranch_USAD_v2

model = TriBranch_USAD_v2(
    nv=51, ws=60, static_edge_index=static_ei,
    prior_edge_index=prior_ei, prior_weights=prior_w,
    hidden_dim=32, gat_heads=2, dropout=0.2, latent_dim=64,
    encoder_mode='tri_branch_residual_gate',
    gamma_mode='fixed', gamma_value=0.05, gate_scale=1.0,
    use_temporal_attn_pooling=True,
    use_node_conditioned_fusion=True,
    gate_type='scalar',
).to(device)

model.train()
print(f"Model loaded. Params: {sum(p.numel() for p in model.parameters()):,}")

# ── Get a batch ──
_, val_ds, _, _, info = prepare_data(cfg)
loader = DataLoader(val_ds, batch_size=4, shuffle=False)
batch = next(iter(loader))
x = batch['x'].to(device)

print(f"\nInput batch shape: {x.shape}")

# ═══════════════════════════════════════════
# CHECK 1: Forward path — is Branch 3 used?
# ═══════════════════════════════════════════
print("\n" + "=" * 70)
print("CHECK 1: Forward path verification")
print("=" * 70)

# Add debug_print flag to encoder
model.encoder._debug_print = True
original_forward = model.encoder.forward

def debug_forward(self, x_input):
    b, w, n = x_input.shape

    # Branch 1
    xv = x_input.permute(0, 2, 1).reshape(b * n, 1, w)
    h_node = self.temporal_enc(xv).reshape(b, n, -1)

    # Branch 2
    edges = self.dyn_graph(x_input)
    hg = self.gat(x_input.permute(0, 2, 1), edges)
    hp = self.prior_embed().unsqueeze(0).expand(b, -1, -1)
    g = self.gate(torch.cat([hg, hp], -1))
    h_space = g * hg + (1 - g) * hp

    # Base fusion
    h_base = self.base_fusion(torch.cat([h_node, h_space], -1))

    print(f"\n  [DEBUG] Shapes inside encoder:")
    print(f"    x:            {x_input.shape}  range=[{x_input.min():.4f}, {x_input.max():.4f}]")
    print(f"    H_node:       {h_node.shape}  norm={h_node.norm():.4f}")
    print(f"    H_space:      {h_space.shape}  norm={h_space.norm():.4f}")
    print(f"    H_base:       {h_base.shape}  norm={h_base.norm():.4f}")

    # Branch 3
    if self.encoder_mode == "tri_branch_residual_gate" and self.global_temp is not None:
        h_global, node_gate, g_global, attn_w = self.global_temp(x_input, h_base)
        h_fuse = self.gated_fusion(h_base, h_global, node_gate)

        print(f"    H_global:     {h_global.shape}  norm={h_global.norm():.4f}")
        print(f"    node_gate:    {node_gate.shape}  mean={node_gate.mean():.4f} min={node_gate.min():.4f} max={node_gate.max():.4f}")
        print(f"    H_fuse:       {h_fuse.shape}  norm={h_fuse.norm():.4f}")
        print(f"    g_global:     {g_global.shape}  norm={g_global.norm():.4f}")
        print(f"    attn_w:       {attn_w.shape}  mean={attn_w.mean():.4f} min={attn_w.min():.4f} max={attn_w.max():.4f}")
        print(f"    (H_fuse - H_base).norm() = {(h_fuse - h_base).norm().item():.6f}")
        print(f"    gamma = {self.gated_fusion.gamma.item():.4f}")
        print(f"    gate_scale = {self.gated_fusion.gate_scale}")
    else:
        h_fuse = h_base
        print(f"    [WARNING] Branch 3 NOT USED — encoder_mode={self.encoder_mode}")

    # Latent
    z = h_fuse.reshape(b, n * h_fuse.shape[-1]) if self.use_flatten else h_fuse.mean(1)
    z = self.latent_proj(z)
    print(f"    z (latent):   {z.shape}  (from H_fuse: {'YES' if self.use_flatten else 'YES'})")

    self._debug_print = False
    return z, edges, {
        'h_node': h_node, 'h_space': h_space, 'h_base': h_base,
        'h_fuse': h_fuse, 'g_global': g_global if self.global_temp else None,
        'temporal_attn_weights': attn_w if self.global_temp else None,
        'node_gate': node_gate if self.global_temp else None}

model.encoder.forward = debug_forward.__get__(model.encoder)

# Run forward with debug
with torch.no_grad():
    r1, r2, r12, extras = model(x, static_ei)

fusion_diff = (extras['h_fuse'] - extras['h_base']).norm().item()
print(f"\n  >>> CONCLUSION: (H_fuse - H_base).norm() = {fusion_diff:.6f}")
if fusion_diff < 1e-4:
    print("  >>> [FAIL] Branch 3 has NO EFFECT on H_fuse!")
else:
    print(f"  >>> [PASS] Branch 3 IS injecting into H_fuse (diff={fusion_diff:.4f})")

# Restore original forward
model.encoder.forward = original_forward.__get__(model.encoder)


# ═══════════════════════════════════════════
# CHECK 2: Branch 3 on/off comparison
# ═══════════════════════════════════════════
print("\n" + "=" * 70)
print("CHECK 2: Branch 3 on/off comparison")
print("=" * 70)

model.eval()
with torch.no_grad():
    # Normal forward (Branch 3 ON)
    r1_on, r2_on, r12_on, _ = model(x, static_ei)

    # Disable Branch 3: manually replace H_fuse with H_base
    # We do this by temporarily setting gamma to 0
    original_gamma = model.encoder.gated_fusion.gamma.clone()
    model.encoder.gated_fusion.gamma.fill_(0.0)
    r1_off, r2_off, r12_off, _ = model(x, static_ei)
    model.encoder.gated_fusion.gamma.copy_(original_gamma)

    diff_r1 = (r1_on - r1_off).abs().mean().item()
    diff_r2 = (r2_on - r2_off).abs().mean().item()
    diff_r12 = (r12_on - r12_off).abs().mean().item()

    print(f"  r1 diff (on vs gamma=0): {diff_r1:.8f}")
    print(f"  r2 diff (on vs gamma=0): {diff_r2:.8f}")
    print(f"  r12 diff (on vs gamma=0): {diff_r12:.8f}")

    if diff_r1 < 1e-6 and diff_r2 < 1e-6:
        print("  >>> [FAIL] Branch 3 has NO effect on model output!")
    else:
        print(f"  >>> [PASS] Branch 3 changes model output (r1 diff={diff_r1:.6f})")


# ═══════════════════════════════════════════
# CHECK 3: Branch 3 gradients
# ═══════════════════════════════════════════
print("\n" + "=" * 70)
print("CHECK 3: Branch 3 gradient verification")
print("=" * 70)

model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
mse = nn.MSELoss()

x_batch = x
optimizer.zero_grad()
r1, r2, r12, _ = model(x_batch, static_ei)
loss = mse(r1, x_batch) + mse(r2, x_batch) + 0.5 * mse(r12, x_batch)
loss.backward()

branch3_keywords = ['global_temp', 'gated_fusion', 'temporal_score',
                    'fusion_mlp', 'gate_mlp', 'proj_in', 'attn', 'norm_attn']
print(f"\n  {'Param':<50s} {'Shape':<20s} {'req_grad':>8s} {'grad_none':>8s} {'grad_mean':>10s} {'grad_max':>10s}")
print(f"  {'-'*106}")

all_have_grad = True
for name, param in model.named_parameters():
    is_branch3 = any(kw in name for kw in branch3_keywords)
    if not is_branch3:
        continue
    grad_none = param.grad is None
    if grad_none:
        all_have_grad = False
    grad_mean = param.grad.abs().mean().item() if not grad_none else 0.0
    grad_max = param.grad.abs().max().item() if not grad_none else 0.0
    flag = " [FAIL]" if grad_none else ""
    print(f"  {name:<50s} {str(list(param.shape)):<20s} {str(param.requires_grad):>8s} {str(grad_none):>8s} {grad_mean:>10.6f} {grad_max:>10.6f}{flag}")

if all_have_grad:
    print(f"\n  >>> [PASS] All Branch 3 params have gradients")
else:
    print(f"\n  >>> [FAIL] Some Branch 3 params have grad=None — not in computation graph!")


# ═══════════════════════════════════════════
# CHECK 4: Optimizer coverage
# ═══════════════════════════════════════════
print("\n" + "=" * 70)
print("CHECK 4: Optimizer parameter coverage")
print("=" * 70)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
opt_param_ids = set()
for group in optimizer.param_groups:
    for p in group['params']:
        opt_param_ids.add(id(p))

total_trainable = 0
missing = []
branch3_missing = []
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    total_trainable += 1
    if id(param) not in opt_param_ids:
        missing.append(name)
        if any(kw in name for kw in branch3_keywords):
            branch3_missing.append(name)

print(f"  Trainable params: {total_trainable}")
print(f"  Optimizer params: {len(opt_param_ids)}")
print(f"  Missing from optimizer: {len(missing)}")

if branch3_missing:
    print(f"  >>> [FAIL] Branch 3 params missing from optimizer:")
    for n in branch3_missing:
        print(f"      {n}")
else:
    print(f"  >>> [PASS] All Branch 3 params are in optimizer")

if missing:
    print(f"  Non-Branch3 missing params ({len(missing)}):")
    for n in missing[:5]:
        print(f"      {n}")


# ═══════════════════════════════════════════
# CHECK 5: Parameter update verification
# ═══════════════════════════════════════════
print("\n" + "=" * 70)
print("CHECK 5: Parameter update (before vs after optimizer.step)")
print("=" * 70)

# Clone Branch 3 params before step
before_params = {}
for name, param in model.named_parameters():
    if any(kw in name for kw in branch3_keywords) and param.requires_grad:
        before_params[name] = param.detach().clone()

# Do one training step
optimizer.zero_grad()
r1, r2, r12, _ = model(x_batch, static_ei)
loss = mse(r1, x_batch) + mse(r2, x_batch) + 0.5 * mse(r12, x_batch)
loss.backward()
optimizer.step()

print(f"  {'Param':<50s} {'Before mean':>12s} {'After mean':>12s} {'Max diff':>12s} {'Updated?':>10s}")
print(f"  {'-'*100}")
any_updated = False
for name, before in before_params.items():
    after = dict(model.named_parameters())[name].detach()
    max_diff = (after - before).abs().max().item()
    before_mean = before.abs().mean().item()
    after_mean = after.abs().mean().item()
    updated = "YES" if max_diff > 1e-8 else "NO [FAIL]"
    if max_diff > 1e-8:
        any_updated = True
    print(f"  {name:<50s} {before_mean:>12.6f} {after_mean:>12.6f} {max_diff:>12.8f} {updated:>10s}")

if any_updated:
    print(f"\n  >>> [PASS] Branch 3 params ARE being updated by optimizer")
else:
    print(f"\n  >>> [FAIL] Branch 3 params NOT being updated!")


# ═══════════════════════════════════════════
# FINAL DIAGNOSIS
# ═══════════════════════════════════════════
print("\n" + "=" * 70)
print("FINAL DIAGNOSIS SUMMARY")
print("=" * 70)

checks = {
    '1. Branch 3 enters forward path': fusion_diff > 1e-4,
    '2. Branch 3 affects output (on/off test)': diff_r1 > 1e-6,
    '3. Branch 3 params have gradients': all_have_grad,
    '4. Branch 3 params in optimizer': len(branch3_missing) == 0,
    '5. Branch 3 params update after step': any_updated,
}
for check, passed in checks.items():
    status = "[PASS]" if passed else "[FAIL]"
    print(f"  {status}  {check}")

all_pass = all(checks.values())
if all_pass:
    print(f"\n  >>> Branch 3 is FULLY FUNCTIONAL. Training speed difference is NOT due to Branch 3 being skipped.")
    print(f"  >>> v2 training is ~12 min vs v1 ~26 min — likely due to different epoch counts or early stopping.")
else:
    print(f"\n  >>> [ACTION REQUIRED] One or more checks failed. See above for details.")
