"""消融实验对比图：非USAD + USAD系"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "plots")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── 非USAD消融 (最新 config_dev stride=10 重跑) ───
non_usad_data = [
    ("Baseline\nGATv2+TCN+GRU", 0.6676, "#607D8B"),
    ("+Temporal\nAttention",     0.7063, "#A5D6A7"),
    ("+Prior\nFusion",           0.6988, "#FFCC80"),
    ("+Prior\nDynamic",          0.6584, "#FFAB91"),
    ("+MultiScale\nTCN",         0.6916, "#FFB74D"),
    ("+Dynamic\nPearson",        0.6596, "#64B5F6"),
    ("+DynGraph\nDiff",          0.6618, "#42A5F5"),
    ("+DynPrior\nFeatFusion",    0.7122, "#CE93D8"),
]

# ─── USAD系 (均用 config_dev stride=10) ───
usad_data = [
    ("static_usad\n(no dynamic)",   0.7296, "#90A4AE"),
    ("dynamic_usad\n(+dynamic)",    0.7494, "#1E88E5"),
    ("usad_dual\n(original arch)",  0.7446, "#BBDEFB"),
    ("dyn_ms\n_usad (+MS-TCN)",     0.7461, "#FFB74D"),
    ("dyn_usad\n_prior",            0.7145, "#FFCC80"),
    ("dyn_temporal\n_usad (+TA)",   0.7126, "#A5D6A7"),
]


def draw_ablation_chart(data, title_prefix, filename, save_dir, ref_idx=0):
    """通用消融柱状图：左侧 F1 + 右侧 delta"""
    names = [x[0] for x in data]
    values = [x[1] for x in data]
    colors = [x[2] for x in data]
    baseline_val = values[ref_idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(12, len(data)*1.2), 5.5))

    # ── 左: F1 ──
    x = np.arange(len(data))
    bars = ax1.bar(x, values, color=colors, edgecolor="#37474F", linewidth=1.0, width=0.6)
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.004,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.axhline(y=baseline_val, color="#78909C", linestyle="--", alpha=0.6, linewidth=1.2)
    ax1.text(len(data)-0.8, baseline_val + 0.003, f"Ref: {baseline_val:.4f}",
             fontsize=8, color="#78909C")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=8)
    ax1.set_ylabel("F1 Score", fontsize=11, fontweight="bold")
    ax1.set_title(f"{title_prefix} — F1 Score", fontsize=12, fontweight="bold")
    ax1.set_ylim(min(values)*0.95, max(values)*1.04)
    ax1.grid(axis="y", alpha=0.25)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ── 右: Delta ──
    deltas = [v - baseline_val for v in values]
    d_colors = ["#2E7D32" if d > 0 else "#C62828" for d in deltas]
    for i, d in enumerate(deltas):
        if i == ref_idx:
            d_colors[i] = "#9E9E9E"
    bars2 = ax2.bar(x, deltas, color=d_colors, edgecolor="#37474F", linewidth=1.0, width=0.6)
    for bar, d in zip(bars2, deltas):
        offset = 0.002 if d >= 0 else -0.006
        ax2.text(bar.get_x() + bar.get_width()/2, d + offset,
                 f"{d:+.4f}", ha="center", va="bottom" if d >= 0 else "top",
                 fontsize=10, fontweight="bold")
    ax2.axhline(y=0, color="black", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=8)
    ax2.set_ylabel("Delta F1", fontsize=11, fontweight="bold")
    ax2.set_title(f"{title_prefix} — Delta vs Reference", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", alpha=0.25)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.suptitle(f"{title_prefix}\nSWaT Dataset | config_dev: stride=10, epoch=5, batch=256, hidden=32",
                 fontsize=13, fontweight="bold", y=1.04)
    plt.tight_layout()
    path = os.path.join(save_dir, filename)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    print(f"[OK] {filename}")


# ─── 生成两张图 ───
draw_ablation_chart(non_usad_data, "Ablation Study: Non-USAD Variants",
                    "ablation_non_usad.png", OUT_DIR, ref_idx=0)
draw_ablation_chart(usad_data, "USAD-Based Models Comparison",
                    "ablation_usad.png", OUT_DIR, ref_idx=0)

print(f"\nSaved to: {OUT_DIR}")
