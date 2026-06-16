"""对比模型结果可视化 —— 直接读取 summary JSON"""
import os, json
import matplotlib.pyplot as plt
import numpy as np

def plot_comparison(summary_path=None):
    if summary_path is None:
        summary_path = os.path.join(os.path.dirname(__file__), "..", "quick_summary.json")
    if not os.path.exists(summary_path):
        print(f"Summary not found: {summary_path}")
        return

    with open(summary_path) as f:
        summary = json.load(f)

    names = list(summary.keys())
    raw_f1 = [summary[n]["raw_f1"] for n in names]
    pa_f1  = [summary[n]["pa_f1"] for n in names]
    precision = [summary[n]["raw_p"] for n in names]
    recall    = [summary[n]["raw_r"] for n in names]
    roc_auc   = [summary[n].get("roc_auc", 0) for n in names]

    # 高亮 Yours 模型
    colors_f1 = ["#ff6b6b" if "Yours" in n else "#4ecdc4" for n in names]
    x = np.arange(len(names))
    width = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

    # ── 图1: F1 (Raw + PA) ──
    ax = axes[0]
    b1 = ax.bar(x - width/2, raw_f1, width, label="Raw F1", color=colors_f1, edgecolor="white")
    b2 = ax.bar(x + width/2, pa_f1, width, label="PA F1", color=["#ffa07a" if "Yours" in n else "#74b9ff" for n in names], edgecolor="white")
    for bar, v in zip(b1, raw_f1):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for bar, v in zip(b2, pa_f1):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("F1", fontsize=11)
    ax.set_title("F1 Comparison", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.3)

    # ── 图2: Precision + Recall ──
    ax = axes[1]
    b3 = ax.bar(x - width/2, precision, width, label="Precision", color="#27ae60", edgecolor="white")
    b4 = ax.bar(x + width/2, recall, width, label="Recall", color="#f39c12", edgecolor="white")
    for bar, v in zip(b3, precision):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    for bar, v in zip(b4, recall):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Precision & Recall", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.3)

    # ── 图3: ROC-AUC ──
    ax = axes[2]
    auc_colors = ["#ff6b6b" if "Yours" in n else plt.cm.viridis(i/len(names)) for i, n in enumerate(names)]
    b5 = ax.bar(x, roc_auc, 0.5, color=auc_colors, edgecolor="white")
    for bar, v in zip(b5, roc_auc):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.002, f"{v:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("ROC-AUC", fontsize=11)
    ax.set_title("ROC-AUC Comparison", fontsize=12, fontweight="bold")
    ax.set_ylim(0.90, 0.97)
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Anomaly Detection Model Comparison — SWaT Dataset", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()

    save_path = os.path.join(os.path.dirname(__file__), "..", "results", "plots", "model_comparison.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {save_path}")
    plt.close()


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    plot_comparison(path)
