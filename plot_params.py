"""模型参数量对比图"""
import matplotlib.pyplot as plt
import numpy as np

models = [
    ("Baseline\n(Yours)",   128199, "#e74c3c"),
    ("LSTM-AE",             133107, "#3498db"),
    ("MAD-GAN",             150000, "#3498db"),
    ("GDN",                 160083, "#2ecc71"),
    ("CAN",                 169011, "#2ecc71"),
    ("DCdetector",          173598, "#2ecc71"),
    ("TranAD",              177603, "#2ecc71"),
    ("TimesNet",            101303, "#2ecc71"),
    ("USAD",                250000, "#95a5a6"),
    ("DAGMM",               300000, "#95a5a6"),
    ("AnoTrans",            500000, "#95a5a6"),
]

fig, ax = plt.subplots(figsize=(12, 6))
names = [m[0] for m in models]
params = [m[1] for m in models]
colors = [m[2] for m in models]

bars = ax.barh(range(len(models)), params, color=colors, edgecolor='white', height=0.6)

# 标注数值
for i, (bar, p) in enumerate(zip(bars, params)):
    ax.text(p + 5000, i, f'{p/1000:.0f}K', va='center', fontsize=10, fontweight='bold')

# 标注训练时长（估算）
times = {
    "Baseline\n(Yours)": "~25 min",
    "LSTM-AE": "~10 min",
    "MAD-GAN": "~15 min",
    "GDN": "~12 min",
    "CAN": "~15 min",
    "DCdetector": "~15 min",
    "TranAD": "~25 min",
    "TimesNet": "~20 min",
    "USAD": "~15 min",
    "DAGMM": "~12 min",
    "AnoTrans": "~20 min",
}
for i, (bar, name) in enumerate(zip(bars, models)):
    t = times.get(name[0], "?")
    ax.text(bar.get_width() + 5000, bar.get_y() + bar.get_height()/2 - 0.15,
            t, va='center', fontsize=8, color='gray')

ax.set_yticks(range(len(models)))
ax.set_yticklabels(names, fontsize=9)
ax.set_xlabel('Parameters', fontsize=12)
ax.axvline(x=128199, color='#e74c3c', linestyle='--', alpha=0.5, label='Baseline (128K)')
ax.set_title('Model Parameter Comparison', fontsize=14, fontweight='bold')
ax.legend(fontsize=9)
ax.set_xlim(0, max(params) * 1.25)
plt.tight_layout()
plt.savefig('results/plots/param_comparison.png', dpi=200, bbox_inches='tight')
print('Saved to results/plots/param_comparison.png')
plt.close()
