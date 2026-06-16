"""运行所有对比模型（2个一批并行）并收集结果"""
import os, sys, json, time, subprocess, concurrent.futures

# ─── 模型注册 ──────────────────────────────────────────────
MODEL_ORDER = [
    "LSTM-AE",       # ~10 min
    "USAD",          # ~15 min
    "DAGMM",         # ~12 min
    "MAD-GAN",       # ~15 min
    "AnoTrans",      # ~20 min
    "TranAD",        # ~20 min
    "TimesNet",      # ~15 min
    "DCdetector",    # ~12 min
    "GDN",           # ~10 min
    "CAN",           # ~15 min
]

BATCH_SIZE = 2                    # 每批并行的模型数
PYTHON_EXE = sys.executable       # 使用当前 conda 环境的 python


def run_model_subprocess(name):
    """用子进程跑单个模型"""
    script = os.path.join(os.path.dirname(__file__), "run_single.py")
    t0 = time.time()
    proc = subprocess.run(
        [PYTHON_EXE, "-u", script, name],
        cwd=os.path.join(os.path.dirname(__file__), ".."),
        capture_output=True, text=True,
    )
    elapsed = time.time() - t0
    # 解析输出
    status = "FAILED"
    for line in proc.stdout.split("\n") + proc.stderr.split("\n"):
        if f"RESULT:{name}:" in line:
            status = line.split("RESULT:")[-1].strip()
    print(f"\n  [{name}] finished in {elapsed/60:.1f} min — {status}")
    if proc.stdout:
        # 只打印关键行
        for line in proc.stdout.split("\n"):
            if any(k in line for k in ["F1", "Epoch", "raw", "Threshold", "Training", "Saved"]):
                print(f"    {line.strip()}")
    return name


def load_metrics(name):
    """加载已跑模型的指标"""
    path = os.path.join(os.path.dirname(__file__), "..", "results",
                        "models", name, "metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def main():
    print(f"Running {len(MODEL_ORDER)} models in batches of {BATCH_SIZE}")
    print(f"Models: {MODEL_ORDER}")
    t_start = time.time()

    # 分批并行执行
    for i in range(0, len(MODEL_ORDER), BATCH_SIZE):
        batch = MODEL_ORDER[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(MODEL_ORDER) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"\n{'='*60}")
        print(f"BATCH {batch_num}/{total_batches}: {batch}")
        print(f"{'='*60}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
            futures = [executor.submit(run_model_subprocess, name) for name in batch]
            concurrent.futures.wait(futures)

    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"ALL MODELS COMPLETED in {total/60:.1f} min")

    # ─── 收集结果并生成汇总 ───
    summary = {}
    for name in MODEL_ORDER:
        m = load_metrics(name)
        if m and "raw" in m:
            summary[name] = {
                "status": "OK",
                "raw_f1": m["raw"]["f1"],
                "raw_p": m["raw"]["precision"],
                "raw_r": m["raw"]["recall"],
                "roc_auc": m["raw"].get("roc_auc"),
                "pa_f1": m["point_adjust"]["f1"],
                "threshold": float(m["threshold"]),
            }
        else:
            summary[name] = {"status": "FAILED"}

    # 保存汇总
    save_path = os.path.join(os.path.dirname(__file__), "..",
                             "results", "summary", "comparison_summary.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {save_path}")

    # ─── 打印对比表 ───
    print(f"\n{'Model':<16} {'F1':>7} {'P':>7} {'R':>7} {'ROC-AUC':>8} {'PA-F1':>7}")
    print("-" * 60)
    for name in MODEL_ORDER:
        s = summary.get(name, {})
        if "raw_f1" in s:
            print(f"{name:<16} {s['raw_f1']:7.4f} {s['raw_p']:7.4f} {s['raw_r']:7.4f} {str(s['roc_auc']):>8} {s['pa_f1']:7.4f}")
        else:
            print(f"{name:<16} {'FAILED':>7}")

    return summary


if __name__ == "__main__":
    main()
