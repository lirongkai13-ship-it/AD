"""运行新增加的 5 个对比模型（2+2+1 分批）"""
import os, sys, time, subprocess, concurrent.futures, json

NEW_MODELS = ["TranAD", "TimesNet", "DCdetector", "GDN", "CAN"]
BATCH_SIZE = 2
PYTHON = r"D:/Anaconda/envs/swat_ad/python.exe"
SCRIPT = os.path.join(os.path.dirname(__file__), "run_single.py")
PROJECT = os.path.join(os.path.dirname(__file__), "..")


def run_one(name):
    t0 = time.time()
    proc = subprocess.run([PYTHON, "-u", SCRIPT, name], cwd=PROJECT,
                          capture_output=True, text=True)
    elapsed = time.time() - t0
    status = "FAILED"
    for line in (proc.stdout + proc.stderr).split("\n"):
        if f"RESULT:{name}:" in line:
            status = line.split("RESULT:")[-1].strip()
    print(f"\n  [{name}] {elapsed/60:.1f}min — {status}")
    # 打印关键行
    for line in (proc.stdout + proc.stderr).split("\n"):
        if any(k in line for k in ["F1=", "Epoch 00", "Training finished"]):
            print(f"    {line.strip()}")
    return name


def load_metrics(name):
    path = os.path.join(PROJECT, "results", "models", name, "metrics.json")
    if not os.path.exists(path):
        path = os.path.join(PROJECT, "outputs", name, "metrics.json")  # fallback
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def main():
    print(f"Models: {NEW_MODELS}")
    t_start = time.time()

    for i in range(0, len(NEW_MODELS), BATCH_SIZE):
        batch = NEW_MODELS[i:i + BATCH_SIZE]
        print(f"\n{'='*50}\nBATCH {i//BATCH_SIZE+1}: {batch}\n{'='*50}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=BATCH_SIZE) as ex:
            list(ex.map(run_one, batch))

    total = time.time() - t_start
    print(f"\nALL DONE in {total/60:.1f} min")

    # 汇总
    print(f"\n{'Model':<15} {'F1':>7} {'P':>7} {'R':>7} {'AUC':>8}")
    print("-" * 45)
    for name in NEW_MODELS:
        m = load_metrics(name)
        if m and "raw" in m:
            r = m["raw"]
            print(f"{name:<15} {r['f1']:7.4f} {r['precision']:7.4f} {r['recall']:7.4f} {str(r.get('roc_auc','?')):>8}")
        else:
            print(f"{name:<15} {'FAILED':>7}")


if __name__ == "__main__":
    main()
