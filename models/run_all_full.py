"""Sequential full-parameter training for external comparison models."""
import subprocess, sys, os, time

models = [
    "USAD", "DAGMM", "LSTM-AE", "MAD-GAN",
    "DCdetector", "TranAD", "MTAD-GAT", "CAN",
    "AnoTrans", "GDN", "TimesNet",
]
config = os.path.join(os.path.dirname(__file__), "base_config_full.yaml")
runner = os.path.join(os.path.dirname(__file__), "run_single.py")

print(f"Full-parameter external model training")
print(f"Config: {config}")
print(f"Models: {len(models)} total")
print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

for i, model_name in enumerate(models):
    print(f"\n[{i+1}/{len(models)}] {model_name} ...")
    t0 = time.time()
    ret = subprocess.run(
        [sys.executable, runner, "--model", model_name, "--config", config],
        cwd=os.path.dirname(__file__)
    )
    elapsed = (time.time() - t0) / 60
    status = "OK" if ret.returncode == 0 else f"FAIL({ret.returncode})"
    print(f"  {status}  time={elapsed:.1f}min")

print(f"\nDone! {time.strftime('%Y-%m-%d %H:%M:%S')}")
