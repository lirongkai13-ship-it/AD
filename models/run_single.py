"""运行单个对比模型（供 run_all.py 并行调用）"""
import sys, os, json, time, torch, traceback, importlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.base_trainer import BaseTrainer

# ─── 模型注册 ──────────────────────────────────────────────
MODELS = {
    "LSTM-AE":  ("models.lstm_ae.model",   "LSTMAE",
                 dict(n_vars=51, window=60, hidden=64, num_layers=2, dropout=0.1)),
    "DAGMM":    ("models.dagmm.model",     "DAGMM",
                 dict(n_vars=51, window=60, hidden=64, latent=16, n_gmm=4, dropout=0.1)),
    "USAD":     ("models.usad.model",      "USAD",
                 dict(n_vars=51, window=60, hidden=64, latent=32, dropout=0.1)),
    "MTAD-GAT": ("models.mtad_gat.model",  "MTADGAT",
                 dict(n_vars=51, window=60, hidden=48, heads=2, latent=32, dropout=0.1)),
    "MAD-GAN":  ("models.mad_gan.model",   "MADGAN",
                 dict(n_vars=51, window=60, noise_dim=32, hidden=64)),
    "AnoTrans":    ("models.ano_trans.model",   "AnomalyTransformer",
                    dict(n_vars=51, window=60, d_model=64, n_heads=4, n_layers=2, dropout=0.1)),
    "TranAD":      ("models.tranad.model",      "TranAD",
                    dict(n_vars=51, window=60, d_model=48, n_heads=4, n_layers=2, dropout=0.1)),
    "TimesNet":    ("models.timesnet.model",    "TimesNet",
                    dict(n_vars=51, window=60, d_model=256, n_blocks=4, dropout=0.1)),
    "DCdetector":  ("models.dcdetector.model",  "DCdetector",
                    dict(n_vars=51, window=60, d_model=96, n_heads=4, dropout=0.1)),
    "GDN":         ("models.gdn.model",         "GDN",
                    dict(n_vars=51, window=60, hidden=96, top_k=30, dropout=0.1)),
    "CAN":         ("models.can.model",         "CAN",
                    dict(n_vars=51, window=60, d_model=96, n_heads=4, dropout=0.1)),
    "GCN":         ("models.gcn.model",         "GCN_TCN_GRU",
                    dict(num_variables=51, window_size=60, hidden_dim=32,
                         gru_hidden=32, tcn_channels=32, dropout=0.2)),
}


def run_one_model(name, config_path):
    """运行单个模型并返回指标"""
    print(f"\n{'#'*60}\n#  {name}\n{'#'*60}")
    try:
        path, cls_name, kwargs = MODELS[name]
        mod = importlib.import_module(path)
        cls = getattr(mod, cls_name)
        model = cls(**kwargs)
        trainer = BaseTrainer(model, name, config_path)

        # MTAD-GAT 需要 edge_index
        if name == "MTAD-GAT":
            edge = trainer.edge_index.to(trainer.device)
            trainer.model_forward = lambda batch: trainer.model(
                batch["x"].to(trainer.device), edge)

        # USAD 自定义训练
        if name == "USAD":
            _orig_fwd = trainer.model_forward
            _orig_loss = trainer.compute_loss
            def usad_forward(batch):
                x = batch["x"].to(trainer.device)
                return trainer.model(x)  # returns (r1, r2)
            def usad_loss(out, batch, mse):
                x = batch["x"].to(trainer.device)
                r1, r2 = out
                return mse(r1, x) + mse(r2, x) + 0.1 * mse(r1, r2.detach())
            trainer.model_forward = usad_forward
            trainer.compute_loss = usad_loss
            trainer._eval_forward = lambda batch: trainer.model(batch["x"].to(trainer.device))[0]

        metrics = trainer.run()
        return {"name": name, "metrics": metrics, "status": "OK"}
    except Exception as e:
        print(f"ERROR in {name}: {e}")
        traceback.print_exc()
        return {"name": name, "metrics": None, "status": f"FAILED"}


if __name__ == "__main__":
    model_name = sys.argv[1]
    config_path = os.path.join(os.path.dirname(__file__), "base_config.yaml")
    result = run_one_model(model_name, config_path)
    status = result.get("status", "?")
    print(f"\n=== RESULT:{model_name}:{status} ===")
    if result.get("metrics") and "raw" in result["metrics"]:
        m = result["metrics"]["raw"]
        print(f"F1={m['f1']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} AUC={m.get('roc_auc','N/A')}")
