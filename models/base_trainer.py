"""对比模型共享训练器 —— 所有对比模型复用此模块"""
import os, sys, json, time, argparse
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_loader import prepare_data
from utils import (load_config, set_seed, get_device, ensure_dir,
                   fit_iqr_params, apply_iqr_normalize, aggregate_topk_score,
                   point_adjust, save_json)
from sklearn.metrics import (average_precision_score, precision_recall_fscore_support, roc_auc_score)


class BaseTrainer:
    def __init__(self, model, model_name, config_path):
        self.model_name = model_name
        self.cfg = load_config(config_path)
        self.device = get_device(self.cfg["train"].get("device", "cuda"))
        self.save_dir = os.path.join(self.cfg["output"]["base_dir"], model_name)
        ensure_dir(self.save_dir)

        set_seed(int(self.cfg["train"]["seed"]))
        self.use_amp = self.cfg["train"].get("amp", False) and self.device.type == "cuda"
        self.scaler = torch.GradScaler("cuda") if self.use_amp else None

        # 从父目录的 config.yaml 读取数据路径等信息
        parent_cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))
        self.data_cfg = {
            "data": {
                "train_csv": parent_cfg["data"]["train_csv"],
                "test_csv": parent_cfg["data"]["test_csv"],
                "timestamp_col": parent_cfg["data"]["timestamp_col"],
                "label_col": parent_cfg["data"]["label_col"],
                "normal_label": parent_cfg["data"]["normal_label"],
                "val_ratio": parent_cfg["data"]["val_ratio"],
                "window_size": self.cfg["data"]["window_size"],
                "train_stride": self.cfg["data"]["train_stride"],
                "val_stride": self.cfg["data"]["val_stride"],
                "test_stride": self.cfg["data"]["test_stride"],
                "horizon": self.cfg["data"]["horizon"],
                "label_mode": self.cfg["data"]["label_mode"],
                "corr_threshold": self.cfg["data"]["corr_threshold"],
            }
        }
        train_ds, val_ds, test_ds, edge_index, info = prepare_data(self.data_cfg)
        self.datasets = (train_ds, val_ds, test_ds, edge_index)
        self.info = info
        self.edge_index = edge_index  # MTAD-GAT 等需要

        self.model = model.to(self.device)
        nw = self.cfg["train"].get("num_workers", 0)
        bs = int(self.cfg["train"]["batch_size"])
        self.train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                                       num_workers=nw, pin_memory=True)
        self.val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                                     num_workers=nw, pin_memory=True)
        self.test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False,
                                      num_workers=nw, pin_memory=True)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=float(self.cfg["train"]["lr"]),
                                          weight_decay=float(self.cfg["train"]["weight_decay"]))
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5,
            patience=self.cfg["train"].get("lr_patience", 3), min_lr=1e-6)
        self.early_stop = self.cfg["train"].get("early_stop_patience", 5)
        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0

    def train_epoch(self, epoch):
        self.model.train()
        total_loss, mse = 0.0, nn.MSELoss()
        pbar = tqdm(self.train_loader, desc=f"E{epoch}", leave=False)
        for batch in pbar:
            x = batch["x"].to(self.device)
            self.optimizer.zero_grad()
            if self.use_amp and self.scaler:
                with torch.autocast(device_type="cuda"):
                    out = self.model_forward(batch)
                    loss = self.compute_loss(out, batch, mse)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                out = self.model_forward(batch)
                loss = self.compute_loss(out, batch, mse)
                loss.backward()
                self.optimizer.step()
            total_loss += loss.item() * x.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        return total_loss / len(self.train_loader.dataset)

    def model_forward(self, batch):
        """子类重写此方法来定义前向传播"""
        x = batch["x"].to(self.device)
        return self.model(x)

    def compute_loss(self, out, batch, mse):
        """默认重构损失"""
        x = batch["x"].to(self.device)
        if isinstance(out, tuple):
            recon = out[0]
        else:
            recon = out
        return mse(recon, x)

    @torch.no_grad()
    def validate_epoch(self, loader, desc="val"):
        self.model.eval()
        total_loss, mse = 0.0, nn.MSELoss()
        for batch in tqdm(loader, desc=desc, leave=False):
            x = batch["x"].to(self.device)
            out = self.model_forward(batch)
            loss = self.compute_loss(out, batch, mse)
            total_loss += loss.item() * x.size(0)
        return total_loss / len(loader.dataset)

    def train(self):
        epochs = int(self.cfg["train"]["epochs"])
        history = []
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate_epoch(self.val_loader, "val")
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            lr0 = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(val_loss)
            lr1 = self.optimizer.param_groups[0]["lr"]
            lr_s = f"| lr {lr0:.2e}" + (f" -> {lr1:.2e}" if lr1 < lr0 else "")
            print(f"Epoch {epoch:03d} | train {train_loss:.6f} | val {val_loss:.6f} {lr_s}")
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                torch.save({"model": self.model.state_dict(), "best_val_loss": float(val_loss)},
                           os.path.join(self.save_dir, "best_model.pt"))
            else:
                self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.early_stop:
                print(f"Early stopping at epoch {epoch}")
                break
        save_json({"history": history, "best_val_loss": float(self.best_val_loss)},
                  os.path.join(self.save_dir, "train_history.json"))
        print("Training finished.")

    @torch.no_grad()
    def evaluate(self):
        ckpt_path = os.path.join(self.save_dir, "best_model.pt")
        if os.path.exists(ckpt_path):
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device)["model"])
        self.model.eval()

        # 收集 val 误差 → 拟合 IQR
        val_errors = self._collect_errors(self.val_loader)
        iqr_params = fit_iqr_params(val_errors)
        val_norm = apply_iqr_normalize(val_errors, iqr_params)
        val_score = aggregate_topk_score(val_norm, topk=5)
        threshold = float(np.quantile(val_score, 0.995))

        # 收集 test 误差 → 评分
        test_errors, test_labels = self._collect_errors(self.test_loader, collect_labels=True)
        test_norm = apply_iqr_normalize(test_errors, iqr_params)
        test_score = aggregate_topk_score(test_norm, topk=5)
        test_pred = (test_score > threshold).astype(int)

        np.save(os.path.join(self.save_dir, "test_score.npy"), test_score)
        np.save(os.path.join(self.save_dir, "test_labels.npy"), test_labels)
        np.save(os.path.join(self.save_dir, "test_pred.npy"), test_pred)

        metrics = {"threshold": float(threshold), "topk": 5}
        if test_labels is not None:
            p, r, f1, _ = precision_recall_fscore_support(test_labels, test_pred, average="binary", zero_division=0)
            metrics["raw"] = {"precision": float(p), "recall": float(r), "f1": float(f1)}
            try:
                metrics["raw"]["roc_auc"] = float(roc_auc_score(test_labels, test_score))
            except:
                metrics["raw"]["roc_auc"] = None
            try:
                metrics["raw"]["pr_auc"] = float(average_precision_score(test_labels, test_score))
            except:
                metrics["raw"]["pr_auc"] = None

            pred_pa = point_adjust(test_pred, test_labels)
            p, r, f1, _ = precision_recall_fscore_support(test_labels, pred_pa, average="binary", zero_division=0)
            metrics["point_adjust"] = {"precision": float(p), "recall": float(r), "f1": float(f1)}
            metrics["point_adjust"]["roc_auc"] = metrics["raw"]["roc_auc"]

        save_json(metrics, os.path.join(self.save_dir, "metrics.json"))
        print(f"Threshold: {threshold:.2f}")
        print(f"Raw F1: {metrics.get('raw',{}).get('f1','N/A')}, "
              f"PA F1: {metrics.get('point_adjust',{}).get('f1','N/A')}")
        return metrics

    def _collect_errors(self, loader, collect_labels=False):
        all_errors, all_labels = [], []
        for batch in tqdm(loader, desc="eval", leave=False):
            x = batch["x"].to(self.device)
            out = self.model_forward(batch)
            if isinstance(out, tuple):
                recon = out[0]
            else:
                recon = out
            err = (recon - x).abs().mean(dim=1).cpu().numpy()  # [B, N]
            all_errors.append(err)
            if collect_labels and "label" in batch:
                all_labels.append(batch["label"].cpu().numpy())
        errors = np.concatenate(all_errors, axis=0)
        labels = np.concatenate(all_labels, axis=0) if all_labels else None
        return (errors, labels) if collect_labels else errors

    def run(self):
        print(f"\n{'='*50}\n{self.model_name}\n{'='*50}")
        print(f"Params: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Samples: train={len(self.datasets[0]):,} val={len(self.datasets[1]):,} test={len(self.datasets[2]):,}")
        t0 = time.time()
        self.train()
        train_time = time.time() - t0
        t0 = time.time()
        metrics = self.evaluate()
        eval_time = time.time() - t0
        print(f"Time: train={train_time/60:.1f}min eval={eval_time/60:.1f}min total={train_time/60+eval_time/60:.1f}min")
        return metrics
