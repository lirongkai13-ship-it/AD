import json
import os
import random
from typing import Dict

import numpy as np
import torch
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_cfg: str = "auto"):
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(obj: Dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def fit_iqr_params(errors: np.ndarray, eps: float = 1e-8):
    """
    只在 validation normal errors 上拟合归一化参数，避免 test 泄露。
    errors: [num_samples, num_variables]
    """
    median = np.median(errors, axis=0, keepdims=True)
    q1 = np.quantile(errors, 0.25, axis=0, keepdims=True)
    q3 = np.quantile(errors, 0.75, axis=0, keepdims=True)
    iqr = q3 - q1
    return {"median": median, "iqr": iqr, "eps": eps}


def apply_iqr_normalize(errors: np.ndarray, params: Dict):
    return (errors - params["median"]) / (params["iqr"] + params.get("eps", 1e-8))


def aggregate_topk_score(norm_var_errors: np.ndarray, topk: int):
    """
    norm_var_errors: [M, N]
    返回样本级异常分数 [M]
    """
    k = min(topk, norm_var_errors.shape[1])
    return np.sort(norm_var_errors, axis=1)[:, -k:].mean(axis=1)


def point_adjust(pred: np.ndarray, label: np.ndarray):
    """
    异常检测论文常见 point-adjust。
    只用于和部分论文公平比较；实际工程报警不要滥用。
    """
    pred = pred.astype(int).copy()
    label = label.astype(int)
    in_anomaly = False

    for i in range(len(label)):
        if label[i] == 1 and pred[i] == 1 and not in_anomaly:
            in_anomaly = True

            j = i
            while j >= 0 and label[j] == 1:
                pred[j] = 1
                j -= 1

            j = i
            while j < len(label) and label[j] == 1:
                pred[j] = 1
                j += 1

        elif label[i] == 0:
            in_anomaly = False

        if in_anomaly:
            pred[i] = 1

    return pred