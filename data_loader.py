from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


def read_swat_csv(path: str, timestamp_col: str, label_col: Optional[str]):
    """
    读取 SWaT CSV。

    会做几件事：
    1. 清理列名前后空格
    2. 删除 Timestamp
    3. 单独取出 Normal/Attack 标签列
    4. 只保留数值特征
    5. 前向/后向填充缺失值
    """
    df = pd.read_csv(path)

    # SWaT CSV 的列名经常带空格，例如 " MV101"
    df.columns = [str(c).strip() for c in df.columns]

    if timestamp_col and timestamp_col in df.columns:
        df = df.drop(columns=[timestamp_col])

    labels = None
    if label_col and label_col in df.columns:
        labels = df[label_col].copy()
        df = df.drop(columns=[label_col])

    # 所有过程变量转成数值
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 删除完全无法转成数值的列
    df = df.dropna(axis=1, how="all")

    # 填充少量缺失值
    df = df.ffill().bfill()

    return df, labels


def build_labels(raw_labels, normal_label="Normal"):
    """
    Normal -> 0
    Attack -> 1
    """
    if raw_labels is None:
        return None

    if raw_labels.dtype == object:
        y = (
            raw_labels.astype(str)
            .str.strip()
            .ne(str(normal_label))
            .astype(int)
            .values
        )
    else:
        y = raw_labels.astype(int).values

    return y.astype(np.int64)


def split_train_val(values: np.ndarray, labels: Optional[np.ndarray], val_ratio: float):
    """
    normal.csv 按时间顺序切分 train / val。
    不 shuffle，避免时间序列泄露。
    """
    total = len(values)
    split_idx = int(total * (1.0 - val_ratio))
    split_idx = max(1, min(split_idx, total - 1))

    train_values = values[:split_idx]
    val_values = values[split_idx:]

    if labels is None:
        train_labels = None
        val_labels = None
    else:
        train_labels = labels[:split_idx]
        val_labels = labels[split_idx:]

    return train_values, val_values, train_labels, val_labels


def make_windows(
    values: np.ndarray,
    labels: Optional[np.ndarray],
    window_size: int,
    horizon: int,
    stride: int,
    label_mode: str = "future",
):
    """
    values: [T, N]

    返回：
      x: [M, W, N]
      y_future: [M, N]
      y_recon: [M, W, N]
      y_labels: [M]

    label_mode:
      future:
        标签只看 future_idx。
        推荐作为主实验，更严格，score 和标签时间点对齐。

      window:
        从 start 到 future_idx 中，只要有任意 Attack，则该窗口为异常。
        检测更宽松，但会把异常标签向前扩散。
    """
    xs, yfs, yrs, ylabels = [], [], [], []

    total = len(values)
    end_limit = total - window_size - horizon + 1

    if end_limit <= 0:
        raise ValueError(
            f"数据长度 {total} 太短，无法构造窗口："
            f"window_size={window_size}, horizon={horizon}"
        )

    for start in range(0, end_limit, stride):
        end = start + window_size
        future_idx = end + horizon - 1

        x = values[start:end]
        y_future = values[future_idx]
        y_recon = x

        xs.append(x)
        yfs.append(y_future)
        yrs.append(y_recon)

        if labels is not None:
            if label_mode == "future":
                lab = int(labels[future_idx])
            elif label_mode == "window":
                lab = int(labels[start:future_idx + 1].max())
            else:
                raise ValueError(f"Unknown label_mode: {label_mode}")

            ylabels.append(lab)

    xs = np.asarray(xs, dtype=np.float32)
    yfs = np.asarray(yfs, dtype=np.float32)
    yrs = np.asarray(yrs, dtype=np.float32)

    if labels is not None:
        ylabels = np.asarray(ylabels, dtype=np.int64)
    else:
        ylabels = None

    return xs, yfs, yrs, ylabels


def build_pearson_edge_index(
    train_values: np.ndarray,
    corr_threshold: float = 0.3,
    self_loop: bool = True,
):
    """
    只用 normal train 数据构建 Pearson 静态图。
    不使用 val，不使用 merged test，避免信息泄露。

    train_values: [T, N]
    """
    # 某些常量列会导致 std=0，产生 NaN/Inf 的警告，用 nan_to_num 统一处理
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(train_values.T)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    n = corr.shape[0]
    edges = []

    for i in range(n):
        for j in range(n):
            if i == j:
                if self_loop:
                    edges.append([i, j])
            else:
                if abs(corr[i, j]) >= corr_threshold:
                    edges.append([i, j])

    # 极端情况下阈值太高，至少保留自环
    if len(edges) == 0:
        edges = [[i, i] for i in range(n)]

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return edge_index, corr


class SWaTWindowDataset(Dataset):
    def __init__(self, x, y_future, y_recon, labels=None):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y_future = torch.tensor(y_future, dtype=torch.float32)
        self.y_recon = torch.tensor(y_recon, dtype=torch.float32)
        self.labels = None if labels is None else torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        item = {
            "x": self.x[idx],
            "y_future": self.y_future[idx],
            "y_recon": self.y_recon[idx],
        }

        if self.labels is not None:
            item["label"] = self.labels[idx]

        return item

class SWaTDynamicWindowDataset(Dataset):
    """
    动态滑窗 Dataset。

    适合 merged.csv / test.csv 这种很大的测试集。
    不提前生成 [M, W, N] 大数组，而是在 __getitem__ 时按需切窗口。
    """

    def __init__(
        self,
        values: np.ndarray,
        labels: Optional[np.ndarray],
        window_size: int,
        horizon: int,
        stride: int,
        label_mode: str = "future",
    ):
        self.values = values.astype(np.float32)
        self.labels = labels
        self.window_size = int(window_size)
        self.horizon = int(horizon)
        self.stride = int(stride)
        self.label_mode = label_mode

        total = len(self.values)
        self.num_samples = (total - self.window_size - self.horizon) // self.stride + 1

        if self.num_samples <= 0:
            raise ValueError(
                f"数据长度 {total} 太短，无法构造窗口："
                f"window_size={self.window_size}, horizon={self.horizon}, stride={self.stride}"
            )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.window_size
        future_idx = end + self.horizon - 1

        # .copy() 切断对完整 values 数组的引用
        # 避免 DataLoader num_workers>0 时 pickle 整个 [T,N] 底层存储
        x_win = self.values[start:end].copy()            # [W, N]
        y_future = self.values[future_idx].copy()        # [N]

        item = {
            "x": torch.from_numpy(x_win),
            "y_future": torch.from_numpy(y_future),
            "y_recon": torch.from_numpy(x_win),          # 重构目标 = 输入窗口
        }

        if self.labels is not None:
            if self.label_mode == "future":
                lab = int(self.labels[future_idx])
            elif self.label_mode == "window":
                lab = int(self.labels[start:future_idx + 1].max())
            else:
                raise ValueError(f"Unknown label_mode: {self.label_mode}")

            item["label"] = torch.tensor(lab, dtype=torch.long)

        return item
def prepare_data(cfg):
    """
    当前版本适配你的 Kaggle SWaT 文件结构：

      normal.csv  -> train / val
      merged.csv  -> test

    关键防泄露原则：
      1. scaler 只 fit normal train
      2. Pearson 图只用 normal train 构建
      3. validation 只来自 normal.csv
      4. merged.csv 只用于最终测试

    内存优化：
      全部使用 SWaTDynamicWindowDataset 惰性生成窗口。
      内存占用 O(T×N) 而不是 O(M×W×N)。
    """
    dcfg = cfg["data"]

    normal_df, normal_raw_labels = read_swat_csv(
        dcfg["train_csv"],
        dcfg.get("timestamp_col"),
        dcfg.get("label_col"),
    )

    merged_df, merged_raw_labels = read_swat_csv(
        dcfg["test_csv"],
        dcfg.get("timestamp_col"),
        dcfg.get("label_col"),
    )

    # 对齐 normal.csv 和 merged.csv 的公共变量列
    common_cols = [c for c in normal_df.columns if c in merged_df.columns]

    if len(common_cols) == 0:
        raise ValueError("normal.csv 和 merged.csv 没有可对齐的数值列，请检查列名。")

    normal_df = normal_df[common_cols]
    merged_df = merged_df[common_cols]

    normal_labels = build_labels(
        normal_raw_labels,
        normal_label=dcfg.get("normal_label", "Normal"),
    )

    merged_labels = build_labels(
        merged_raw_labels,
        normal_label=dcfg.get("normal_label", "Normal"),
    )

    normal_raw_values = normal_df.values.astype(np.float32)
    merged_raw_values = merged_df.values.astype(np.float32)

    # 只切分 normal.csv
    train_raw_values, val_raw_values, train_labels, val_labels = split_train_val(
        normal_raw_values,
        normal_labels,
        float(dcfg.get("val_ratio", 0.2)),
    )

    # 标准化器只 fit normal train
    scaler = StandardScaler()
    train_values = scaler.fit_transform(train_raw_values)
    val_values = scaler.transform(val_raw_values)
    test_values = scaler.transform(merged_raw_values)

    # Pearson 图只用 normal train 构建
    edge_index, corr = build_pearson_edge_index(
        train_values,
        corr_threshold=float(dcfg.get("corr_threshold", 0.3)),
        self_loop=True,
    )

    window_size = int(dcfg["window_size"])
    horizon = int(dcfg["horizon"])
    stride = int(dcfg["stride"])
    label_mode = dcfg.get("label_mode", "future")

    # 全部使用惰性 Dataset：窗口在 __getitem__ 中实时生成，避免 OOM
    train_dataset = SWaTDynamicWindowDataset(
        values=train_values,
        labels=train_labels,
        window_size=window_size,
        horizon=horizon,
        stride=stride,
        label_mode=label_mode,
    )

    val_dataset = SWaTDynamicWindowDataset(
        values=val_values,
        labels=val_labels,
        window_size=window_size,
        horizon=horizon,
        stride=stride,
        label_mode=label_mode,
    )

    test_dataset = SWaTDynamicWindowDataset(
        values=test_values,
        labels=merged_labels,
        window_size=window_size,
        horizon=horizon,
        stride=stride,
        label_mode=label_mode,
    )

    info = {
        "num_variables": len(common_cols),
        "columns": common_cols,
        "corr": corr,
        "scaler": scaler,
        "num_edges": int(edge_index.size(1)),
        "raw_lengths": {
            "normal_total": int(len(normal_raw_values)),
            "train": int(len(train_raw_values)),
            "val": int(len(val_raw_values)),
            "merged_test": int(len(merged_raw_values)),
        },
        "window_samples": {
            "train": int(len(train_dataset)),
            "val": int(len(val_dataset)),
            "test": int(len(test_dataset)),
        },
    }

    if merged_labels is not None:
        unique, counts = np.unique(merged_labels, return_counts=True)
        info["merged_label_distribution"] = {
            int(k): int(v) for k, v in zip(unique, counts)
        }

    return train_dataset, val_dataset, test_dataset, edge_index, info
