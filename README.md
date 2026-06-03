# SWaT GATv2 + TCN-GRU 双分支异常检测 Baseline

这是一个最小可运行版本，路线为：

标准化 + 滑动窗口 → Pearson 图 → GATv2 → TCN-GRU → 预测头 + 重构头 → IQR/Top-k 异常评分

## 1. 创建环境

```bash
conda create -n swat_ad python=3.10 -y
conda activate swat_ad
pip install -r requirements.txt
```

## 2. 放数据

在项目根目录创建 data 文件夹，把 SWaT 的 normal 和 attack CSV 放进去：

```bash
mkdir data
```

然后修改 `config.yaml` 里的路径和列名。

## 3. 训练和测试

```bash
python train.py --config config.yaml
python evaluate.py --config config.yaml --ckpt outputs/best_model.pt
```

## 4. 注意

如果你的 SWaT 标签列不是 `Normal/Attack`，请在 config.yaml 里修改。
如果你的时间列不是 `Timestamp`，也要修改。
