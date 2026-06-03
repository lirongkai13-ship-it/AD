import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from utils import ensure_dir, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config.yaml (读取 output.save_dir)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="直接指定输出目录，优先级高于 config")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    args = parser.parse_args()

    # 优先用 --output_dir，其次从 config 读
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        cfg = load_config(args.config)
        output_dir = cfg["output"]["save_dir"]

    score_path = os.path.join(output_dir, "test_score.npy")
    label_path = os.path.join(output_dir, "test_labels.npy")
    pred_path = os.path.join(output_dir, "test_pred.npy")

    score = np.load(score_path)
    labels = np.load(label_path) if os.path.exists(label_path) else None
    pred = np.load(pred_path) if os.path.exists(pred_path) else None

    end = None if args.end < 0 else args.end
    score = score[args.start:end]
    xs = np.arange(args.start, args.start + len(score))

    plt.figure(figsize=(16, 5))
    plt.plot(xs, score, label="anomaly score")

    if labels is not None:
        lab = labels[args.start:end]
        max_score = float(np.max(score)) if len(score) > 0 else 1.0
        plt.fill_between(
            xs,
            0,
            max_score,
            where=(lab == 1),
            alpha=0.2,
            label="attack label",
        )

    if pred is not None:
        pr = pred[args.start:end]
        max_score = float(np.max(score)) if len(score) > 0 else 1.0
        plt.scatter(
            xs[pr == 1],
            np.full((pr == 1).sum(), max_score),
            marker="x",
            label="pred alarm",
        )

    plt.xlabel("window index")
    plt.ylabel("score")
    plt.legend()
    plt.tight_layout()

    ensure_dir(output_dir)
    out_path = os.path.join(output_dir, "score_plot.png")
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()