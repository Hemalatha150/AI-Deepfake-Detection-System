"""Evaluate a trained model on the test split.

Writes metrics.json, classification_report.txt, confusion_matrix.png and
roc_curve.png to the reports folder.

    python evaluate.py --data-dir data/faces
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import config  # noqa: F401

import matplotlib
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             roc_curve, ConfusionMatrixDisplay)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data import list_split, make_dataset  # noqa: E402
from src.model import load_model  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=config.FACES_DIR)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--model-path", type=Path, default=config.MODEL_PATH)
    ap.add_argument("--report-dir", type=Path, default=config.REPORT_DIR)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--limit-per-class", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args(argv)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    model, meta = load_model(args.model_path)
    threshold = args.threshold if args.threshold is not None else meta["threshold"]
    paths, labels = list_split(args.data_dir, args.split, args.limit_per_class)
    y_true = np.array(labels)
    print(f"Evaluating on {len(paths)} {args.split} images (threshold={threshold:.3f}) ...")

    ds = make_dataset(paths, labels, meta["img_size"], args.batch_size)
    probs = model.predict(ds, verbose=1)[:, 0]
    y_pred = (probs >= threshold).astype(int)

    names = [config.DISPLAY_NAMES[0], config.DISPLAY_NAMES[1]]
    metrics = {
        "split": args.split,
        "samples": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probs) if len(set(labels)) > 1 else None,
    }
    report = classification_report(y_true, y_pred, target_names=names, digits=4, zero_division=0)

    (args.report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (args.report_dir / "classification_report.txt").write_text(report)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=names).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(args.report_dir / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    if metrics["roc_auc"] is not None:
        fpr, tpr, _ = roc_curve(y_true, probs)
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.4f}")
        ax.plot([0, 1], [0, 1], "--", color="grey")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.report_dir / "roc_curve.png", dpi=120)
        plt.close(fig)

    print("\n" + report)
    for k in ("accuracy", "precision", "recall", "f1_score", "roc_auc"):
        if metrics[k] is not None:
            print(f"{k:>10s}: {metrics[k]:.4f}")
    print(f"\nReports saved to {args.report_dir}")
    return metrics


if __name__ == "__main__":
    main()
