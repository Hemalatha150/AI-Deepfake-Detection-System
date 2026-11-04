"""Train the deepfake detector.

Transfer-learning models (efficientnet, xception) are trained in two phases:
  1. backbone frozen, only the classification head learns;
  2. the top part of the backbone is unfrozen and fine-tuned at a lower LR.
The custom 'cnn' architecture is trained end-to-end from scratch.

Examples:
    python train.py --data-dir data/faces
    python train.py --data-dir data/faces --arch cnn --img-size 128 --epochs 15
    python train.py --limit-per-class 5000        # quicker run on a CPU
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import config  # noqa: F401  (sets TF log level before TensorFlow is imported)

import keras
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data import class_weights, list_split, make_dataset  # noqa: E402
from src.model import (GRADCAM_LAYERS, build_model, compile_model,  # noqa: E402
                       save_metadata, set_backbone_trainable)


def callbacks(model_path: Path, report_dir: Path, best: float | None):
    return [
        keras.callbacks.ModelCheckpoint(str(model_path), monitor="val_auc", mode="max",
                                        save_best_only=True, initial_value_threshold=best,
                                        verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=4,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2,
                                          min_lr=1e-7, verbose=1),
        keras.callbacks.CSVLogger(str(report_dir / "training_log.csv"), append=True),
    ]


def best_threshold(labels, probs) -> float:
    """Threshold that maximises Youden's J (TPR - FPR) on validation data."""
    if len(set(labels)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(labels, probs)
    t = float(thr[np.argmax(tpr - fpr)])
    return float(np.clip(t, 0.05, 0.95))


def plot_history(history: dict, path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric in zip(axes, ["loss", "accuracy", "auc"]):
        ax.plot(history[metric], label="train")
        ax.plot(history[f"val_{metric}"], label="validation")
        ax.set_title(metric.upper())
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=config.FACES_DIR)
    ap.add_argument("--arch", default=config.DEFAULT_ARCH, choices=list(GRADCAM_LAYERS))
    ap.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=5,
                    help="head epochs (transfer models) or total epochs (cnn)")
    ap.add_argument("--fine-tune-epochs", type=int, default=10)
    ap.add_argument("--fine-tune-fraction", type=float, default=0.3,
                    help="fraction of backbone layers unfrozen in phase 2")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit-per-class", type=int, default=None,
                    help="cap training images per class (useful on CPU)")
    ap.add_argument("--val-limit-per-class", type=int, default=None,
                    help="cap validation images per class (default: limit-per-class / 4)")
    ap.add_argument("--model-path", type=Path, default=config.MODEL_PATH)
    ap.add_argument("--report-dir", type=Path, default=config.REPORT_DIR)
    ap.add_argument("--no-pretrained", action="store_true", help="don't load ImageNet weights")
    ap.add_argument("--init-model", type=Path, default=None,
                    help="continue fine-tuning from an existing .keras model "
                         "(runs --epochs fine-tuning epochs only)")
    ap.add_argument("--calibrate-only", action="store_true",
                    help="skip training; calibrate the threshold and write metadata/plots for the "
                         "existing checkpoint (e.g. after an interrupted run)")
    args = ap.parse_args(argv)

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    log_csv = args.report_dir / "training_log.csv"
    if not args.calibrate_only:
        log_csv.unlink(missing_ok=True)

    train_paths, train_labels = list_split(args.data_dir, "train", args.limit_per_class)
    val_limit = args.val_limit_per_class or (args.limit_per_class // 4 if args.limit_per_class else None)
    val_paths, val_labels = list_split(args.data_dir, "val", val_limit)
    print(f"Train: {len(train_paths)} images (fake={sum(train_labels)}), "
          f"Val: {len(val_paths)} images (fake={sum(val_labels)})")

    val_ds = make_dataset(val_paths, val_labels, args.img_size, args.batch_size)
    if args.calibrate_only:
        history = pd.read_csv(log_csv).drop(columns="epoch").to_dict("list")
    else:
        history = train(args, train_paths, train_labels, val_ds)
    finalize(args, history, val_ds, val_labels, len(train_paths), len(val_paths))


def train(args, train_paths, train_labels, val_ds) -> dict[str, list]:
    train_ds = make_dataset(train_paths, train_labels, args.img_size, args.batch_size, training=True)
    weights = class_weights(train_labels)

    if args.init_model:
        model = keras.models.load_model(args.init_model, compile=False)
    else:
        model = build_model(args.arch, args.img_size,
                            weights=None if args.no_pretrained else "imagenet")
    history: dict[str, list] = {}

    def run(epochs, initial_epoch=0, best=None):
        h = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                      initial_epoch=initial_epoch, class_weight=weights,
                      callbacks=callbacks(args.model_path, args.report_dir, best))
        for k, v in h.history.items():
            history.setdefault(k, []).extend(v)
        return max(h.history["val_auc"]), len(h.history["loss"])

    if args.init_model:
        print(f"\n=== Fine-tuning {args.init_model.name} (top {args.fine_tune_fraction:.0%} of backbone) ===")
        if args.arch != "cnn":
            set_backbone_trainable(model, True, args.fine_tune_fraction)
        compile_model(model, args.lr)
        run(args.epochs)
    elif args.arch == "cnn":
        compile_model(model, args.lr)
        run(args.epochs)
    else:
        print("\n=== Phase 1: training classification head (backbone frozen) ===")
        set_backbone_trainable(model, False)
        compile_model(model, args.lr)
        best, done = run(args.epochs)
        if args.fine_tune_epochs > 0:
            print(f"\n=== Phase 2: fine-tuning top {args.fine_tune_fraction:.0%} of backbone ===")
            set_backbone_trainable(model, True, args.fine_tune_fraction)
            compile_model(model, args.lr / 10)
            run(done + args.fine_tune_epochs, initial_epoch=done, best=best)
    return history


def finalize(args, history, val_ds, val_labels, n_train, n_val):
    # Reload the best checkpoint and calibrate the decision threshold on validation data.
    model = keras.models.load_model(args.model_path, compile=False)
    val_probs = model.predict(val_ds, verbose=0)[:, 0]
    threshold = best_threshold(val_labels, val_probs)
    val_acc = float(((val_probs >= threshold) == np.array(val_labels)).mean())
    print(f"\nCalibrated threshold: {threshold:.3f}  (val accuracy at threshold: {val_acc:.4f})")

    save_metadata(
        args.model_path,
        arch=args.arch,
        img_size=args.img_size,
        threshold=threshold,
        gradcam_layer=GRADCAM_LAYERS[args.arch],
        class_names=config.CLASS_NAMES,
        trained_at=datetime.now().isoformat(timespec="seconds"),
        data_dir=str(args.data_dir),
        train_images=n_train,
        val_images=n_val,
        best_val_auc=float(max(history["val_auc"])),
        val_accuracy=val_acc,
    )
    plot_history(history, args.report_dir / "training_curves.png")
    (args.report_dir / "training_history.json").write_text(
        json.dumps({k: [float(x) for x in v] for k, v in history.items()}, indent=2))
    print(f"Model saved to {args.model_path}")
    print(f"Training curves saved to {args.report_dir / 'training_curves.png'}")


if __name__ == "__main__":
    main()
