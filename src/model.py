"""CNN architectures for binary deepfake classification.

All models take raw [0, 255] RGB input of shape (img_size, img_size, 3) and
output a single sigmoid unit: P(fake). Head layers are prefixed with
"head_" so the backbone can be frozen / unfrozen by name, even after the
model has been saved and reloaded.
"""
from __future__ import annotations

import json
from pathlib import Path

import keras
from keras import layers

import config

# arch -> layer used for Grad-CAM (last convolutional activation map)
GRADCAM_LAYERS = {
    "efficientnet": "top_activation",
    "xception": "block14_sepconv2_act",
    "cnn": "conv4_act",
}


def _custom_cnn(inputs):
    """Lightweight CNN trained from scratch (fast on CPU)."""
    x = layers.Rescaling(1 / 255.0, name="input_scale")(inputs)
    for i, filters in enumerate([32, 64, 128, 256], start=1):
        for j in ("a", "b"):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"conv{i}{j}")(x)
            x = layers.BatchNormalization(name=f"conv{i}{j}_bn")(x)
            x = layers.ReLU(name=f"conv{i}_act" if j == "b" else f"conv{i}a_act")(x)
        if i < 4:
            x = layers.MaxPooling2D(name=f"pool{i}")(x)
    return x


def build_model(arch: str = config.DEFAULT_ARCH, img_size: int = config.IMG_SIZE,
                dropout: float = 0.4, weights: str | None = "imagenet") -> keras.Model:
    inputs = keras.Input((img_size, img_size, 3), name="input_image")

    if arch == "efficientnet":
        # EfficientNet rescales [0, 255] input internally.
        base = keras.applications.EfficientNetB0(include_top=False, weights=weights,
                                                 input_tensor=inputs)
        features = base.output
    elif arch == "xception":
        scaled = layers.Rescaling(1 / 127.5, offset=-1.0, name="input_scale")(inputs)
        base = keras.applications.Xception(include_top=False, weights=weights,
                                           input_tensor=scaled)
        features = base.output
    elif arch == "cnn":
        features = _custom_cnn(inputs)
    else:
        raise ValueError(f"Unknown architecture '{arch}'")

    x = layers.GlobalAveragePooling2D(name="head_pool")(features)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    x = layers.Dense(128, activation="relu", name="head_dense")(x)
    x = layers.Dropout(dropout / 2, name="head_dropout2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="head_out", dtype="float32")(x)
    return keras.Model(inputs, outputs, name=f"deepfake_{arch}")


def set_backbone_trainable(model: keras.Model, trainable: bool, fraction: float = 1.0):
    """Freeze the backbone, or unfreeze its top `fraction` of layers.

    BatchNormalization layers stay frozen during fine-tuning, which is the
    recommended practice for transfer learning with small batches.
    """
    backbone = [l for l in model.layers
                if not l.name.startswith(("head_", "input_")) and l.weights]
    cutoff = int(len(backbone) * (1 - fraction)) if trainable else len(backbone)
    for i, layer in enumerate(backbone):
        layer.trainable = (trainable and i >= cutoff
                           and not isinstance(layer, layers.BatchNormalization))


def compile_model(model: keras.Model, lr: float):
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )


# ----------------------------------------------------------- persistence
def meta_path(model_path) -> Path:
    return Path(model_path).with_suffix(".json")


def save_metadata(model_path, **meta):
    meta_path(model_path).write_text(json.dumps(meta, indent=2))


def load_model(model_path=config.MODEL_PATH):
    """Load a trained model together with its metadata."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Train one first with: python train.py")
    model = keras.models.load_model(model_path, compile=False)
    meta = {"arch": "efficientnet", "img_size": model.input_shape[1],
            "threshold": config.DEFAULT_THRESHOLD}
    if meta_path(model_path).exists():
        meta.update(json.loads(meta_path(model_path).read_text()))
    meta.setdefault("gradcam_layer", GRADCAM_LAYERS.get(meta["arch"]))
    return model, meta
