"""Dataset discovery and tf.data input pipelines.

Expected layout (folder names are matched case-insensitively):

    <root>/train/{real,fake}/*.jpg
    <root>/val/{real,fake}/*.jpg      (also: valid, validation)
    <root>/test/{real,fake}/*.jpg

This matches the popular Kaggle datasets ("140k Real and Fake Faces",
"Deepfake and real images") as well as the output of preprocess.py.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import tensorflow as tf

import config

SPLIT_ALIASES = {
    "train": {"train", "training"},
    "val": {"val", "valid", "validation"},
    "test": {"test", "testing"},
}
CLASS_ALIASES = {
    0: {"real", "original", "genuine", "authentic"},
    1: {"fake", "deepfake", "manipulated", "synthetic"},
}


def find_dir(parent: Path, aliases: set[str]) -> Path | None:
    parent = Path(parent)
    if not parent.is_dir():
        return None
    for child in sorted(parent.iterdir()):
        if child.is_dir() and child.name.lower() in aliases:
            return child
    return None


def list_split(root, split: str, limit_per_class: int | None = None, seed: int = 42):
    """Return (paths, labels) for one split. Labels: 0 = real, 1 = fake."""
    split_dir = find_dir(Path(root), SPLIT_ALIASES[split])
    if split_dir is None:
        raise FileNotFoundError(f"No '{split}' folder found in {root}")
    rng = random.Random(seed)
    paths, labels = [], []
    for label, aliases in CLASS_ALIASES.items():
        class_dir = find_dir(split_dir, aliases)
        if class_dir is None:
            raise FileNotFoundError(f"No {config.CLASS_NAMES[label]} folder found in {split_dir}")
        files = sorted(p for p in class_dir.rglob("*") if p.suffix.lower() in config.IMAGE_EXTS)
        if limit_per_class and len(files) > limit_per_class:
            rng.shuffle(files)
            files = files[:limit_per_class]
        paths += [str(p) for p in files]
        labels += [label] * len(files)
    if not paths:
        raise FileNotFoundError(f"No images found in {split_dir}")
    return paths, labels


def class_weights(labels) -> dict[int, float]:
    """Inverse-frequency weights so an imbalanced dataset does not bias the model."""
    counts = np.bincount(np.asarray(labels), minlength=2).astype(float)
    total = counts.sum()
    return {i: float(total / (2 * c)) if c else 1.0 for i, c in enumerate(counts)}


def _loader(img_size: int):
    def load(path, label):
        raw = tf.io.read_file(path)
        img = tf.io.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(tf.cast(img, tf.float32), (img_size, img_size))
        img = tf.ensure_shape(img, (img_size, img_size, 3))
        return img, tf.cast(label, tf.float32)
    return load


def _maybe(prob, fn, img):
    return tf.cond(tf.random.uniform([]) < prob, lambda: fn(img), lambda: img)


def _gaussian_blur(img):
    sigma = tf.random.uniform([], 0.4, 1.6)
    x = tf.range(-3.0, 4.0)
    k = tf.exp(-(x ** 2) / (2 * sigma ** 2))
    k = k / tf.reduce_sum(k)
    kernel = tf.tile(tf.tensordot(k, k, axes=0)[:, :, None, None], [1, 1, 3, 1])
    return tf.nn.depthwise_conv2d(img[None], kernel, [1, 1, 1, 1], "SAME")[0]


def _downscale(img_size: int):
    def fn(img):
        small = tf.cast(img_size * tf.random.uniform([], 0.3, 0.9), tf.int32)
        img = tf.image.resize(img, (small, small), method="area")
        return tf.image.resize(img, (img_size, img_size), method="bilinear")
    return fn


def _sensor_noise(img):
    return img + tf.random.normal(tf.shape(img), stddev=tf.random.uniform([], 0.005, 0.05))


def _augmenter(img_size: int):
    """Photometric + camera-pipeline augmentation.

    Real-world uploads (phone / webcam video, social-media re-posts) are blurrier,
    noisier, lower-resolution and more compressed than the clean training photos.
    Applying these degradations to BOTH classes teaches the model that they are
    not evidence of manipulation, which prevents genuine camera footage from
    being flagged as fake.
    """
    def augment(img, label):
        img = img / 255.0
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.15)
        img = tf.image.random_contrast(img, 0.7, 1.3)
        img = tf.image.random_saturation(img, 0.7, 1.3)
        img = _maybe(0.35, _downscale(img_size), img)   # low-resolution capture
        img = _maybe(0.30, _gaussian_blur, img)          # defocus / motion blur
        img = _maybe(0.40, _sensor_noise, img)           # camera sensor noise
        img = tf.clip_by_value(img, 0.0, 1.0)
        # Video codecs and social platforms re-compress media.
        img = _maybe(0.8, lambda x: tf.image.random_jpeg_quality(x, 30, 100), img)
        img = tf.reshape(img, (img_size, img_size, 3))
        return img * 255.0, label
    return augment


def make_dataset(paths, labels, img_size: int = config.IMG_SIZE, batch_size: int = 32,
                 training: bool = False) -> tf.data.Dataset:
    """Pixel values stay in [0, 255]; each model rescales internally."""
    ds = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))
    if training:
        ds = ds.shuffle(len(paths), seed=42, reshuffle_each_iteration=True)
    ds = ds.map(_loader(img_size), num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(_augmenter(img_size), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
