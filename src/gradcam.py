"""Grad-CAM: highlights the face regions that drive the 'fake' score."""
from __future__ import annotations

import cv2
import keras
import numpy as np
import tensorflow as tf


class GradCAM:
    def __init__(self, model: keras.Model, layer_name: str):
        self.grad_model = keras.Model(model.inputs,
                                      [model.get_layer(layer_name).output, model.outputs[0]])

    def heatmap(self, image: np.ndarray) -> np.ndarray:
        """image: (H, W, 3) float in [0, 255] at model input size. Returns (h, w) in [0, 1]."""
        batch = tf.convert_to_tensor(image[None].astype("float32"))
        with tf.GradientTape() as tape:
            conv, preds = self.grad_model(batch, training=False)
            score = preds[:, 0]
        grads = tape.gradient(score, conv)
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam = tf.nn.relu(tf.reduce_sum(conv[0] * weights, axis=-1)).numpy()
        return cam / cam.max() if cam.max() > 0 else cam


def overlay(rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat = cv2.resize(heatmap, (rgb.shape[1], rgb.shape[0]))
    heat = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb.astype(np.uint8), 1 - alpha, heat, alpha, 0)
