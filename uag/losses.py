"""Segmentation loss functions implemented with pure TensorFlow ops."""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras


def dice_coefficient(y_true, y_pred, smooth: float = 1.0):
    """Global soft Dice coefficient over all pixels in the batch."""
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def dice_loss(y_true, y_pred, smooth: float = 1.0):
    """Soft Dice loss (1 - dice_coefficient)."""
    return 1.0 - dice_coefficient(y_true, y_pred, smooth=smooth)


def bce_dice_loss(y_true, y_pred):
    """Combined binary cross-entropy and soft Dice loss."""
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, y_pred.dtype)
    bce = keras.losses.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean(bce) + dice_loss(y_true, y_pred)


def tversky_index(y_true, y_pred, alpha: float = 0.7, smooth: float = 1.0):
    """Global Tversky index (default alpha=0.7).

    `alpha` weights false negatives and `(1 - alpha)` weights false positives;
    with `alpha > 0.5`, false negatives are penalized more heavily, favoring recall
    on the rare landslide class.
    """
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    true_pos = tf.reduce_sum(y_true_f * y_pred_f)
    false_neg = tf.reduce_sum(y_true_f * (1.0 - y_pred_f))
    false_pos = tf.reduce_sum((1.0 - y_true_f) * y_pred_f)
    return (true_pos + smooth) / (
        true_pos + alpha * false_neg + (1.0 - alpha) * false_pos + smooth
    )


def tversky_loss(y_true, y_pred, alpha: float = 0.7, smooth: float = 1.0):
    """Tversky loss `1 - tversky_index` (default alpha=0.7)."""
    return 1.0 - tversky_index(y_true, y_pred, alpha=alpha, smooth=smooth)


def focal_tversky_loss(
    y_true,
    y_pred,
    alpha: float = 0.7,
    gamma: float = 0.75,
    smooth: float = 1.0,
):
    """Focal Tversky loss `(1 - tversky_index)^gamma` (defaults alpha=0.7, gamma=0.75)."""
    # Clamp base to >= 0 to prevent NaN gradients from tiny negative float rounding.
    base = tf.maximum(1.0 - tversky_index(y_true, y_pred, alpha=alpha, smooth=smooth), 0.0)
    return tf.pow(base, gamma)


def structure_loss(y_true, y_pred, kernel_size: int = 31, edge_weight: float = 5.0):
    """PraNet-style structure loss (weighted BCE + weighted IoU) for NHWC probabilities.

    Port notes relative to the PyTorch reference:
    - Models end with a sigmoid activation, so `y_pred` contains probabilities rather
      than logits; BCE is computed directly from clipped probabilities.
    - Inputs are NHWC, so spatial reductions sum over axes `[1, 2]`.
    - Explicit zero-padding + VALID `avg_pool2d` matches PyTorch `count_include_pad=True`.
    - Uses per-pixel weighted BCE (`reduction='none'` semantics) rather than PyTorch's
      deprecated `reduce='none'` truthiness quirk that collapsed BCE before weighting.
    """
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, tf.float32)

    pad = kernel_size // 2
    padded = tf.pad(y_true, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
    local_mean = tf.nn.avg_pool2d(padded, ksize=kernel_size, strides=1, padding="VALID")
    weit = 1.0 + edge_weight * tf.abs(local_mean - y_true)

    eps = 1e-7
    p = tf.clip_by_value(y_pred, eps, 1.0 - eps)
    bce = -(y_true * tf.math.log(p) + (1.0 - y_true) * tf.math.log(1.0 - p))
    wbce = tf.reduce_sum(weit * bce, axis=[1, 2]) / tf.reduce_sum(weit, axis=[1, 2])

    inter = tf.reduce_sum(p * y_true * weit, axis=[1, 2])
    union = tf.reduce_sum((p + y_true) * weit, axis=[1, 2])
    wiou = 1.0 - (inter + 1.0) / (union - inter + 1.0)
    return tf.reduce_mean(wbce + wiou)


def _pearson(a, b, eps: float = 1e-8):
    a_centered = a - tf.reduce_mean(a, axis=1, keepdims=True)
    b_centered = b - tf.reduce_mean(b, axis=1, keepdims=True)
    num = tf.reduce_sum(a_centered * b_centered, axis=1)
    norm_a = tf.sqrt(tf.reduce_sum(tf.square(a_centered), axis=1) + 1e-12)
    norm_b = tf.sqrt(tf.reduce_sum(tf.square(b_centered), axis=1) + 1e-12)
    return num / (norm_a * norm_b + eps)


def dist_loss_binary(
    student_prob,
    teacher_logit,
    tau: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    eps: float = 1e-6,
):
    """Output-level DIST knowledge distillation loss for binary segmentation."""
    student_prob = tf.cast(student_prob, tf.float32)
    teacher_logit = tf.cast(teacher_logit, tf.float32)

    p = tf.clip_by_value(student_prob, eps, 1.0 - eps)
    z_s = tf.math.log(p) - tf.math.log1p(-p)
    zeros = tf.zeros_like(z_s)

    tau_f = float(tau)
    logits_s = tf.reshape(tf.concat([zeros, z_s], axis=-1) / tau_f, [-1, 2])
    logits_t = tf.reshape(tf.concat([zeros, teacher_logit], axis=-1) / tau_f, [-1, 2])

    y_s = tf.nn.softmax(logits_s, axis=-1)
    y_t = tf.nn.softmax(logits_t, axis=-1)

    inter = 1.0 - tf.reduce_mean(_pearson(y_s, y_t))
    intra = 1.0 - tf.reduce_mean(_pearson(tf.transpose(y_s), tf.transpose(y_t)))
    return (tau_f ** 2) * (float(beta) * inter + float(gamma) * intra)


def make_kd_loss(
    task_loss_name: str,
    kd_weight: float = 1.0,
    tau: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
):
    """Build combined task + DIST KD loss and monitoring metric functions."""
    base_loss_fn = get_loss(task_loss_name)
    kd_w = float(kd_weight)

    def kd_total_loss(y_true, y_pred):
        mask = y_true[..., :1]
        teacher_logit = y_true[..., 1:2]
        t_loss = tf.reduce_mean(base_loss_fn(mask, y_pred))
        d_loss = dist_loss_binary(y_pred, teacher_logit, tau=tau, beta=beta, gamma=gamma)
        return t_loss + kd_w * d_loss

    def task_loss(y_true, y_pred):
        mask = y_true[..., :1]
        return tf.reduce_mean(base_loss_fn(mask, y_pred))

    def kd_loss(y_true, y_pred):
        teacher_logit = y_true[..., 1:2]
        return dist_loss_binary(y_pred, teacher_logit, tau=tau, beta=beta, gamma=gamma)

    task_loss.__name__ = "task_loss"
    kd_loss.__name__ = "kd_loss"
    return kd_total_loss, [task_loss, kd_loss]


def get_loss(name: str):
    """Resolve a loss function by name."""
    losses = {
        "dice": dice_loss,
        "bce_dice": bce_dice_loss,
        "bce": keras.losses.BinaryCrossentropy(),
        "structure": structure_loss,
        "tversky": tversky_loss,
        "focal_tversky": focal_tversky_loss,
    }
    key = name.lower()
    if key not in losses:
        raise ValueError(f"Unknown loss {name!r}. Choose from {sorted(losses)}.")
    return losses[key]
