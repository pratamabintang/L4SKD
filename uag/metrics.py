"""Dataset-level pixel confusion metrics compatible with tf.keras 2 and Keras 3.

Note: These metrics accumulate TP, FP, and FN across all batches in an epoch
(dataset-level micro metrics), which differs from `segmentation_models` metrics
that average per-batch scores.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras


class _ConfusionMetric(keras.metrics.Metric):
    """Base metric accumulating global binary TP, FP, and FN counts."""

    def __init__(
        self,
        threshold: float = 0.5,
        smooth: float = 1e-7,
        name: str = "confusion_metric",
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.threshold = float(threshold)
        self.smooth = float(smooth)
        self.tp = self.add_weight(name="tp", shape=(), initializer="zeros")
        self.fp = self.add_weight(name="fp", shape=(), initializer="zeros")
        self.fn = self.add_weight(name="fn", shape=(), initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None) -> None:
        y_true_b = tf.cast(y_true >= 0.5, tf.float32)
        y_pred_b = tf.cast(y_pred >= self.threshold, tf.float32)

        if sample_weight is not None:
            weight = tf.cast(sample_weight, tf.float32)
            y_true_b = y_true_b * weight
            y_pred_b = y_pred_b * weight

        tp = tf.reduce_sum(y_true_b * y_pred_b)
        fp = tf.reduce_sum((1.0 - y_true_b) * y_pred_b)
        fn = tf.reduce_sum(y_true_b * (1.0 - y_pred_b))

        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def reset_state(self) -> None:
        self.tp.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"threshold": self.threshold, "smooth": self.smooth})
        return config


@keras.utils.register_keras_serializable(package="uag")
class BinaryPrecision(_ConfusionMetric):
    """Dataset-level pixel precision at a probability threshold."""

    def __init__(
        self,
        threshold: float = 0.5,
        smooth: float = 1e-7,
        name: str = "precision",
        **kwargs,
    ) -> None:
        super().__init__(threshold=threshold, smooth=smooth, name=name, **kwargs)

    def result(self):
        return (self.tp + self.smooth) / (self.tp + self.fp + self.smooth)


@keras.utils.register_keras_serializable(package="uag")
class BinaryRecall(_ConfusionMetric):
    """Dataset-level pixel recall at a probability threshold."""

    def __init__(
        self,
        threshold: float = 0.5,
        smooth: float = 1e-7,
        name: str = "recall",
        **kwargs,
    ) -> None:
        super().__init__(threshold=threshold, smooth=smooth, name=name, **kwargs)

    def result(self):
        return (self.tp + self.smooth) / (self.tp + self.fn + self.smooth)


@keras.utils.register_keras_serializable(package="uag")
class BinaryF1Score(_ConfusionMetric):
    """Dataset-level pixel F1 score at a probability threshold."""

    def __init__(
        self,
        threshold: float = 0.5,
        smooth: float = 1e-7,
        name: str = "f1_score",
        **kwargs,
    ) -> None:
        super().__init__(threshold=threshold, smooth=smooth, name=name, **kwargs)

    def result(self):
        return (2.0 * self.tp + self.smooth) / (
            2.0 * self.tp + self.fp + self.fn + self.smooth
        )


@keras.utils.register_keras_serializable(package="uag")
class BinaryIoU(_ConfusionMetric):
    """Dataset-level pixel IoU (Jaccard) score at a probability threshold."""

    def __init__(
        self,
        threshold: float = 0.5,
        smooth: float = 1e-7,
        name: str = "iou",
        **kwargs,
    ) -> None:
        super().__init__(threshold=threshold, smooth=smooth, name=name, **kwargs)

    def result(self):
        return (self.tp + self.smooth) / (self.tp + self.fp + self.fn + self.smooth)


def get_metrics(threshold: float = 0.5) -> list[keras.metrics.Metric]:
    """Return [precision, recall, f1_score, iou] metric instances."""
    return [
        BinaryPrecision(threshold=threshold, name="precision"),
        BinaryRecall(threshold=threshold, name="recall"),
        BinaryF1Score(threshold=threshold, name="f1_score"),
        BinaryIoU(threshold=threshold, name="iou"),
    ]
