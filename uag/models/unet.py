"""Standard U-Net model definition without attention gates."""

from __future__ import annotations

from typing import Optional

from tensorflow import keras
from tensorflow.keras import layers

from uag.models.attention_unet import KINIT, unet_conv2d


def build_unet(
    input_shape: tuple[int, int, int] = (128, 128, 14),
    filters: int = 4,
    pretrained_weights: Optional[str] = None,
    name: str = "unet",
) -> keras.Model:
    """Build an uncompiled standard U-Net without attention gates.

    Mirrors `build_attention_unet` identically in encoder depth, filter counts,
    batch normalization, and decoder transposed-convolution widths, omitting only
    the attention gates and gating signals so the two form a fair ablation pair.
    """
    height, width, _ = input_shape
    if height % 16 != 0 or width % 16 != 0:
        raise ValueError(
            f"Input height and width must be divisible by 16, got {input_shape[:2]}."
        )

    inputs = keras.Input(shape=input_shape)

    conv1 = unet_conv2d(inputs, filters, is_batchnorm=True, name="conv1")
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = unet_conv2d(pool1, filters, is_batchnorm=True, name="conv2")
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = unet_conv2d(pool2, filters * 2, is_batchnorm=True, name="conv3")
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = unet_conv2d(pool3, filters * 2, is_batchnorm=True, name="conv4")
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    center = unet_conv2d(pool4, filters * 4, is_batchnorm=True, name="center")

    up1 = layers.concatenate(
        [
            layers.Conv2DTranspose(
                filters,
                (3, 3),
                strides=(2, 2),
                padding="same",
                activation="relu",
                kernel_initializer=KINIT,
            )(center),
            conv4,
        ],
        name="up1",
    )

    up2 = layers.concatenate(
        [
            layers.Conv2DTranspose(
                filters * 2,
                (3, 3),
                strides=(2, 2),
                padding="same",
                activation="relu",
                kernel_initializer=KINIT,
            )(up1),
            conv3,
        ],
        name="up2",
    )

    up3 = layers.concatenate(
        [
            layers.Conv2DTranspose(
                filters,
                (3, 3),
                strides=(2, 2),
                padding="same",
                activation="relu",
                kernel_initializer=KINIT,
            )(up2),
            conv2,
        ],
        name="up3",
    )

    up4 = layers.concatenate(
        [
            layers.Conv2DTranspose(
                filters,
                (3, 3),
                strides=(2, 2),
                padding="same",
                activation="relu",
                kernel_initializer=KINIT,
            )(up3),
            conv1,
        ],
        name="up4",
    )

    # Keep final activation in float32 for numerical stability under mixed precision.
    outputs = layers.Conv2D(
        1,
        (1, 1),
        activation="sigmoid",
        kernel_initializer=KINIT,
        dtype="float32",
        name="final",
    )(up4)

    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    if pretrained_weights is not None:
        model.load_weights(pretrained_weights)
    return model
