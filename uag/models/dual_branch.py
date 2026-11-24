"""Dual-branch Attention U-Net separating multispectral and topographic encoders."""

from __future__ import annotations

from typing import Optional, Sequence

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from uag.data.landslide4sense import resolve_dual_branch_indices
from uag.models.attention_unet import (
    KINIT,
    attn_gating_block,
    unet_conv2d,
    unet_gating_signal,
)


@keras.utils.register_keras_serializable(package="uag")
class ChannelSelect(layers.Layer):
    """Select an ordered subset of channels from an input tensor."""

    def __init__(self, indices: Sequence[int], **kwargs) -> None:
        super().__init__(**kwargs)
        self.indices = [int(i) for i in indices]
        if not self.indices:
            raise ValueError("ChannelSelect indices must be non-empty.")

    def call(self, inputs):
        return tf.gather(inputs, self.indices, axis=-1)

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], len(self.indices))

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"indices": self.indices})
        return config


def single_encoder(
    inputs,
    filters: int,
    name: str = "encoder",
):
    """Standard U-Net encoder returning 4 skip maps and the bottleneck."""
    conv1 = unet_conv2d(inputs, filters, True, name=f"{name}_conv1")
    pool1 = layers.MaxPooling2D((2, 2), name=f"{name}_pool1")(conv1)

    conv2 = unet_conv2d(pool1, filters, True, name=f"{name}_conv2")
    pool2 = layers.MaxPooling2D((2, 2), name=f"{name}_pool2")(conv2)

    conv3 = unet_conv2d(pool2, filters * 2, True, name=f"{name}_conv3")
    pool3 = layers.MaxPooling2D((2, 2), name=f"{name}_pool3")(conv3)

    conv4 = unet_conv2d(pool3, filters * 2, True, name=f"{name}_conv4")
    pool4 = layers.MaxPooling2D((2, 2), name=f"{name}_pool4")(conv4)

    center = unet_conv2d(pool4, filters * 4, True, name=f"{name}_center")
    return conv1, conv2, conv3, conv4, center


def dual_encoder(
    inputs1,
    inputs2,
    filters: int,
):
    """Encode two modality branches with separate parameter sets."""
    features1 = single_encoder(inputs1, filters, name="encoder1")
    features2 = single_encoder(inputs2, filters, name="encoder2")
    return features1, features2


def feature_fusion(
    x1,
    x2,
    outdim: int,
    name: str,
):
    """Fuse two feature maps via Concatenate -> Conv2D(3x3) -> BN -> ReLU."""
    x = layers.Concatenate(axis=-1, name=f"{name}_concat")([x1, x2])
    x = layers.Conv2D(
        outdim,
        (3, 3),
        padding="same",
        kernel_initializer=KINIT,
        name=f"{name}_conv",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.Activation("relu", name=f"{name}_relu")(x)
    return x


def fuse_encoder_features(
    features1,
    features2,
    filters: int,
):
    """Fuse encoder features at all five resolution levels."""
    conv1 = feature_fusion(features1[0], features2[0], filters, name="fusion1")
    conv2 = feature_fusion(features1[1], features2[1], filters, name="fusion2")
    conv3 = feature_fusion(features1[2], features2[2], filters * 2, name="fusion3")
    conv4 = feature_fusion(features1[3], features2[3], filters * 2, name="fusion4")
    center = feature_fusion(features1[4], features2[4], filters * 4, name="fusion_center")
    return conv1, conv2, conv3, conv4, center


def attention_unet_decoder(
    conv1,
    conv2,
    conv3,
    conv4,
    center,
    filters: int,
):
    """Attention U-Net decoder shared by single and dual encoder models."""
    g1 = unet_gating_signal(center, is_batchnorm=True, name="g1")
    attn1 = attn_gating_block(conv4, g1, filters * 4, "_1")
    up1 = layers.concatenate(
        [
            layers.Conv2DTranspose(
                filters,
                (3, 3),
                strides=(2, 2),
                padding="same",
                activation="relu",
                kernel_initializer=KINIT,
                name="up1_transpose",
            )(center),
            attn1,
        ],
        name="up1",
    )

    g2 = unet_gating_signal(up1, is_batchnorm=True, name="g2")
    attn2 = attn_gating_block(conv3, g2, filters * 2, "_2")
    up2 = layers.concatenate(
        [
            layers.Conv2DTranspose(
                filters * 2,
                (3, 3),
                strides=(2, 2),
                padding="same",
                activation="relu",
                kernel_initializer=KINIT,
                name="up2_transpose",
            )(up1),
            attn2,
        ],
        name="up2",
    )

    # Preserve original reference implementation: g3 uses up1.
    g3 = unet_gating_signal(up1, is_batchnorm=True, name="g3")
    attn3 = attn_gating_block(conv2, g3, filters, "_3")
    up3 = layers.concatenate(
        [
            layers.Conv2DTranspose(
                filters,
                (3, 3),
                strides=(2, 2),
                padding="same",
                activation="relu",
                kernel_initializer=KINIT,
                name="up3_transpose",
            )(up2),
            attn3,
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
                name="up4_transpose",
            )(up3),
            conv1,
        ],
        name="up4",
    )
    return up4


def build_dual_branch_attention_unet(
    input_shape: tuple[int, int, int] = (128, 128, 14),
    filters: int = 4,
    pretrained_weights: Optional[str] = None,
    bands: Optional[Sequence[int]] = None,
    branch1_indices: Optional[Sequence[int]] = None,
    branch2_indices: Optional[Sequence[int]] = None,
    name: str = "dual_branch_attention_unet",
) -> keras.Model:
    """Build a dual-branch (or single-branch fallback) Attention U-Net.

    When both multispectral (B1-B12) and topographic (Slope, DEM) channels are
    present (or explicit branch indices are provided), encodes them with two
    separate encoders and fuses features across all 5 levels before decoding.
    Falls back to a single encoder when only one modality is present.
    """
    height, width, channels = input_shape

    if height % 16 != 0 or width % 16 != 0:
        raise ValueError(
            "Input height and width must be divisible by 16. "
            f"Received: {input_shape[:2]}"
        )
    if filters < 1:
        raise ValueError("filters must be >= 1.")

    b1_idx, b2_idx = resolve_dual_branch_indices(
        channels=channels,
        bands=bands,
        branch1_indices=branch1_indices,
        branch2_indices=branch2_indices,
    )

    inputs = keras.Input(shape=input_shape, name="image_input")

    if b2_idx is None:
        conv1, conv2, conv3, conv4, center = single_encoder(
            inputs,
            filters,
            name="encoder1",
        )
        model_name = f"{name}_single"
    else:
        branch1_input = ChannelSelect(b1_idx, name="branch1_select")(inputs)
        branch2_input = ChannelSelect(b2_idx, name="branch2_select")(inputs)
        features1, features2 = dual_encoder(branch1_input, branch2_input, filters)
        conv1, conv2, conv3, conv4, center = fuse_encoder_features(
            features1,
            features2,
            filters,
        )
        model_name = f"{name}_dual"

    decoder = attention_unet_decoder(
        conv1,
        conv2,
        conv3,
        conv4,
        center,
        filters,
    )

    # Keep final activation in float32 for numerical stability under mixed precision.
    outputs = layers.Conv2D(
        1,
        (1, 1),
        activation="sigmoid",
        kernel_initializer=KINIT,
        dtype="float32",
        name="final",
    )(decoder)

    model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)
    if pretrained_weights is not None:
        model.load_weights(pretrained_weights)
    return model
