"""Attention U-Net model definition compatible with tf.keras 2 and Keras 3."""

from __future__ import annotations

from typing import Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from uag.pruning.hrank import default_attention_unet_channel_cfg

KINIT = "glorot_normal"


@keras.utils.register_keras_serializable(package="uag")
class RepeatChannels(layers.Layer):
    """Repeat single-channel attention map across feature channels."""

    def __init__(self, rep: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.rep = int(rep)

    def call(self, inputs):
        return tf.repeat(inputs, self.rep, axis=-1)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"rep": self.rep})
        return config


def unet_conv2d(x, outdim: int, is_batchnorm: bool, name: str, mid_dim: Optional[int] = None):
    """Two-layer 3x3 Conv2D block with optional BatchNorm and ReLU."""
    first_dim = int(mid_dim) if mid_dim is not None else int(outdim)
    x = layers.Conv2D(
        first_dim,
        (3, 3),
        strides=(1, 1),
        kernel_initializer=KINIT,
        padding="same",
        name=f"{name}_1",
    )(x)
    if is_batchnorm:
        x = layers.BatchNormalization(name=f"{name}_1_bn")(x)
    x = layers.Activation("relu", name=f"{name}_1_act")(x)

    x = layers.Conv2D(
        outdim,
        (3, 3),
        strides=(1, 1),
        kernel_initializer=KINIT,
        padding="same",
        name=f"{name}_2",
    )(x)
    if is_batchnorm:
        x = layers.BatchNormalization(name=f"{name}_2_bn")(x)
    x = layers.Activation("relu", name=f"{name}_2_act")(x)
    return x


def unet_gating_signal(x, is_batchnorm: bool, name: str):
    """1x1 convolution gating signal block with optional BatchNorm and ReLU."""
    channels = int(x.shape[-1])
    x = layers.Conv2D(
        channels,
        (1, 1),
        strides=(1, 1),
        padding="same",
        kernel_initializer=KINIT,
        name=f"{name}_conv",
    )(x)
    if is_batchnorm:
        x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.Activation("relu", name=f"{name}_act")(x)
    return x


def attn_gating_block(x, g, inter_shape: int, name: str):
    """Additive attention gate conditioning skip tensor `x` on gating tensor `g`."""
    shape_x = x.shape
    shape_g = g.shape

    theta_x = layers.Conv2D(
        inter_shape,
        (2, 2),
        strides=(2, 2),
        padding="same",
        name=f"xl{name}",
    )(x)
    shape_theta_x = theta_x.shape

    phi_g = layers.Conv2D(inter_shape, (1, 1), padding="same")(g)
    upsample_g = layers.Conv2DTranspose(
        inter_shape,
        (3, 3),
        strides=(
            int(shape_theta_x[1]) // int(shape_g[1]),
            int(shape_theta_x[2]) // int(shape_g[2]),
        ),
        padding="same",
        name=f"g_up{name}",
    )(phi_g)

    concat_xg = layers.add([upsample_g, theta_x])
    act_xg = layers.Activation("relu")(concat_xg)
    psi = layers.Conv2D(1, (1, 1), padding="same", name=f"psi{name}")(act_xg)
    sigmoid_xg = layers.Activation("sigmoid")(psi)
    shape_sigmoid = sigmoid_xg.shape

    upsample_psi = layers.UpSampling2D(
        size=(
            int(shape_x[1]) // int(shape_sigmoid[1]),
            int(shape_x[2]) // int(shape_sigmoid[2]),
        )
    )(sigmoid_xg)
    upsample_psi = RepeatChannels(int(shape_x[3]), name=f"psi_up{name}")(upsample_psi)

    y = layers.multiply([upsample_psi, x], name=f"q_attn{name}")
    result = layers.Conv2D(
        int(shape_x[3]),
        (1, 1),
        padding="same",
        name=f"q_attn_conv{name}",
    )(y)
    result_bn = layers.BatchNormalization(name=f"q_attn_bn{name}")(result)
    return result_bn


def build_attention_unet(
    input_shape: tuple[int, int, int] = (128, 128, 14),
    filters: int = 4,
    pretrained_weights: Optional[str] = None,
    name: str = "attention_unet",
    channel_cfg: Optional[dict[str, int]] = None,
) -> keras.Model:
    """Build an uncompiled Attention U-Net model matching the original notebook architecture."""
    height, width, _ = input_shape
    if height % 16 != 0 or width % 16 != 0:
        raise ValueError(
            f"Input height and width must be divisible by 16, got {input_shape[:2]}."
        )

    ch = {**default_attention_unet_channel_cfg(filters), **(channel_cfg or {})}
    inputs = keras.Input(shape=input_shape)

    conv1 = unet_conv2d(inputs, ch["conv1_2"], is_batchnorm=True, name="conv1", mid_dim=ch["conv1_1"])
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = unet_conv2d(pool1, ch["conv2_2"], is_batchnorm=True, name="conv2", mid_dim=ch["conv2_1"])
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = unet_conv2d(pool2, ch["conv3_2"], is_batchnorm=True, name="conv3", mid_dim=ch["conv3_1"])
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = unet_conv2d(pool3, ch["conv4_2"], is_batchnorm=True, name="conv4", mid_dim=ch["conv4_1"])
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    center = unet_conv2d(pool4, ch["center_2"], is_batchnorm=True, name="center", mid_dim=ch["center_1"])

    g1 = unet_gating_signal(center, is_batchnorm=True, name="g1")
    attn1 = attn_gating_block(conv4, g1, ch["inter_1"], "_1")
    up1 = layers.concatenate(
        [
            layers.Conv2DTranspose(
                ch["up1_deconv"],
                (3, 3),
                strides=(2, 2),
                padding="same",
                activation="relu",
                kernel_initializer=KINIT,
            )(center),
            attn1,
        ],
        name="up1",
    )

    g2 = unet_gating_signal(up1, is_batchnorm=True, name="g2")
    attn2 = attn_gating_block(conv3, g2, ch["inter_2"], "_2")
    up2 = layers.concatenate(
        [
            layers.Conv2DTranspose(
                ch["up2_deconv"],
                (3, 3),
                strides=(2, 2),
                padding="same",
                activation="relu",
                kernel_initializer=KINIT,
            )(up1),
            attn2,
        ],
        name="up2",
    )

    # g3 uses up1 as in the original reference notebook code.
    g3 = unet_gating_signal(up1, is_batchnorm=True, name="g3")
    attn3 = attn_gating_block(conv2, g3, ch["inter_3"], "_3")
    up3 = layers.concatenate(
        [
            layers.Conv2DTranspose(
                ch["up3_deconv"],
                (3, 3),
                strides=(2, 2),
                padding="same",
                activation="relu",
                kernel_initializer=KINIT,
            )(up2),
            attn3,
        ],
        name="up3",
    )

    up4 = layers.concatenate(
        [
            layers.Conv2DTranspose(
                ch["up4_deconv"],
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
