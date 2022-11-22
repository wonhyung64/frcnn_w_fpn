#%%
import numpy as np
import tensorflow as tf
from typing import *
from tensorflow.python.framework.ops import EagerTensor


#%%
class AnchorBox:

    def __init__(
        self,
        aspect_ratios: List[float]=[0.5, 1.0, 2.0],
        scales: List[float]=[2 ** x for x in [0., 1/3, 2/3]],
        strides: List[int]=[2 ** i for i in range(3, 8)],
        areas: List[float]=[x ** 2 for x in [32.0, 64.0, 128.0, 256.0, 512.0]],
        ):

        self._aspect_ratios: List[float] = aspect_ratios
        self._scales: List[float] = scales
        self._strides: List[int] = strides
        self._areas: List[float] = areas
        self._num_anchors: int = len(aspect_ratios) * len(scales)
        self._anchor_dims: List[EagerTensor] = self._computeDims()


    def _computeDims(self) -> EagerTensor:
        anchor_dims_all: list = []
        for area in self._areas:
            anchor_dims: list = []
            for ratio in self._aspect_ratios:
                anchor_height: EagerTensor = tf.math.sqrt(area / ratio)
                anchor_width: EagerTensor = tf.cast(area, tf.float32) / anchor_height
                dims: EagerTensor = tf.reshape(
                    tf.stack([anchor_width, anchor_height], axis=-1), [1, 1, 2]
                )
                for scale in self._scales:
                    anchor_dims.append(scale * dims)
            anchor_dims_all.append(tf.stack(anchor_dims, axis=-2))

        return anchor_dims_all


    def _getAnchors(self, feature_height: int, feature_width: int, level: int) -> EagerTensor:
        ry: EagerTensor = tf.range(feature_height, dtype=tf.float32) + tf.cast(0.5, dtype=tf.float32)
        rx: EagerTensor = tf.range(feature_width, dtype=tf.float32) + tf.cast(0.5, dtype=tf.float32)
        centers: EagerTensor = tf.stack(tf.meshgrid(ry, rx), axis=-1)
        centers *= tf.cast(self._strides[level - 3], dtype=tf.float32)
        centers = tf.expand_dims(centers, axis=-2)
        centers = tf.tile(centers, [1, 1, self._num_anchors, 1])
        dims: EagerTensor = tf.tile(
            self._anchor_dims[level - 3], [feature_height, feature_width, 1, 1]
        )
        anchors = tf.concat([centers, dims], axis=-1)
        anchors = tf.reshape(
                anchors, [feature_height * feature_width * self._num_anchors, 4]
            )
        
        return anchors


    def getAnchors(self, image_height: int, image_width: int) -> EagerTensor:
        anchors: List[EagerTensor] = [
            self._getAnchors(
                tf.math.ceil(image_height / 2 ** i),
                tf.math.ceil(image_width / 2 ** i),
                i,
            )
            for i in range(3, 8)
        ]
        anchors_tf: EagerTensor = tf.concat(anchors, axis=0)

        return anchors_tf


#%%
AnchorBox().getAnchors(512, 512)


feature_height=512 // 2**3
feature_width=512 // 2**3
level = 3




image_height: int = 512

image_width: int = 512


getAnchors(image_height, image_width)



