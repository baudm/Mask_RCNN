#!/usr/bin/env python

import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.layers import UpSampling2D

def resize_images(x, height_factor, width_factor, data_format):
    """Resizes the images contained in a 4D tensor.

    # Arguments
        x: Tensor or variable to resize.
        height_factor: Positive integer.
        width_factor: Positive integer.
        data_format: string, `"channels_last"` or `"channels_first"`.

    # Returns
        A tensor.

    # Raises
        ValueError: if `data_format` is neither `"channels_last"` or `"channels_first"`.
    """
    if data_format == 'channels_first':
        original_shape = K.int_shape(x)
        new_shape = tf.shape(x)[2:]
        new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
        x = K.permute_dimensions(x, [0, 2, 3, 1])
        x = tf.image.resize_bilinear(x, new_shape)
        x = K.permute_dimensions(x, [0, 3, 1, 2])
        x.set_shape((None, None, original_shape[2] * height_factor if original_shape[2] is not None else None,
                     original_shape[3] * width_factor if original_shape[3] is not None else None))
        return x
    elif data_format == 'channels_last':
        original_shape = K.int_shape(x)
        new_shape = tf.shape(x)[1:3]
        new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
        x = tf.image.resize_bilinear(x, new_shape)
        x.set_shape((None, original_shape[1] * height_factor if original_shape[1] is not None else None,
                     original_shape[2] * width_factor if original_shape[2] is not None else None, None))
        return x
    else:
        raise ValueError('Unknown data_format: ' + str(data_format))


class BilinearUpSampling2D(UpSampling2D):

    def call(self, inputs):
        return resize_images(inputs, self.size[0], self.size[1],
                               self.data_format)
