import tensorflow as tf
from tensorflow.python.keras import models
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K

import numpy as np


def build_model(input_tensor,
                include_top=True,
                weights_path=None,
                num_anchors=5,
                num_classes=80,
                trainable=True,
                **kwargs):
    '''Instantiates the darknet yolo-v2 architecture'''
    if not K.is_keras_tensor(input_tensor):
        img_input = layers.Input(input_tensor)
    else:
        img_input = input_tensor

    with tf.variable_scope('darknet'):
        x = _build_body(img_input, trainable)

        if include_top:
            x = layers.Conv2D(num_anchors * (5 + num_classes),
                              kernel_size=1,
                              strides=1,
                              padding='SAME',
                              use_bias=True,
                              activation='linear',
                              name='block23_conv')(x)

    model = models.Model(img_input, x, name='darknet')

    if weights_path is not None:
        model.load_weights(weights_path)
    return model


def _build_body(input_layer, trainable):
    def conv2d_bn_leaky(prev,
                        filter_size,
                        kernel_size,
                        strides,
                        prefix=''):
        prev = layers.Conv2D(filter_size,
                             kernel_size=kernel_size,
                             strides=strides,
                             padding='SAME',
                             use_bias=False,
                             trainable=trainable,
                             activation=None,
                             name='{}_conv'.format(prefix))(prev)
        prev = layers.BatchNormalization(momentum=0.9,
                                         epsilon=1e-5,
                                         trainable=trainable,
                                         name='{}_bn'.format(prefix))(prev)
        prev = layers.LeakyReLU(alpha=0.1,
                                name='{}_leaky'.format(prefix))(prev)
        return prev

    x = input_layer

    # block 1
    x = conv2d_bn_leaky(x, 32, 3, 1, 'block01')
    x = layers.MaxPool2D(2, 2, name='block01_maxpool')(x)

    # block 2
    x = conv2d_bn_leaky(x, 64, 3, 1, 'block02')
    x = layers.MaxPool2D(2, 2, name='block02_maxpool')(x)

    # block 3 ~ 8
    x = conv2d_bn_leaky(x, 128, 3, 1, 'block03')
    x = conv2d_bn_leaky(x, 64, 1, 1, 'block04')
    x = conv2d_bn_leaky(x, 128, 3, 1, 'block05')
    x = layers.MaxPool2D(2, 2, name='block05_maxpool')(x)
    x = conv2d_bn_leaky(x, 256, 3, 1, 'block06')
    x = conv2d_bn_leaky(x, 128, 1, 1, 'block07')
    x = conv2d_bn_leaky(x, 256, 3, 1, 'block08')
    x = layers.MaxPool2D(2, 2, name='block08_maxpool')(x)

    # block 9 ~ 13
    x = conv2d_bn_leaky(x, 512, 3, 1, 'block09')
    x = conv2d_bn_leaky(x, 256, 1, 1, 'block10')
    x = conv2d_bn_leaky(x, 512, 3, 1, 'block11')
    x = conv2d_bn_leaky(x, 256, 1, 1, 'block12')
    x = conv2d_bn_leaky(x, 512, 3, 1, 'block13')
    skip = x
    x = layers.MaxPool2D(2, 2, name='block13_maxpool')(x)

    # block 14 ~ 18
    x = conv2d_bn_leaky(x, 1024, 3, 1, 'block14')
    x = conv2d_bn_leaky(x, 512, 1, 1, 'block15')
    x = conv2d_bn_leaky(x, 1024, 3, 1, 'block16')
    x = conv2d_bn_leaky(x, 512, 1, 1, 'block17')
    x = conv2d_bn_leaky(x, 1024, 3, 1, 'block18')

    # block 19 ~ 20
    x = conv2d_bn_leaky(x, 1024, 3, 1, 'block19')
    x = conv2d_bn_leaky(x, 1024, 3, 1, 'block20')

    # reorg
    skip = conv2d_bn_leaky(skip, 64, 1, 1, 'block21')
    skip = layers.Lambda(lambda l: tf.space_to_depth(l, block_size=2),
                         name='block21_s2d')(skip)
    x = layers.Concatenate(name='block21_concat')([skip, x])

    # block 22
    x = conv2d_bn_leaky(x, 1024, 3, 1, 'block22')

    return x


def build_loss_fn(anchors, class_names):
    b = len(anchors)
    c = len(class_names)

    def loss_fn(y_true, y_pred):
        # y_true: [None, 13, 13, b * (5 + c)]
        h, w = y_pred.get_shape().as_list()[1:3]
        gt = K.reshape(y_true, [-1, h, w, b, 5 + c])
        pred = K.reshape(y_pred, [-1, h, w, b, 5 + c])

        # grid cell
        cell_h = K.tile(K.arange(h), [w])
        cell_w = K.tile(K.expand_dims(K.arange(w), 0), [h, 1])
        cell_w = K.reshape(K.transpose(cell_w), [-1])
        cell_hw = K.stack([cell_h, cell_w], 1)
        cell_hw = K.reshape(cell_hw, [-1, h, w, 1, 2])
        cell_hw = K.cast(cell_hw, dtype='float32')

        # anchor
        anchor_tensor = np.reshape(anchors, [1, 1, 1, b, 2])

        # prediction
        pred_xy = K.sigmoid(pred[..., 0:2]) + cell_hw
        pred_wh = K.exp(pred[..., 2:4]) * anchor_tensor
        pred_obj = K.sigmoid(pred[..., 4:5])
        pred_class = pred[..., 5:]

        # ground truth
        gt_xy = gt[..., 0:2]
        gt_wh = gt[..., 2:4]
        gt_obj = gt[..., 4:5]
        gt_class = K.argmax(gt[..., 5:], axis=-1)

        mask_ij = gt[..., 4:5]
        print(mask_ij)
        # K.print_tensor(mask_ij, message='mask_ij: ')

        # print(tf.shape(y_true))
        # print(tf.shape(y_pred))
        # print(y_true)
        # print(y_pred)
        # y_true = tf.reshape(y_true, [None, h, w, b, 5 + c])
        # y_pred = tf.reshape(y_pred, [None, h, w, b, 5 + c])

        t = K.mean(y_true - y_pred)
        return t

    return loss_fn
