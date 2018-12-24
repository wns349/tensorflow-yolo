import tensorflow as tf
from tensorflow.python import keras

models = keras.models
layers = keras.layers
K = keras.backend

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
                              kernel_initializer='he_uniform',
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
        gt_class = K.argmax(gt[..., 5:], axis=-1)

        # iou
        gt_wh_half = gt_wh / 2.
        gt_min = gt_xy - gt_wh_half
        gt_max = gt_xy + gt_wh_half
        gt_area = gt_wh[..., 0] * gt_wh[..., 1]
        pred_wh_half = pred_wh / 2.
        pred_min = pred_xy - pred_wh_half
        pred_max = pred_xy + pred_wh_half
        pred_area = pred_wh[..., 0] * pred_wh[..., 1]
        intersection_min = K.maximum(pred_min, gt_min)
        intersection_max = K.minimum(pred_max, gt_max)
        intersection_wh = K.maximum(intersection_max - intersection_min, 0.)
        intersection_area = intersection_wh[..., 0] * intersection_wh[..., 1]
        union_area = pred_area + gt_area - intersection_area
        iou_scores = tf.truediv(intersection_area, union_area)

        # update gt
        gt_obj = iou_scores * gt[..., 4]
        gt_obj = K.expand_dims(gt_obj, axis=-1)

        # get mask info
        mask_ij = gt[..., 4:5]
        mask_i = K.sum(mask_ij, axis=-2)

        n_mask_ij = K.maximum(K.sum(tf.to_float(mask_ij > 0.0)), 1e-6)
        n_mask_i = K.maximum(K.sum(tf.to_float(mask_i > 0.0)), 1e-6)

        loss_xy = 1. * tf.reduce_sum(mask_ij * tf.square(gt_xy - pred_xy)) / n_mask_ij
        loss_wh = 1. * tf.reduce_sum(mask_ij * tf.square(tf.sqrt(gt_wh) - tf.sqrt(pred_wh))) / n_mask_ij
        loss_obj = 5. * tf.reduce_sum(mask_ij * tf.square(gt_obj - pred_obj)) / n_mask_ij
        loss_noobj = 1. * tf.reduce_sum((1 - mask_ij) * tf.square(gt_obj - pred_obj)) / n_mask_ij
        loss_class = 1. * tf.reduce_sum(
            mask_i * tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt_class, logits=pred_class)) / n_mask_i

        loss = loss_xy + loss_wh + loss_obj + loss_noobj + loss_class
        return loss

    return loss_fn
