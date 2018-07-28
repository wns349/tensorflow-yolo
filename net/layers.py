import tensorflow as tf

slim = tf.contrib.slim

# yolo defaults
_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-5
_LEAKY_RELU = 0.1


def _pad(prev, kernel_size, mode="CONSTANT"):
    pad_total = kernel_size - 1
    pad_start = pad_total // 2
    pad_end = pad_total - pad_start

    return tf.pad(prev, [[0, 0], [pad_start, pad_end], [pad_start, pad_end], [0, 0]], mode=mode)


class conv2d_bn_act(object):
    def __init__(self, prev, filter_size, kernel_size, stride=1,
                 use_batch_normalization=True,
                 activation_fn="leaky",
                 is_training=False):

        def _activation_fn(x):  # activation layer
            if activation_fn == "leaky":
                return tf.nn.leaky_relu(x, alpha=_LEAKY_RELU)
            else:
                return x

        def _batch_norm(x):  # batch normalization layer
            x = slim.batch_norm(x, decay=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, scale=True,
                                fused=None, is_training=is_training)
            return x

        if stride > 1:
            prev = _pad(prev, kernel_size)
        padding = "SAME" if stride == 1 else "VALID"
        self.out = slim.conv2d(prev, filter_size, kernel_size,
                               stride=stride,
                               padding=padding,
                               normalizer_fn=_batch_norm if use_batch_normalization else None,
                               activation_fn=_activation_fn)

        # order variable names in sequence to match darknet's pre-trained weights
        variable_prefix = self.out.name.rsplit("/", 1)[0]
        self.variable_names = []
        if use_batch_normalization:
            self.variable_names.append("{}/BatchNorm/beta".format(variable_prefix))
            self.variable_names.append("{}/BatchNorm/gamma".format(variable_prefix))
            self.variable_names.append("{}/BatchNorm/moving_mean".format(variable_prefix))
            self.variable_names.append("{}/BatchNorm/moving_variance".format(variable_prefix))
        else:
            self.variable_names.append("{}/biases".format(variable_prefix))
        self.variable_names.append("{}/weights".format(variable_prefix))


class max_pool2d(object):
    def __init__(self, prev, kernel_size, stride=2):
        if stride > 1:
            prev = _pad(prev, kernel_size)
        padding = "SAME" if stride == 1 else "VALID"
        self.out = slim.max_pool2d(prev, kernel_size, stride, padding)
        self.variable_names = []


class route(object):
    def __init__(self, prevs):
        self.out = tf.concat(prevs, axis=3)
        self.variable_names = []


class reorg(object):
    def __init__(self, prev, stride):
        self.out = tf.extract_image_patches(prev,
                                            ksizes=[1, stride, stride, 1],
                                            strides=[1, stride, stride, 1],
                                            rates=[1] * 4,
                                            padding="VALID")
        self.variable_names = []


class shortcut(object):
    def __init__(self, prev, shortcut_out):
        self.out = prev + shortcut_out
        self.variable_names = []


class input_layer(object):
    def __init__(self, shape, name="input"):
        self.out = tf.placeholder(dtype=tf.float32, shape=shape, name=name)
        self.variable_names = []


class upsample(object):
    def __init__(self, prev, stride):
        h, w = prev.get_shape().as_list()[1:3]
        self.out = tf.image.resize_nearest_neighbor(prev, (h * stride, w * stride))
        self.variable_names = []


class detection_layer(object):
    def __init__(self, yolos):
        self.yolos = yolos
        self.out = tf.concat([l.out for l in yolos], axis=1)
        self.variable_names = []


class yolo_layer(object):
    def __init__(self, prev, sub_anchors, no_c, input_shape):
        out_shape = prev.get_shape().as_list()
        self.h, self.w = out_shape[1:3]
        stride = (input_shape[0] / self.h, input_shape[1] / self.w)
        self.anchors = [(a[0] / stride[0], a[1] / stride[1]) for a in sub_anchors]  # anchor is in (w, h)
        self.b = len(self.anchors)
        self.out = tf.reshape(prev, [-1, self.h * self.w * self.b, 5 + no_c])
        self.variable_names = []
