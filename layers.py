import tensorflow as tf

slim = tf.contrib.slim

# yolo defaults
_BATCH_NORM_MOMENTUM = 0.9
_BATCH_NORM_EPSILON = 1e-5
_LEAKY_RELU = 0.1


def _pad(prev, kernel_size, mode="CONSTANT"):
    pad_total = kernel_size - 1
    pad_start = pad_total // 2
    pad_end = pad_total - pad_start

    return tf.pad(prev, [[0, 0], [pad_start, pad_end], [pad_start, pad_end], [0, 0]], mode=mode)


class conv2d_bn_act(object):
    def __init__(self, net, filter_size, kernel_size, stride=1,
                 use_batch_normalization=True,
                 activation_fn="leaky"):
        self.variable_names = []
        if stride > 1:
            net = _pad(net, kernel_size)
        padding = "SAME" if stride == 1 else "VALID"
        self.out = tf.layers.conv2d(net, filter_size, kernel_size, strides=stride, padding=padding)
        tensor_prefix = "/".join(self.out.name.split("/")[:-1])
        self.variable_names.insert(0, "{}/kernel".format(tensor_prefix))
        self.variable_names.insert(0, "{}/bias".format(tensor_prefix))

        if use_batch_normalization:
            self.out = tf.layers.batch_normalization(self.out, momentum=_BATCH_NORM_MOMENTUM,
                                                     epsilon=_BATCH_NORM_EPSILON)
            tensor_prefix = "/".join(self.out.name.split("/")[:-1])
            self.variable_names.insert(0, "{}/moving_variance".format(tensor_prefix))
            self.variable_names.insert(0, "{}/moving_mean".format(tensor_prefix))
            self.variable_names.insert(0, "{}/beta".format(tensor_prefix))

        if activation_fn == "leaky":
            self.out = tf.nn.leaky_relu(self.out, alpha=_LEAKY_RELU)


class max_pool2d(object):
    def __init__(self, net, kernel_size, stride=2):
        self.variable_names = []
        if stride > 1:
            net = _pad(net, kernel_size)
        padding = "SAME" if stride == 1 else "VALID"
        self.out = slim.max_pool2d(net, kernel_size, stride, padding)


class input_layer(object):
    def __init__(self, shape, name="input"):
        self.variable_names = []
        self.out = tf.placeholder(dtype=tf.float32, shape=shape, name=name)
