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
    def __init__(self, net, filter_size, kernel_size, stride=1,
                 use_batch_normalization=True,
                 activation_fn="leaky",
                 is_training=False):

        def _activation_fn(x):  # activation layer
            if activation_fn == "leaky":
                return tf.nn.leaky_relu(x, alpha=_LEAKY_RELU)
            else:
                return x

        def _batch_norm(x):  # batch normalization layer
            x = slim.batch_norm(x, center=False, decay=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, scale=True,
                                fused=None, is_training=is_training)
            x = tf.nn.bias_add(x, slim.variable("biases", shape=[x.shape.as_list()[-1]],
                                                initializer=tf.zeros_initializer()))  # add bias explicitly
            return x

        if stride > 1:
            net = _pad(net, kernel_size)
        padding = "SAME" if stride == 1 else "VALID"
        self.out = slim.conv2d(net, filter_size, kernel_size,
                               stride=stride,
                               padding=padding,
                               normalizer_fn=_batch_norm if use_batch_normalization else None,
                               activation_fn=_activation_fn)

        # variable names in order to read darknet's pre-trained weights
        variable_prefix = self.out.name.rsplit("/", 1)[0]
        self.variable_names = ["{}/biases".format(variable_prefix)]
        if use_batch_normalization:
            self.variable_names.append("{}/BatchNorm/gamma".format(variable_prefix))
            self.variable_names.append("{}/BatchNorm/moving_mean".format(variable_prefix))
            self.variable_names.append("{}/BatchNorm/moving_variance".format(variable_prefix))
        self.variable_names.append("{}/weights".format(variable_prefix))


class max_pool2d(object):
    def __init__(self, net, kernel_size, stride=2):
        if stride > 1:
            net = _pad(net, kernel_size)
        padding = "SAME" if stride == 1 else "VALID"
        self.out = slim.max_pool2d(net, kernel_size, stride, padding)
        self.variable_names = []


class input_layer(object):
    def __init__(self, shape, name="input"):
        self.out = tf.placeholder(dtype=tf.float32, shape=shape, name=name)
        self.variable_names = []
