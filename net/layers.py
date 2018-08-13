import tensorflow as tf

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
    name_count = 0

    def __init__(self, prev, filter_size, kernel_size, stride=1,
                 use_batch_normalization=True,
                 activation_fn="leaky",
                 is_training=False,
                 scope="yolo"):
        name = "{}_{}".format(conv2d_bn_act.__name__, conv2d_bn_act.name_count)
        conv2d_bn_act.name_count += 1

        if stride > 1:
            prev = _pad(prev, kernel_size)
        padding = "SAME" if stride == 1 else "VALID"
        self.out = tf.layers.conv2d(
            inputs=prev,
            filters=filter_size,
            kernel_size=kernel_size,
            padding=padding,
            strides=(stride, stride),
            use_bias=not use_batch_normalization,
            name=name
        )

        if use_batch_normalization:
            self.out = tf.layers.batch_normalization(
                self.out,
                training=is_training,
                momentum=_BATCH_NORM_DECAY,
                epsilon=_BATCH_NORM_EPSILON,
                name=name
            )

        if activation_fn == "leaky":
            self.out = tf.nn.leaky_relu(self.out, alpha=_LEAKY_RELU, name=name)

        # order variable names in sequence to match darknet's pre-trained weights
        variable_prefix = name
        self.variable_names = []
        if use_batch_normalization:
            self.variable_names.append("{}/{}/beta".format(scope, variable_prefix))
            self.variable_names.append("{}/{}/gamma".format(scope, variable_prefix))
            self.variable_names.append("{}/{}/moving_mean".format(scope, variable_prefix))
            self.variable_names.append("{}/{}/moving_variance".format(scope, variable_prefix))
        else:
            self.variable_names.append("{}/{}/bias".format(scope, variable_prefix))
        self.variable_names.append("{}/{}/kernel".format(scope, variable_prefix))


class max_pool2d(object):
    def __init__(self, prev, kernel_size, stride=2):
        if stride > 1:
            prev = _pad(prev, kernel_size)
        padding = "SAME" if stride == 1 else "VALID"
        self.out = tf.layers.max_pooling2d(
            inputs=prev,
            pool_size=kernel_size,
            strides=stride,
            padding=padding
        )
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
