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
            x = slim.batch_norm(x, center=False, decay=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, scale=True,
                                fused=None, is_training=is_training)
            x = tf.nn.bias_add(x, slim.variable("biases", shape=[x.shape.as_list()[-1]],
                                                initializer=tf.zeros_initializer()))  # add bias explicitly
            return x

        if stride > 1:
            prev = _pad(prev, kernel_size)
        padding = "SAME" if stride == 1 else "VALID"
        self.out = slim.conv2d(prev, filter_size, kernel_size,
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
    def __init__(self, yolo_layers):
        self.out = tf.concat(yolo_layers, axis=1)
        self.variable_names = []


class yolo_layer(object):
    def __init__(self, prev, input_layer, no_c, sub_anchors):
        input_h, input_w = input_layer.get_shape().as_list()[1:3]

        out_shape = prev.get_shape().as_list()
        grid_h, grid_w = out_shape[1:3]
        dim = grid_h * grid_w
        bbox_attrs = 5 + no_c

        stride = (input_w // grid_w, input_h // grid_h)
        anchors = [(sub_anchors[i] / stride[0], sub_anchors[i + 1] / stride[1]) for i in range(0, len(sub_anchors), 2)]
        no_b = len(anchors)

        self.out = tf.reshape(prev, [-1, no_b * dim, bbox_attrs])

        box_xy, box_wh, prob_obj, prob_class = tf.split(self.out, [2, 2, 1, no_c], axis=-1)

        box_xy = tf.nn.sigmoid(box_xy)
        prob_obj = tf.nn.sigmoid(prob_obj)

        grid_x = tf.range(grid_w, dtype=tf.float32)
        grid_y = tf.range(grid_h, dtype=tf.float32)
        a, b = tf.meshgrid(grid_x, grid_y)
        x_offset = tf.reshape(a, [-1, 1])
        y_offset = tf.reshape(b, [-1, 1])

        xy_offset = tf.concat([x_offset, y_offset], axis=-1)
        _t = tf.tile(xy_offset, [1, no_b])
        xy_offset = tf.reshape(_t, [1, -1, 2])  # flatten

        box_xy = box_xy + xy_offset
        box_xy = box_xy * stride

        anchors = tf.tile(anchors, [dim, 1])
        box_wh = tf.exp(box_wh) * anchors
        box_wh = box_wh * stride

        detections = tf.concat([box_xy, box_wh, prob_obj], axis=-1)
        classes = tf.nn.sigmoid(prob_class)
        self.out = tf.concat([detections, classes], axis=-1)
        self.variable_names = []
