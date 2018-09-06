import numpy as np
import tensorflow as tf

from . import base
from .layers import conv2d_bn_act, input_layer, route, detection_layer, yolo_layer, shortcut, upsample


@staticmethod
def create_network(anchors, class_names, is_training, scope="yolo", input_shape=(416, 416, 3)):
    num_classes = len(class_names)
    anchors = np.reshape(anchors, [3, -1, 2])[::-1, :, :]  # reverse it for yolo layers
    tf.reset_default_graph()
    layers = []

    def _conv_shortcut(filter_size):
        layers.append(conv2d_bn_act(layers[-1].out, filter_size, 1, is_training=is_training, scope=scope))
        layers.append(conv2d_bn_act(layers[-1].out, filter_size * 2, 3, is_training=is_training, scope=scope))
        layers.append(shortcut(layers[-1].out, layers[-3].out))

    with tf.variable_scope(scope):
        layers.append(input_layer([None, input_shape[0], input_shape[1], input_shape[2]], "input"))

        # START darknet53
        layers.append(conv2d_bn_act(layers[-1].out, 32, 3, 1, is_training=is_training, scope=scope))
        layers.append(conv2d_bn_act(layers[-1].out, 64, 3, 2, is_training=is_training, scope=scope))
        _conv_shortcut(32)
        layers.append(conv2d_bn_act(layers[-1].out, 128, 3, 2, is_training=is_training, scope=scope))
        for _ in range(2):
            _conv_shortcut(64)
        layers.append(conv2d_bn_act(layers[-1].out, 256, 3, 2, is_training=is_training, scope=scope))
        for _ in range(8):
            _conv_shortcut(128)
        layers.append(conv2d_bn_act(layers[-1].out, 512, 3, 2, is_training=is_training, scope=scope))
        for _ in range(8):
            _conv_shortcut(256)

        layers.append(conv2d_bn_act(layers[-1].out, 1024, 3, 2, is_training=is_training, scope=scope))
        for _ in range(4):
            _conv_shortcut(512)
        # END darknet53

        # START yolo
        for _ in range(3):
            layers.append(conv2d_bn_act(layers[-1].out, 512, 1, is_training=is_training, scope=scope))
            layers.append(conv2d_bn_act(layers[-1].out, 1024, 3, is_training=is_training, scope=scope))
        layers.append(conv2d_bn_act(layers[-1].out, len(anchors[0]) * (5 + num_classes), 1, 1,
                                    use_batch_normalization=False,
                                    activation_fn="linear",
                                    is_training=is_training,
                                    scope=scope))

        layers.append(yolo_layer(layers[-1].out, anchors[0], num_classes, input_shape))
        yolo_1 = layers[-1]

        layers.append(route([layers[-4].out]))
        layers.append(conv2d_bn_act(layers[-1].out, 256, 1, is_training=is_training, scope=scope))
        layers.append(upsample(layers[-1].out, 2))
        layers.append(route([layers[-1].out, layers[61 + 1].out]))  # since input layer is included
        for _ in range(3):
            layers.append(conv2d_bn_act(layers[-1].out, 256, 1, is_training=is_training, scope=scope))
            layers.append(conv2d_bn_act(layers[-1].out, 512, 3, is_training=is_training, scope=scope))
        layers.append(conv2d_bn_act(layers[-1].out, len(anchors[1]) * (5 + num_classes), 1, 1,
                                    use_batch_normalization=False,
                                    activation_fn="linear",
                                    is_training=is_training,
                                    scope=scope))

        layers.append(yolo_layer(layers[-1].out, anchors[1], num_classes, input_shape))
        yolo_2 = layers[-1]

        layers.append(route([layers[-4].out]))
        layers.append(conv2d_bn_act(layers[-1].out, 128, 1, is_training=is_training, scope=scope))
        layers.append(upsample(layers[-1].out, 2))
        layers.append(route([layers[-1].out, layers[36 + 1].out]))  # since input layer is included
        for _ in range(3):
            layers.append(conv2d_bn_act(layers[-1].out, 128, 1, is_training=is_training, scope=scope))
            layers.append(conv2d_bn_act(layers[-1].out, 256, 3, is_training=is_training, scope=scope))
        layers.append(conv2d_bn_act(layers[-1].out, len(anchors[2]) * (5 + num_classes), 1, 1,
                                    use_batch_normalization=False,
                                    activation_fn="linear",
                                    is_training=is_training,
                                    scope=scope))

        layers.append(yolo_layer(layers[-1].out, anchors[2], num_classes, input_shape))
        yolo_3 = layers[-1]
        # END yolo

        # combine all
        layers.append(detection_layer([yolo_1, yolo_2, yolo_3]))

        # rename last layer for convenience
        layers[-1].out = tf.identity(layers[-1].out, "output")
    return layers


@staticmethod
def load_weights(layers, weights_path):
    print("Reading pre-trained weights from {}".format(weights_path))
    # header
    with open(weights_path, "rb") as f:
        major, minor, revision, subversion, n = np.fromfile(f, count=5, dtype=np.int32)
        print("{} {} {} {} {}".format(major, minor, revision, subversion, n))
        weights = np.fromfile(f, dtype=np.float32)
    print("Found {} weight values.".format(len(weights)))
    return base.load_weights(layers, weights)


def _find_bounding_boxes(out, anchors, threshold):
    h, w = out.shape[0:2]
    no_b = len(anchors)
    bboxes = []

    # TODO: maybe use matrix operation instead of for loops?
    for cy in range(h):
        for cw in range(w):
            for b in range(no_b):
                # calculate p(class|obj)
                prob_obj = base.sigmoid(out[cy, cw, b, 4])
                prob_classes = base.sigmoid(out[cy, cw, b, 5:])
                class_idx = np.argmax(prob_classes)
                class_prob = prob_classes[class_idx]
                p = prob_obj
                if p < threshold:  # if lower than threshold, pass
                    continue

                coords = out[cy, cw, b, 0:4]
                bbox = base.BoundingBox()
                bbox.x = (base.sigmoid(coords[0]) + cw) / w
                bbox.y = (base.sigmoid(coords[1]) + cy) / h
                bbox.w = (anchors[b][0] * np.exp(coords[2])) / w
                bbox.h = (anchors[b][1] * np.exp(coords[3])) / h
                bbox.class_idx = class_idx
                bbox.prob = prob_obj
                bboxes.append(bbox)
    return bboxes


@staticmethod
def find_bounding_boxes(net_out, net, threshold, iou_threshold, anchors, class_names):
    results = []
    for out in net_out:
        idx = 0
        boxes = []
        for l in net[-1].yolos:
            dim = l.h * l.w * l.b
            l_out = np.reshape(out[idx:idx + dim, ...], [l.h, l.w, l.b, -1])
            boxes.extend(_find_bounding_boxes(l_out, l.anchors, threshold))
            idx += dim
        results.append(base.non_maximum_suppression(boxes, iou_threshold))
    return results
