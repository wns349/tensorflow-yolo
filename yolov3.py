import numpy as np
import tensorflow as tf

from layers import input_layer, conv2d_bn_act, route, shortcut, yolo_layer, upsample, detection_layer
from yolo import load_weights as _load_weights


def create_network(anchors, labels, is_training=False, scope="yolo"):
    no_b = len(anchors) // 2
    no_c = len(labels)

    anchor_split = no_b // 3 * 2

    input_h = 416
    input_w = 416

    layers = []

    def _conv_shortcut(filter_size):
        layers.append(conv2d_bn_act(layers[-1].out, filter_size, 1))
        layers.append(conv2d_bn_act(layers[-1].out, filter_size * 2, 3))
        layers.append(shortcut(layers[-1].out, layers[-3].out))

    with tf.variable_scope(scope):
        layers.append(input_layer([None, input_h, input_w, 3], "input"))

        # START darknet53
        layers.append(conv2d_bn_act(layers[-1].out, 32, 3, 1))
        layers.append(conv2d_bn_act(layers[-1].out, 64, 3, 2))
        _conv_shortcut(32)
        layers.append(conv2d_bn_act(layers[-1].out, 128, 3, 2))
        for _ in range(2):
            _conv_shortcut(64)
        layers.append(conv2d_bn_act(layers[-1].out, 256, 3, 2))
        for _ in range(8):
            _conv_shortcut(128)

        layers.append(conv2d_bn_act(layers[-1].out, 512, 3, 2))
        for _ in range(8):
            _conv_shortcut(256)

        layers.append(conv2d_bn_act(layers[-1].out, 1024, 3, 2))
        for _ in range(4):
            _conv_shortcut(512)
        # END darknet53

        # START yolo
        for _ in range(3):
            layers.append(conv2d_bn_act(layers[-1].out, 512, 1))
            layers.append(conv2d_bn_act(layers[-1].out, 1024, 3))
        layers.append(conv2d_bn_act(layers[-1].out, no_b * (5 + no_c), 1, 1,
                                    use_batch_normalization=False,
                                    activation_fn="linear",
                                    is_training=is_training))

        layers.append(yolo_layer(layers[-1].out, anchors[:anchor_split], len(labels)))
        yolo_1 = layers[-1].out

        layers.append(route([layers[-4].out]))
        layers.append(conv2d_bn_act(layers[-1].out, 256, 1))
        layers.append(upsample(layers[-1].out, 2))
        layers.append(route([layers[-1].out, layers[61].out]))
        for _ in range(3):
            layers.append(conv2d_bn_act(layers[-1].out, 256, 1))
            layers.append(conv2d_bn_act(layers[-1].out, 512, 3))
        layers.append(conv2d_bn_act(layers[-1].out, no_b * (5 + no_c), 1, 1,
                                    use_batch_normalization=False,
                                    activation_fn="linear",
                                    is_training=is_training))

        layers.append(yolo_layer(layers[-1].out, anchors[anchor_split:2 * anchor_split], len(labels)))
        yolo_2 = layers[-1].out

        layers.append(route([layers[-4].out]))
        layers.append(conv2d_bn_act(layers[-1].out, 128, 1))
        layers.append(upsample(layers[-1].out, 2))
        layers.append(route([layers[-1].out, layers[36].out]))
        for _ in range(3):
            layers.append(conv2d_bn_act(layers[-1].out, 128, 1))
            layers.append(conv2d_bn_act(layers[-1].out, 256, 3))
        layers.append(conv2d_bn_act(layers[-1].out, no_b * (5 + no_c), 1, 1,
                                    use_batch_normalization=False,
                                    activation_fn="linear",
                                    is_training=is_training))

        layers.append(yolo_layer(layers[-1].out, anchors[2 * anchor_split:], len(labels)))
        yolo_3 = layers[-1].out
        # END yolo

        # combine all
        layers.append(detection_layer([yolo_1, yolo_2, yolo_3]))

        # rename last layer for convenience
        layers[-1].out = tf.identity(layers[-1].out, "output")
    return layers


def load_weights(layers, weights_path):
    print("Reading pre-trained weights from {}".format(weights_path))

    # header
    with open(weights_path, "rb") as f:
        major, minor, revision, subversion, n = np.fromfile(f, count=5, dtype=np.int32)
        print("{} {} {} {} {}".format(major, minor, revision, subversion, n))
        weights = np.fromfile(f, dtype=np.float32)

    print("Found {} weight values.".format(len(weights)))

    return _load_weights(layers, weights)


if __name__ == "__main__":
    with open("./resource/voc.labels", "r") as f:
        v_labels = [l.strip() for l in f.readlines()]
    v_anchors = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
    layers = create_network(v_anchors, v_labels)
    for i, l in enumerate(layers):
        print(i, l.out)

    ops = load_weights(layers, "./bin/yolov3.weights")
    with tf.Session() as sess:
        sess.run(ops)
    print("Done!")
