import numpy as np
import tensorflow as tf

from net.layers import input_layer, conv2d_bn_act, route, shortcut, yolo_layer, upsample, detection_layer
from inference import _find_bounding_boxes, _non_maximum_suppression
from yolo import load_weights as _load_weights


def create_network(anchors, class_names, is_training=False, scope="yolo", input_shape=(416, 416, 3)):
    tf.reset_default_graph()
    no_b = len(anchors) // 2
    no_c = len(class_names)

    anchor_split = no_b // 3 * 2

    layers = []

    def _conv_shortcut(filter_size):
        layers.append(conv2d_bn_act(layers[-1].out, filter_size, 1))
        layers.append(conv2d_bn_act(layers[-1].out, filter_size * 2, 3))
        layers.append(shortcut(layers[-1].out, layers[-3].out))

    with tf.variable_scope(scope):
        layers.append(input_layer([None, input_shape[0], input_shape[1], input_shape[2]], "input"))

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
        sub_anchors = anchors[:anchor_split]
        sub_no_b = len(sub_anchors) // 2
        layers.append(conv2d_bn_act(layers[-1].out, sub_no_b * (5 + no_c), 1, 1,
                                    use_batch_normalization=False,
                                    activation_fn="linear",
                                    is_training=is_training))

        layers.append(yolo_layer(layers[-1].out, sub_anchors, no_c))
        yolo_1 = layers[-1]

        layers.append(route([layers[-4].out]))
        layers.append(conv2d_bn_act(layers[-1].out, 256, 1))
        layers.append(upsample(layers[-1].out, 2))
        layers.append(route([layers[-1].out, layers[61].out]))
        for _ in range(3):
            layers.append(conv2d_bn_act(layers[-1].out, 256, 1))
            layers.append(conv2d_bn_act(layers[-1].out, 512, 3))
        sub_anchors = anchors[anchor_split:2 * anchor_split]
        sub_no_b = len(sub_anchors) // 2
        layers.append(conv2d_bn_act(layers[-1].out, sub_no_b * (5 + no_c), 1, 1,
                                    use_batch_normalization=False,
                                    activation_fn="linear",
                                    is_training=is_training))

        layers.append(yolo_layer(layers[-1].out, sub_anchors, no_c))
        yolo_2 = layers[-1]

        layers.append(route([layers[-4].out]))
        layers.append(conv2d_bn_act(layers[-1].out, 128, 1))
        layers.append(upsample(layers[-1].out, 2))
        layers.append(route([layers[-1].out, layers[36].out]))
        for _ in range(3):
            layers.append(conv2d_bn_act(layers[-1].out, 128, 1))
            layers.append(conv2d_bn_act(layers[-1].out, 256, 3))
        sub_anchors = anchors[2 * anchor_split:]
        sub_no_b = len(sub_anchors) // 2
        layers.append(conv2d_bn_act(layers[-1].out, sub_no_b * (5 + no_c), 1, 1,
                                    use_batch_normalization=False,
                                    activation_fn="linear",
                                    is_training=is_training))

        layers.append(yolo_layer(layers[-1].out, sub_anchors, no_c))
        yolo_3 = layers[-1]
        # END yolo

        # combine all
        layers.append(detection_layer([yolo_1.out, yolo_2.out, yolo_3.out]))

    return layers, [yolo_1.yolo, yolo_2.yolo, yolo_3.yolo]


def load_weights(layers, weights_path):
    print("Reading pre-trained weights from {}".format(weights_path))

    # header
    with open(weights_path, "rb") as f:
        major, minor, revision, subversion, n = np.fromfile(f, count=5, dtype=np.int32)
        print("{} {} {} {} {}".format(major, minor, revision, subversion, n))
        weights = np.fromfile(f, dtype=np.float32)

    print("Found {} weight values.".format(len(weights)))

    return _load_weights(layers, weights)


def postprocess(net_outs, anchors, threshold, iou_threshold):
    results = []

    anchor_idx = (len(anchors) // 2) // len(net_outs)

    net_boxes = {}
    for net_out in net_outs:  # [3, ?, H, W, B, 5 + C]
        h, w = net_out.shape[1:3]
        for i, out in enumerate(net_out):  # [?, H, W, B, 5 + C]
            if i in net_boxes:
                net_boxes[i].extend(
                    _find_bounding_boxes(out, anchors[i * anchor_idx:(i + 1) * anchor_idx], threshold, h, w))
            else:
                net_boxes[i] = _find_bounding_boxes(out, anchors[i * anchor_idx:(i + 1) * anchor_idx], threshold, h, w)
    for boxes in net_boxes.values():
        results.append(_non_maximum_suppression(boxes, iou_threshold))
    return results


def preprocess(img, size=(416, 416)):
    imsz = cv2.resize(img, size)
    imsz = imsz / 255.  # to make values lie between 0 and 1
    imsz = imsz[:, :, ::-1]  # BGR to RGB
    return imsz


if __name__ == "__main__":
    with open("./resource/voc.names", "r") as f:
        v_class_names = [l.strip() for l in f.readlines()]
    v_anchors = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
    layers, yolo_layers = create_network(v_anchors, v_class_names)

    import cv2
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Use this to run on CPU

    img = cv2.imread("./img/person.jpg")
    img = preprocess(img)
    net_img = np.expand_dims(img, axis=0)

    ops = load_weights(layers, "./bin/yolov3.weights")
    with tf.Session() as sess:
        sess.run(ops)
        net_outs = sess.run(yolo_layers, feed_dict={layers[0].out: net_img})
        r = postprocess(net_outs, v_anchors, 0.5, 0.5)
        print(r)

    print("Done!")
