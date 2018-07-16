import os

import cv2
import numpy as np
import tensorflow as tf

from inference import preprocess, postprocess
from layers import conv2d_bn_act, input_layer, max_pool2d

COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255)]


def _create_network(anchors, labels, is_training=False, scope="yolo"):
    no_b = len(anchors) // 2
    no_c = len(labels)

    layers = []
    with tf.variable_scope(scope):
        layers.append(input_layer([None, 416, 416, 3], "input"))

        for filter_size, pool_stride in zip([16, 32, 64, 128, 256, 512], [2, 2, 2, 2, 2, 1]):
            layers.append(conv2d_bn_act(layers[-1].out, filter_size, 3, is_training=is_training))
            layers.append(max_pool2d(layers[-1].out, 2, stride=pool_stride))

        for _ in range(2):
            layers.append(conv2d_bn_act(layers[-1].out, 1024, 3, is_training=is_training))

        layers.append(conv2d_bn_act(layers[-1].out, no_b * (5 + no_c), 1, 1,
                                    use_batch_normalization=False,
                                    activation_fn="linear",
                                    is_training=is_training))

        # rename last layer for convenience
        layers[-1].out = tf.identity(layers[-1].out, "output")
    return layers


def _load_weights(layers, tf_session, weights_path):
    print("Reading pre-trained weights from {}".format(weights_path))
    file_size = os.path.getsize(weights_path)

    major, minor, revision = np.memmap(weights_path, shape=3, offset=0, dtype=np.int)
    print("major, minor, revision: {}, {}, {}".format(major, minor, revision))

    if (major * 10 + minor) >= 2 and major < 1000 and minor < 1000:
        seen = np.memmap(weights_path, shape=1, offset=12, dtype=np.float32)
        offset = 20
    else:
        seen = np.memmap(weights_path, shape=1, offset=12, dtype=np.int)
        offset = 16
    print("SEEN: ", seen)

    variables = {v.op.name: v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="yolo")}
    for layer in layers:
        for variable_name in layer.variable_names:
            var = variables[variable_name]
            print("Loading pre-trained values for {}".format(var.name))
            tokens = var.name.split("/")
            size = np.prod(var.shape.as_list())
            shape = var.shape.as_list()
            if "weights" in tokens[-1]:
                shape = [shape[3], shape[2], shape[0], shape[1]]
            data = np.memmap(weights_path, shape=size, offset=offset, dtype=np.float32)
            value = np.reshape(data, shape)
            if "weights" in tokens[-1]:
                value = np.transpose(value, (2, 3, 1, 0))
            tf_session.run(var.assign(value))
            offset += size * 4

    print("Weights loaded. ({}/{} read)".format(offset, file_size))
    if offset != file_size:
        print("(warning) Offset and file size do not match. Possibly an incorrect weights file.")


def predict(test_image_path, params):
    anchors = params["anchors"]
    labels = params["labels"]
    weights = params["weights"]
    threshold = float(params["threshold"])
    iou_threshold = float(params["iou_threshold"])
    scope = "yolo"

    # create network
    layers = _create_network(anchors, labels, False, scope)
    layer_in = layers[0].out
    layer_in_h, layer_in_w = layer_in.shape.as_list()[1:3]
    layer_out = layers[-1].out
    layer_out_h, layer_out_w = layer_out.shape.as_list()[1:3]

    # run prediction
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _load_weights(layers, sess, weights)

        print("Test image: {}".format(test_image_path))
        img = cv2.imread(test_image_path)
        net_img = preprocess(img, (layer_in_h, layer_in_w))
        net_img = np.expand_dims(net_img, axis=0)  # since only a single image is evaluated here

        net_out = sess.run(layer_out, feed_dict={layer_in: net_img})
        net_out = np.reshape(net_out, [-1, layer_out_h, layer_out_w, len(anchors) // 2, 5 + len(labels)])

        results = postprocess(net_out, anchors, threshold, iou_threshold, layer_out_h, layer_out_w)
        result = results[0]  # only a single image

        # write result to file
        org_img_h, org_img_w = img.shape[0:2]
        name, ext = os.path.splitext(os.path.abspath(test_image_path))
        out_path = "{}_out{}".format(name, ext)
        for box in result:
            top_left = box.get_top_left(org_img_h, org_img_w)
            bottom_right = box.get_bottom_right(org_img_h, org_img_w)
            top_left = (int(top_left[0]), int(top_left[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
            cv2.rectangle(img, top_left, bottom_right, COLORS[box.class_idx % len(COLORS)], thickness=3)
            label = labels[box.class_idx]
            cv2.putText(img, "{} {:.3f}".format(label, box.prob), (top_left[0], top_left[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[box.class_idx % len(COLORS)], thickness=1)
        cv2.imwrite(out_path, img)
        return out_path
