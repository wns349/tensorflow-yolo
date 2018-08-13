import os

import numpy as np
import tensorflow as tf

from . import yolo
from .layers import conv2d_bn_act, input_layer, route, detection_layer, yolo_layer, shortcut, upsample


def create_network(anchors, num_classes, is_training, scope="yolo", input_shape=(416, 416, 3)):
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


def load_weights(layers, weights_path):
    print("Reading pre-trained weights from {}".format(weights_path))

    # header
    with open(weights_path, "rb") as f:
        major, minor, revision, subversion, n = np.fromfile(f, count=5, dtype=np.int32)
        print("{} {} {} {} {}".format(major, minor, revision, subversion, n))
        weights = np.fromfile(f, dtype=np.float32)

    print("Found {} weight values.".format(len(weights)))

    return yolo.load_weights(layers, weights)


def find_bounding_boxes(out, anchors, threshold):
    h, w = out.shape[0:2]
    no_b = len(anchors)
    bboxes = []

    # TODO: maybe use matrix operation instead of for loops?
    for cy in range(h):
        for cw in range(w):
            for b in range(no_b):
                # calculate p(class|obj)
                prob_obj = yolo.sigmoid(out[cy, cw, b, 4])
                prob_classes = yolo.sigmoid(out[cy, cw, b, 5:])
                class_idx = np.argmax(prob_classes)
                class_prob = prob_classes[class_idx]
                p = prob_obj
                if p < threshold:  # if lower than threshold, pass
                    continue

                coords = out[cy, cw, b, 0:4]
                bbox = yolo.BoundingBox()
                bbox.x = (yolo.sigmoid(coords[0]) + cw) / w
                bbox.y = (yolo.sigmoid(coords[1]) + cy) / h
                bbox.w = (anchors[b][0] * np.exp(coords[2])) / w
                bbox.h = (anchors[b][1] * np.exp(coords[3])) / h
                bbox.class_idx = class_idx
                bbox.prob = prob_obj
                bboxes.append(bbox)
    return bboxes


class YoloV3(yolo.Yolo):
    def __init__(self):
        super().__init__()

    def initialize(self, params, is_training):
        assert "anchors" in params and isinstance(params["anchors"], list)
        self.anchors = np.reshape(params["anchors"], [3, -1, 2])
        self.anchors = self.anchors[::-1, :, :]  # reverse it for yolo layers

        assert "names" in params
        self.names = params["names"]

        # build network
        self.net = create_network(self.anchors, len(self.names), is_training)

    def predict(self, test_img_dir, out_dir, threshold, iou_threshold,
                batch_size=1,
                pretrained_weights_path=None,
                checkpoint_path=None):
        if pretrained_weights_path is None and checkpoint_path is None:
            raise ValueError("Path to either pretrained weights or checkpoint file is required.")

        # load images
        test_img_paths = yolo.load_image_paths(test_img_dir)
        if len(test_img_paths) == 0:
            print("No test images found in {}".format(test_img_dir))
            return
        os.makedirs(out_dir, exist_ok=True)  # create out directory, if not exists

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # load pretrained weights/checkpoint
            if pretrained_weights_path is not None:
                print("Running pretrained weights")
                ops = load_weights(self.net, pretrained_weights_path)
                sess.run(ops)
            if checkpoint_path is not None:
                # load_checkpoint(checkpoint_path)
                # TODO: pass
                pass

            # run prediction
            input_shape = tuple(self.net[0].out.get_shape().as_list()[1:3])

            test_batches = yolo.generate_test_batch(test_img_paths, batch_size, input_shape)
            for x_batch, paths in test_batches:
                net_out = sess.run(self.net[-1].out, feed_dict={self.net[0].out: x_batch})
                net_boxes = self.postprocess(net_out, threshold, iou_threshold)
                for boxes, path in zip(net_boxes, paths):
                    # draw box on image
                    new_img = yolo.draw_boxes(path, boxes, self.names)
                    # write to file
                    file_name, file_ext = os.path.splitext(os.path.basename(path))
                    out_path = os.path.join(out_dir, "{}_out{}".format(file_name, file_ext))
                    yolo.save_image(new_img, out_path)
                    print("{}: Found {} objects. Saved to {}".format(file_name, len(boxes), out_path))

    def postprocess(self, net_out, threshold, iou_threshold):
        results = []
        for out in net_out:
            idx = 0
            boxes = []
            for l in self.net[-1].yolos:
                dim = l.h * l.w * l.b
                l_out = np.reshape(out[idx:idx + dim, ...], [l.h, l.w, l.b, -1])
                boxes.extend(find_bounding_boxes(l_out, l.anchors, threshold))
                idx += dim
            results.append(yolo.non_maximum_suppression(boxes, iou_threshold))
        return results


if __name__ == "__main__":
    with open("./resource/coco.names", "r") as f:
        class_names = [n.lstrip().rstrip() for n in f.readlines()]
    o = YoloV3()
    o.initialize({
        "anchors": [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326],
        "names": class_names
    }, False)
    o.predict("./img/", "./out/", 0.5, 0.5, 1, "./bin/yolov3.weights")

    print("Done")
