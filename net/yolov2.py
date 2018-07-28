import numpy as np
import tensorflow as tf
import os

from net.layers import conv2d_bn_act, input_layer, max_pool2d, route, reorg
from net import yolo


def create_full_network(num_anchors, num_classes, is_training, scope="yolo", input_shape=(416, 416, 3)):
    tf.reset_default_graph()
    layers = []
    with tf.variable_scope(scope):
        layers.append(input_layer([None, input_shape[0], input_shape[1], input_shape[2]], "input"))

        for filter_size in (32, 64):
            layers.append(conv2d_bn_act(layers[-1].out, filter_size, 3, stride=1, is_training=is_training))
            layers.append(max_pool2d(layers[-1].out, 2, stride=2))

        for filter_size in (128, 256):
            layers.append(conv2d_bn_act(layers[-1].out, filter_size, 3, stride=1, is_training=is_training))
            layers.append(conv2d_bn_act(layers[-1].out, filter_size // 2, 1, stride=1, is_training=is_training))
            layers.append(conv2d_bn_act(layers[-1].out, filter_size, 3, stride=1, is_training=is_training))
            layers.append(max_pool2d(layers[-1].out, 2, stride=2))

        layers.append(conv2d_bn_act(layers[-1].out, 512, 3, stride=1, is_training=is_training))
        layers.append(conv2d_bn_act(layers[-1].out, 256, 1, stride=1, is_training=is_training))
        layers.append(conv2d_bn_act(layers[-1].out, 512, 3, stride=1, is_training=is_training))
        layers.append(conv2d_bn_act(layers[-1].out, 256, 1, stride=1, is_training=is_training))
        layers.append(conv2d_bn_act(layers[-1].out, 512, 3, stride=1, is_training=is_training))
        layers.append(max_pool2d(layers[-1].out, 2, stride=2))

        layers.append(conv2d_bn_act(layers[-1].out, 1024, 3, stride=1, is_training=is_training))
        layers.append(conv2d_bn_act(layers[-1].out, 512, 1, stride=1, is_training=is_training))
        layers.append(conv2d_bn_act(layers[-1].out, 1024, 3, stride=1, is_training=is_training))
        layers.append(conv2d_bn_act(layers[-1].out, 512, 1, stride=1, is_training=is_training))
        layers.append(conv2d_bn_act(layers[-1].out, 1024, 3, stride=1, is_training=is_training))

        layers.append(conv2d_bn_act(layers[-1].out, 1024, 3, stride=1, is_training=is_training))
        layers.append(conv2d_bn_act(layers[-1].out, 1024, 3, stride=1, is_training=is_training))
        layers.append(route([layers[-9].out]))
        layers.append(conv2d_bn_act(layers[-1].out, 64, 1, stride=1, is_training=is_training))
        layers.append(reorg(layers[-1].out, 2))
        layers.append(route([layers[-1].out, layers[-4].out]))
        layers.append(conv2d_bn_act(layers[-1].out, 1024, 3, stride=1, is_training=is_training))

        layers.append(conv2d_bn_act(layers[-1].out, num_anchors * (5 + num_classes), 1, 1,
                                    use_batch_normalization=False,
                                    activation_fn="linear",
                                    is_training=is_training))

        # rename last layer for convenience
        layers[-1].out = tf.identity(layers[-1].out, "output")
    return layers


def load_weights(layers, weights_path):
    print("Reading pre-trained weights from {}".format(weights_path))

    # header
    with open(weights_path, "rb") as f:
        major, minor, revision = np.fromfile(f, count=3, dtype=np.int32)
        print("major, minor, revision: {}, {}, {}".format(major, minor, revision))

        if (major * 10 + minor) >= 2 and major < 1000 and minor < 1000:
            seen = np.fromfile(f, count=1, dtype=np.float32)
        else:
            seen = np.fromfile(f, count=1, dtype=np.int32)
        print("SEEN: ", seen)
        weights = np.fromfile(f, dtype=np.float32)

    return yolo.load_weights(layers, weights)


class YoloV2(yolo.Yolo):
    def __init__(self):
        super(yolo.Yolo, self).__init__()

    def initialize(self, params, is_training):
        assert "anchors" in params and isinstance(params["anchors"], list)
        self.anchors = np.reshape(params["anchors"], [-1, 2])
        assert "names" in params
        self.names = params["names"]

        # build network
        self.net = create_full_network(len(self.anchors), len(self.names), is_training)

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
            yolo.generate_test_batch(test_img_paths, batch_size)


if __name__ == "__main__":
    o = YoloV2()
    o.initialize({
        "anchors": [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828],
        "names": [str(i) for i in range(80)]
    }, False)
    o.predict("./img/", "./out/", 0.5, 0.5, 1, "./bin/yolov2.weights")

    print("Done")
