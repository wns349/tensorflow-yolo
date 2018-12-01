import tensorflow as tf
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import InputLayer, Conv2D, BatchNormalization, LeakyReLU, MaxPool2D, Lambda, \
    Concatenate, Input
from tensorflow.python.keras import backend as K
import numpy as np


def get_variable_names(layer, type=None, has_bias=False):
    names = []
    if type == "bn":
        variable_name = layer.name.rsplit("/", maxsplit=2)[0]
        names.append("{}/beta".format(variable_name))
        names.append("{}/gamma".format(variable_name))
        names.append("{}/moving_mean".format(variable_name))
        names.append("{}/moving_variance".format(variable_name))
    elif type == "conv":
        variable_name = layer.name.rsplit("/", maxsplit=1)[0]
        if has_bias:
            names.append("{}/bias".format(variable_name))
        names.append("{}/kernel".format(variable_name))
    return names


def build_feature_extractor_full_yolo(input_layer, scope="yolo", variable_placeholder=[]):
    with tf.variable_scope(scope):
        # define helper methods
        def conv2d_bn_leaky(_prev, _filter_size, _kernel_size, _strides):
            _conv = Conv2D(_filter_size,
                           kernel_size=_kernel_size,
                           strides=_strides,
                           padding="SAME",
                           use_bias=False,
                           activation=None)(_prev)
            _bn = BatchNormalization(momentum=0.9, epsilon=1e-5)(_conv)
            _leaky = LeakyReLU(alpha=0.1)(_bn)

            variable_placeholder.extend(get_variable_names(_bn, "bn"))
            variable_placeholder.extend(get_variable_names(_conv, "conv"))
            return _leaky

        def space_to_depth_x2(x):
            return tf.space_to_depth(x, block_size=2)

        x = input_layer
        for f in (32, 64):
            x = conv2d_bn_leaky(x, f, 3, 1)
            x = MaxPool2D(2, strides=2)(x)

        for f in (128, 256):
            x = conv2d_bn_leaky(x, f, 3, 1)
            x = conv2d_bn_leaky(x, f // 2, 1, 1)
            x = conv2d_bn_leaky(x, f, 3, 1)
            x = MaxPool2D(2, strides=2)(x)

        for _ in range(2):
            x = conv2d_bn_leaky(x, 512, 3, 1)
            x = conv2d_bn_leaky(x, 256, 1, 1)
        x = conv2d_bn_leaky(x, 512, 3, 1)
        skip = x
        x = MaxPool2D(2, strides=2)(x)

        for _ in range(2):
            x = conv2d_bn_leaky(x, 1024, 3, 1)
            x = conv2d_bn_leaky(x, 512, 1, 1)
        for _ in range(3):
            x = conv2d_bn_leaky(x, 1024, 3, 1)

        # reorg
        skip = conv2d_bn_leaky(skip, 64, 1, 1)
        skip = Lambda(space_to_depth_x2)(skip)

        x = Concatenate()([skip, x])

        x = conv2d_bn_leaky(x, 1024, 3, 1)

        return x


def build_output_layer(prev, num_anchors=5, num_classes=80, variable_placeholder=[]):
    x = Conv2D(num_anchors * (5 + num_classes),
               kernel_size=1,
               strides=1,
               padding="SAME",
               use_bias=True,
               activation="linear")(prev)

    variable_placeholder.extend(get_variable_names(x, "conv", True))

    return x


def load_darknet_weights(model, variable_placeholder, weights_path):
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

    variables = {v.op.name: v for v in model.variables}
    read = 0
    for variable_name in variable_placeholder:
        var = variables[variable_name]
        tokens = var.name.split("/")
        size = np.prod(var.shape.as_list())
        shape = var.shape.as_list()
        if "kernel" in tokens[-1]:
            shape = [shape[3], shape[2], shape[0], shape[1]]
        value = np.reshape(weights[read: read + size], shape)
        if "kernel" in tokens[-1]:
            value = np.transpose(value, (2, 3, 1, 0))
        K.set_value(var, value)
        read += size

    print("Weights ({}/{} read)".format(read, len(weights)))


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class BoundingBox(object):
    def __init__(self, x=0., y=0., w=0., h=0., cx=0, cy=0, class_idx=-1, prob=-1.):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.cx = cx
        self.cy = cy
        self.class_idx = class_idx
        self.prob = prob

    def get_top_left(self, h=1., w=1.):
        return (self.x - self.w / 2.) * w, (self.y - self.h / 2.) * h

    def get_bottom_right(self, h=1., w=1.):
        return (self.x + self.w / 2.) * w, (self.y + self.h / 2.) * h


def iou_score(box1, box2):
    box1_min, box1_max = box1.get_top_left(), box1.get_bottom_right()
    box1_area = box1.w * box1.h
    box2_min, box2_max = box2.get_top_left(), box2.get_bottom_right()
    box2_area = box2.w * box2.h

    intersect_min = np.maximum(box1_min, box2_min)
    intersect_max = np.minimum(box1_max, box2_max)
    intersect_wh = np.maximum(intersect_max - intersect_min, 0)
    intersect_area = intersect_wh[0] * intersect_wh[1]
    union_area = np.maximum(box1_area + box2_area - intersect_area, 1e-8)

    return intersect_area / union_area


if __name__ == '__main__':
    variable_placeholder = []
    input_layer = Input(shape=[416, 416, 3])
    feature_extractor = build_feature_extractor_full_yolo(input_layer, variable_placeholder=variable_placeholder)
    output_layer = build_output_layer(feature_extractor, num_anchors=5, num_classes=80,
                                      variable_placeholder=variable_placeholder)

    model = Model([input_layer], [output_layer])
    print(model.summary())
    print("variables: ", len(variable_placeholder))
    load_darknet_weights(model, variable_placeholder, "../bin/yolov2.weights")

    model.save_weights("./checkpoints/full-yolov2-coco.weights")

    import cv2

    # read, preprocess
    org_img = cv2.imread("../img/dog.jpg")
    img = cv2.resize(org_img, (416, 416))
    img = img[:, :, ::-1]
    img = img / 255.
    img = np.expand_dims(img, axis=0)

    out = model.predict(img)

    # postprocess
    anchors = np.array([0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828])
    anchors = np.reshape(anchors, [5, 2])

    threshold = 0.5
    iou_threshold = 0.4
    out = out[0]
    h, w = out.shape[0:2]
    out = np.reshape(out, [h, w, len(anchors), -1])
    bboxes = []
    for cy in range(h):
        for cx in range(w):
            for b in range(len(anchors)):
                prob_obj = sigmoid(out[cy, cx, b, 4])
                prob_classes = softmax(out[cy, cx, b, 5:])
                class_idx = np.argmax(prob_classes)
                class_prob = prob_classes[class_idx]
                p = prob_obj * class_prob
                if p < threshold:
                    continue
                coords = out[cy, cx, b, 0:4]
                bbox = BoundingBox()
                bbox.x = (sigmoid(coords[0]) + cx) / w
                bbox.y = (sigmoid(coords[1]) + cy) / h
                bbox.w = (anchors[b][0] * np.exp(coords[2])) / w
                bbox.h = (anchors[b][1] * np.exp(coords[3])) / h
                bbox.class_idx = class_idx
                bbox.prob = p
                bboxes.append(bbox)

    bboxes.sort(key=lambda box: box.prob, reverse=True)
    new_boxes = [bboxes[0]]
    for i in range(1, len(bboxes)):
        overlapping = False
        for new_box in new_boxes:
            if iou_score(new_box, bboxes[i]) >= iou_threshold:
                overlapping = True
                break
        if not overlapping:
            new_boxes.append(bboxes[i])
    print(new_boxes)
    h, w = org_img.shape[0:2]
    for box in new_boxes:
        tl = np.maximum(box.get_top_left(h, w), 0)
        br = np.maximum(box.get_bottom_right(h, w), 0)
        tl = (int(tl[0]), int(tl[1]))
        br = (int(br[0]), int(br[1]))
        org_img = cv2.rectangle(org_img, tl, br, (0, 0, 255), thickness=3)
        print(box.prob)
    cv2.imshow("img", org_img)
    cv2.waitKey(0)
    # print(out)

    # plot_model(model, "out.png", show_shapes=True)
