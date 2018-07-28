import numpy as np
import tensorflow as tf


def load_weights(layers, weights, scope="yolo"):
    variables = {v.op.name: v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)}
    ops = []
    read = 0
    for layer in layers:
        for variable_name in layer.variable_names:
            var = variables[variable_name]
            tokens = var.name.split("/")
            size = np.prod(var.shape.as_list())
            shape = var.shape.as_list()
            if "weights" in tokens[-1]:
                shape = [shape[3], shape[2], shape[0], shape[1]]
            value = np.reshape(weights[read: read + size], shape)
            if "weights" in tokens[-1]:
                value = np.transpose(value, (2, 3, 1, 0))  # darknet format tf format
            ops.append(tf.assign(var, value, validate_shape=True))
            read += size

    print("Weights ready ({}/{} read)".format(read, len(weights)))
    if read != len(weights):
        print("(warning) read count and total count do not match. Possibly an incorrect weights file.")

    return ops


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


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


def _find_bounding_boxes(result, anchors, threshold, output_h, output_w):
    no_b = len(anchors)

    bboxes = []
    # TODO: maybe use matrix operation instead of for loops?
    for cy in range(output_h):
        for cw in range(output_w):
            for b in range(no_b):
                # calculate p(class|obj)
                prob_obj = sigmoid(result[cy, cw, b, 4])
                prob_classes = softmax(result[cy, cw, b, 5:])
                class_idx = np.argmax(prob_classes)
                class_prob = prob_classes[class_idx]
                p = prob_obj * class_prob
                if p < threshold:  # if lower than threshold, pass
                    continue

                coords = result[cy, cw, b, 0:4]
                bbox = BoundingBox()
                bbox.x = (sigmoid(coords[0]) + cw) / output_w
                bbox.y = (sigmoid(coords[1]) + cy) / output_h
                bbox.w = (anchors[2 * b] * np.exp(coords[2])) / output_w
                bbox.h = (anchors[2 * b + 1] * np.exp(coords[3])) / output_h
                bbox.class_idx = class_idx
                bbox.prob = p
                bboxes.append(bbox)
    return bboxes


class BoundingBox(object):
    def __init__(self, x=float(), y=float(), w=float(), h=float()):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.class_idx = -1
        self.prob = 0

    def get_top_left(self, h=1., w=1.):
        return np.maximum((self.x - self.w / 2.) * w, 0.), np.maximum((self.y - self.h / 2.) * h, 0.)

    def get_bottom_right(self, h=1., w=1.):
        return np.minimum((self.x + self.w / 2.) * w, w), np.minimum((self.y + self.h / 2.) * h, h)
