import argparse
import os

import cv2
import numpy as np
import tensorflow as tf

from layers import conv2d, max_pool2d, leaky_relu, input_layer

parser = argparse.ArgumentParser()
parser.add_argument("--img", dest="img", help="Path to a test image")

#####
# TODO: extract to configs
VOC_ANCHORS = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]
VOC_LABELS = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
              "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
VOC_WEIGHTS = "./bin/yolov2-tiny-voc.weights"

COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255)]


#####

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


class yolo_tiny(object):
    def __init__(self, anchors, labels, threshold=0.5, iou_threshold=0.5):
        self.anchors = anchors
        self.labels = labels
        self.no_b = int(len(self.anchors) / 2.)
        self.no_c = len(self.labels)
        self.threshold = threshold
        self.iou_threshold = iou_threshold

        self.layers = self.create_network()

        self.input = self.layers[0].out
        self.input_h, self.input_w = self.input.shape.as_list()[1:3]
        self.output = self.layers[-1].out
        self.output_h, self.output_w = self.output.shape.as_list()[1:3]

    def load_weights(self, tf_session, weights_path):
        print("Reading pre-trained weights from {}".format(weights_path))
        file_size = os.path.getsize(weights_path)
        versions = np.memmap(weights_path, shape=4, offset=0, dtype=np.int)
        print("Weights versions: ", versions)

        offset = 16  # versions contains 4 int

        for layer in self.layers:
            if type(layer) == conv2d:
                offset += layer.load_weights(tf_session, offset, weights_path)
        print("Weights loaded. ({}/{} read)".format(offset, file_size))
        if offset != file_size:
            print("(warning) Offset and file size do not match. Possibly an incorrect weights file.")

    def create_network(self):
        _input = input_layer([None, 416, 416, 3], name="input")

        conv1 = conv2d(_input, 3, 3, 16, 1, batch_normalize=True)
        act1 = leaky_relu(conv1)
        pool1 = max_pool2d(act1, 2, 2, padding="VALID")

        conv2 = conv2d(pool1, 3, 3, 32, 1, batch_normalize=True)
        act2 = leaky_relu(conv2)
        pool2 = max_pool2d(act2, 2, 2, padding="VALID")

        conv3 = conv2d(pool2, 3, 3, 64, 1, batch_normalize=True)
        act3 = leaky_relu(conv3)
        pool3 = max_pool2d(act3, 2, 2, padding="VALID")

        conv4 = conv2d(pool3, 3, 3, 128, 1, batch_normalize=True)
        act4 = leaky_relu(conv4)
        pool4 = max_pool2d(act4, 2, 2, padding="VALID")

        conv5 = conv2d(pool4, 3, 3, 256, 1, batch_normalize=True)
        act5 = leaky_relu(conv5)
        pool5 = max_pool2d(act5, 2, 2, padding="VALID")

        conv6 = conv2d(pool5, 3, 3, 512, 1, batch_normalize=True)
        act6 = leaky_relu(conv6)
        pool6 = max_pool2d(act6, 2, 1, padding="SAME")

        conv7 = conv2d(pool6, 3, 3, 1024, 1, batch_normalize=True)
        act7 = leaky_relu(conv7)
        conv8 = conv2d(act7, 3, 3, 1024, 1, batch_normalize=True)
        act8 = leaky_relu(conv8)
        conv9 = conv2d(act8, 1, 1, self.no_b * (5 + self.no_c), 1, batch_normalize=False)
        conv9.out = tf.identity(conv9.out, "output")

        layers = [_input, conv1, act1, pool1, conv2, act2, pool2, conv3, act3, pool3, conv4, act4, pool4, conv5, act5,
                  pool5, conv6, act6, pool6, conv7, act7, conv8, act8, conv9]

        layers[-1].out = tf.identity(layers[-1].out, name="output")
        return layers

    def preprocess(self, img):
        imsz = cv2.resize(img, (self.input_h, self.input_w))
        imsz = imsz / 255.  # to make values lie between 0 and 1
        imsz = imsz[:, :, ::-1]  # BGR to RGB
        return imsz

    def postprocess(self, result):
        bboxes = self.find_bounding_boxes(result)
        return self.non_maximum_suppression(bboxes)

    def non_maximum_suppression(self, bboxes):
        if len(bboxes) == 0:
            return []

        bboxes.sort(key=lambda box: box.prob, reverse=True)  # sort by prob
        new_boxes = [bboxes[0]]  # add first element
        for i in range(1, len(bboxes)):
            overlapping = False
            for new_box in new_boxes:
                if iou_score(new_box, bboxes[i]) >= self.iou_threshold:
                    overlapping = True
                    break
            if not overlapping:
                new_boxes.append(bboxes[i])
        return new_boxes

    def find_bounding_boxes(self, result):
        bboxes = []
        for cy in range(self.output_h):
            for cw in range(self.output_w):
                for b in range(self.no_b):
                    # calculate p(class|obj)
                    prob_obj = sigmoid(result[cy, cw, b, 4])
                    prob_classes = softmax(result[cy, cw, b, 5:])
                    class_idx = np.argmax(prob_classes)
                    class_prob = prob_classes[class_idx]
                    p = prob_obj * class_prob
                    if p < self.threshold:  # if lower, pass
                        continue

                    coords = result[cy, cw, b, 0:4]
                    bbox = bounding_box()
                    bbox.x = (sigmoid(coords[0]) + cw) / self.output_w
                    bbox.y = (sigmoid(coords[1]) + cy) / self.output_h
                    bbox.w = (self.anchors[2 * b] * np.exp(coords[2])) / self.output_w
                    bbox.h = (self.anchors[2 * b + 1] * np.exp(coords[3])) / self.output_h
                    bbox.class_idx = class_idx
                    bbox.prob = p
                    bboxes.append(bbox)
        return bboxes

    def predict(self, tf_session, img):
        net_img = self.preprocess(img)
        net_img = np.expand_dims(net_img, axis=0)  # since there is only a single image to predict

        results = tf_session.run(self.output, feed_dict={self.input: net_img})
        results = np.reshape(results, [-1, self.output_h, self.output_w, self.no_b, 5 + self.no_c])
        result = results[0]  # single image prediction only

        bboxes = self.postprocess(result)  # find bounding boxes

        org_img_h, org_img_w = img.shape[0:2]
        for bbox in bboxes:
            top_left = bbox.get_top_left(org_img_h, org_img_w)
            bottom_right = bbox.get_bottom_right(org_img_h, org_img_w)
            top_left = (int(top_left[0]), int(top_left[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
            cv2.rectangle(img, top_left, bottom_right, COLORS[bbox.class_idx % len(COLORS)], thickness=3)
            label = self.labels[bbox.class_idx]
            cv2.putText(img, "{} {:.3f}".format(label, bbox.prob), (top_left[0], top_left[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[bbox.class_idx % len(COLORS)], thickness=1)
        return img


class bounding_box(object):
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


def _main(args):
    print("Hello")

    with tf.Session() as sess:
        yolo = yolo_tiny(VOC_ANCHORS, VOC_LABELS)
        sess.run(tf.global_variables_initializer())

        # load pre-trained weights
        yolo.load_weights(sess, VOC_WEIGHTS)

        print("Test image: {}".format(args.img))
        if args.img is None or (not os.path.exists(args.img)):
            print("File does not exist.")
            return

        # predict
        org_img = cv2.imread(args.img)
        img = yolo.predict(sess, org_img)
        cv2.imwrite("./out.jpg", img)

    print("Done")


if __name__ == "__main__":
    _main(parser.parse_args())
