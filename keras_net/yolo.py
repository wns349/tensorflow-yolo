from tensorflow.python.keras.layers import Input
import numpy as np
import random
from keras_net import darknet_yolov2
from tensorflow.python.keras.optimizers import Adam
import os
import tensorflow as tf
from tqdm import tqdm
import xml.etree.ElementTree as ET
import cv2
from imgaug import augmenters as iaa
import imgaug as ia

# for img aug
seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.GaussianBlur((0, 3.0)),
    iaa.Dropout(0.02),
    iaa.AdditiveGaussianNoise(scale=0.01 * 255),
    iaa.AdditiveGaussianNoise(loc=32, scale=0.0001 * 255),
    iaa.Affine(translate_px={"x": (-40, 40)})
])


def augment_image(image, objects, new_shape):
    seq_det = seq.to_deterministic()

    bb_objects = [ia.BoundingBox(x1=o[0], y1=o[1], x2=o[2], y2=o[3], label=o[4]) for o in objects]
    bbs = ia.BoundingBoxesOnImage(bb_objects, shape=new_shape)

    image_aug = seq_det.augment_images([image])[0]
    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]

    new_objects = [(o.x1, o.y1, o.x2, o.y2, o.label) for o in
                   # bbs_aug.bounding_boxes]
                   bbs_aug.remove_out_of_image().cut_out_of_image().bounding_boxes]
    return image_aug, new_objects


def preprocess_image(image_path, new_shape, objects=None, augment_prob=0.):
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to read {}".format(image_path))
        return

    net_image = cv2.resize(image, new_shape[0:2])
    net_image = net_image[:, :, ::-1]  # BGR to RGB

    new_objects = None
    if objects is not None:
        new_objects = []
        for obj in objects:
            xmin = obj[0] / image.shape[1] * new_shape[1]
            ymin = obj[1] / image.shape[0] * new_shape[0]
            xmax = obj[2] / image.shape[1] * new_shape[1]
            ymax = obj[3] / image.shape[0] * new_shape[0]
            new_objects.append((xmin, ymin, xmax, ymax, obj[4]))

        if np.random.uniform() < augment_prob:
            net_image, new_objects = augment_image(net_image, new_objects, new_shape)

    net_image = net_image / 255.

    return net_image, new_objects


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


def test_model(model, img_path, anchors):
    def sigmoid(x):
        return 1. / (1. + np.exp(-x))

    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    # read, preprocess
    org_img = cv2.imread(img_path)
    img = cv2.resize(org_img, (416, 416))
    img = img[:, :, ::-1]
    img = img / 255.
    img = np.expand_dims(img, axis=0)

    out = model.predict(img)

    # postprocess

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


def make_ground_truth(input_h, input_w, output_h, output_w, anchors, class_names, objects):
    gts = np.zeros(shape=[output_h, output_w, len(anchors), 5 + len(class_names)])

    boxes = []
    for obj in objects:
        # xmin, ymin, xmax, ymax, name
        bx = (obj[0] + obj[2]) * .5 / input_w * output_w
        by = (obj[1] + obj[3]) * .5 / input_h * output_h
        bw = (obj[2] - obj[0]) / input_w * output_w
        bh = (obj[3] - obj[1]) / input_h * output_h
        cx = int(np.floor(bx))
        cy = int(np.floor(by))

        boxes.append(BoundingBox(bx, by, bw, bh, cx, cy, class_names.index(obj[4])))

    best_boxes = {}
    for box in boxes:
        key = "{}_{}".format(box.cx, box.cy)
        if key in best_boxes:
            best_box, best_iou, best_anchor_idx = best_boxes[key]
        else:
            best_box, best_iou, best_anchor_idx = None, -1, -1

        _box = BoundingBox(output_w / 2., output_h / 2., box.w, box.h)
        for idx, anchor in enumerate(anchors):
            _anchor_box = BoundingBox(output_w / 2., output_h / 2., anchor[0], anchor[1])
            iou = iou_score(_box, _anchor_box)
            if best_iou < iou:
                best_iou = iou
                best_box = box
                best_anchor_idx = idx
        best_boxes[key] = (best_box, best_iou, best_anchor_idx)

    for b in best_boxes:
        best_box, best_iou, best_anchor_idx = best_boxes[b]
        gts[best_box.cy, best_box.cx, best_anchor_idx, 0] = best_box.x
        gts[best_box.cy, best_box.cx, best_anchor_idx, 1] = best_box.y
        gts[best_box.cy, best_box.cx, best_anchor_idx, 2] = best_box.w
        gts[best_box.cy, best_box.cx, best_anchor_idx, 3] = best_box.h
        gts[best_box.cy, best_box.cx, best_anchor_idx, 4] = 1.
        gts[best_box.cy, best_box.cx, best_anchor_idx, 5 + best_box.class_idx] = 1

    return np.reshape(gts, [output_h, output_w, len(anchors) * (5 + len(class_names))])


def batch_gen(input_tensor, output_tensor, annotations, batch_size, anchors, class_names, augment_prob):
    if len(annotations) == 0:
        yield [], {}

    random.shuffle(annotations)

    while True:
        input_h, input_w, input_c = input_tensor.get_shape().as_list()[1:4]
        output_h, output_w = output_tensor.get_shape().as_list()[1:3]
        net_images = []  # X
        net_objects = []  # Y
        for i in range(batch_size):
            img_path, img_objects = annotations[i]
            net_image, objects = preprocess_image(img_path, (input_h, input_w, input_c), img_objects, augment_prob)
            gt = make_ground_truth(input_h, input_w, output_h, output_w, anchors, class_names, objects)
            net_images.append(np.expand_dims(net_image, axis=0))
            net_objects.append(np.expand_dims(gt, axis=0))
        yield np.concatenate(net_images, axis=0), np.concatenate(net_objects, axis=0)


def parse_annotations(annotation_dir, image_dir, normalize=False):
    annotations = [os.path.join(os.path.abspath(annotation_dir), f) for f in os.listdir(annotation_dir)
                   if f.lower().endswith(".xml")]

    result = []
    for annotation in tqdm(annotations):
        root = ET.parse(annotation).getroot()
        img_path = os.path.join(image_dir, root.find("filename").text)

        size = root.find("size")
        w = int(size.find("width").text)
        h = int(size.find("height").text)

        img_objects = []
        objects = root.findall("object")
        for object in objects:
            name = object.find("name").text
            bndbox = object.find("bndbox")
            x1 = int(bndbox.find("xmin").text)
            y1 = int(bndbox.find("ymin").text)
            x2 = int(bndbox.find("xmax").text)
            y2 = int(bndbox.find("ymax").text)
            if normalize:
                x1, x2 = x1 / w, x2 / w
                y1, y2 = y1 / h, y2 / h
            img_objects.append((x1, y1, x2, y2, name))

        result.append((img_path, img_objects))
    return result


def main():
    anchors = np.array([0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828])
    anchors = np.reshape(anchors, [5, 2])
    class_names = ["tower"]
    batch_size = 4
    augment_prob = 0.3

    train_annotation_dir = "../resource/eiffel/train"
    train_image_dir = "../resource/eiffel/train"

    input_tensor = Input(shape=[416, 416, 3])
    model = darknet_yolov2.build_model(input_tensor,
                                       include_top=True,
                                       weights_path="./checkpoints/full-yolov2-coco-fe",
                                       num_anchors=len(anchors),
                                       num_classes=len(class_names))
    output_tensor = model.layers[-1].output

    train_annotations = parse_annotations(train_annotation_dir, train_image_dir)

    print(model.summary())

    generator = batch_gen(input_tensor, output_tensor, train_annotations, batch_size, anchors, class_names,
                          augment_prob)

    # optimizer = Adam(lr=1e-3)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss=darknet_yolov2.build_loss_fn(anchors, class_names))
    model.fit_generator(generator, steps_per_epoch=np.ceil(len(train_annotation_dir) / batch_size), epochs=10)

    # test_model(model, "../img/dog.jpg")


if __name__ == '__main__':
    tf.enable_eager_execution()
    main()
