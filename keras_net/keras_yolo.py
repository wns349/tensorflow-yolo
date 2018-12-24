import ast
import os
import random
import xml.etree.ElementTree as ET

import cv2
import imgaug as ia
import numpy as np
from imgaug import augmenters as iaa
from tensorflow.python import keras

from keras_net import darknet_yolov2

layers = keras.layers
optimizers = keras.optimizers

# for img aug
seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.GaussianBlur((0, 3.0)),
    iaa.Dropout(0.02),
    iaa.AdditiveGaussianNoise(scale=0.01 * 255),
    iaa.AdditiveGaussianNoise(loc=32, scale=0.0001 * 255),
    iaa.Affine(translate_px={'x': (-40, 40)})
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
        print('Failed to read {}'.format(image_path))
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


class DataGenerator(keras.utils.Sequence):
    def __init__(self,
                 image_dir,
                 annotation_dir,
                 batch_size,
                 input_shape,
                 output_shape,
                 anchors,
                 class_names,
                 augment_probability=0.3,
                 normalize=False):
        super(DataGenerator, self).__init__()

        self.annotations = [os.path.join(os.path.abspath(annotation_dir), f) for f in os.listdir(annotation_dir)
                            if f.lower().endswith('.xml')]
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.normalize = normalize
        self.augment_probability = augment_probability
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.anchors = anchors
        self.class_names = class_names

        random.shuffle(self.annotations)

    def __getitem__(self, idx):
        minibatch = self.annotations[idx * self.batch_size: min((idx + 1) * self.batch_size, len(self.annotations))]

        x = []
        y = []
        for annotation in minibatch:
            img_path, objects = self._parse_annotation(annotation)
            image, objects = preprocess_image(img_path, self.input_shape, objects, self.augment_probability)
            gt = self._make_ground_truth(objects)
            x.append(image)
            y.append(gt)

        return np.array(x), np.array(y)

    def __len__(self):
        return int(np.ceil(len(self.annotations) / self.batch_size))

    def _parse_annotation(self, annotation):
        root = ET.parse(annotation).getroot()
        img_path = os.path.join(self.image_dir, root.find('filename').text)

        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        img_objects = []
        objects = root.findall('object')
        for obj in objects:
            name = obj.find('name').text
            bndbox = obj.find('bndbox')
            x1 = int(bndbox.find('xmin').text)
            y1 = int(bndbox.find('ymin').text)
            x2 = int(bndbox.find('xmax').text)
            y2 = int(bndbox.find('ymax').text)
            if self.normalize:
                x1, x2 = x1 / w, x2 / w
                y1, y2 = y1 / h, y2 / h
            img_objects.append((x1, y1, x2, y2, name))
        return img_path, img_objects

    def _make_ground_truth(self, objects):
        input_h, input_w = self.input_shape
        output_h, output_w = self.output_shape
        anchors = self.anchors
        class_names = self.class_names

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
            key = '{}_{}'.format(box.cx, box.cy)
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


def train_model(model,
                anchors,
                class_names,
                epochs,
                learning_rate,
                batch_size,
                training_image_dir,
                training_annotation_dir,
                validation_image_dir=None,
                validation_annotation_dir=None,
                augment_probability=0.2,
                weights_dir='./out'):
    input_shape = model.input_shape
    output_shape = model.output_shape

    t_gen = DataGenerator(training_image_dir,
                          training_annotation_dir,
                          batch_size,
                          input_shape[1:3],
                          output_shape[1:3],
                          anchors,
                          class_names,
                          augment_probability=augment_probability)
    v_gen = DataGenerator(validation_image_dir,
                          validation_annotation_dir,
                          batch_size,
                          input_shape[1:3],
                          output_shape[1:3],
                          anchors,
                          class_names,
                          augment_probability=0.0)

    ckpt_file = os.path.join(weights_dir, 'ckpt.h5')
    if os.path.exists(ckpt_file):
        model.load_weights(ckpt_file, by_name=True)
        print('{} loaded.'.format(ckpt_file))

    cbs = [keras.callbacks.ModelCheckpoint(os.path.join(weights_dir, 'ckpt.h5'),
                                           save_best_only=True,
                                           save_weights_only=True),
           keras.callbacks.TerminateOnNaN()]

    optimizer = optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer,
                  loss=darknet_yolov2.build_loss_fn(anchors, class_names))

    model.fit_generator(generator=t_gen,
                        validation_data=v_gen,
                        epochs=epochs,
                        callbacks=cbs)


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
    cv2.imshow('img', org_img)
    cv2.waitKey(0)


def main(args):
    anchors = np.array([0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828])
    anchors = np.reshape(anchors, [5, 2])

    if args['mode'] == 'test':
        class_names = [str(i) for i in range(80)]
        input_tensor = layers.Input(shape=[416, 416, 3])
        model = darknet_yolov2.build_model(input_tensor,
                                           include_top=True,
                                           weights_path='./checkpoints/full-yolov2-coco',
                                           num_anchors=len(anchors),
                                           num_classes=len(class_names),
                                           trainable=False)
        print(model.summary())
        test_model(model, '../img/dog.jpg', anchors)
    elif args['mode'] == 'train':
        class_names = ['tower']
        input_tensor = layers.Input(shape=[416, 416, 3])
        model = darknet_yolov2.build_model(input_tensor,
                                           include_top=True,
                                           weights_path='./checkpoints/full-yolov2-coco-fe',
                                           num_anchors=len(anchors),
                                           num_classes=len(class_names),
                                           trainable=True)
        print(model.summary())
        train_model(model, anchors, class_names,
                    epochs=10,
                    learning_rate=1e-4,
                    batch_size=8,
                    training_image_dir='../resource/eiffel/train',
                    training_annotation_dir='../resource/eiffel/train',
                    validation_image_dir='../resource/eiffel/val',
                    validation_annotation_dir='../resource/eiffel/val',
                    augment_probability=0.3)
    else:
        print('Huh?')


def _update_configs(configs, configs_path):
    for k, v in configs.items():
        if k.endswith('_dir') or k.endswith('_path'):
            if not os.path.isabs(v):
                configs[k] = os.path.join(os.path.dirname(os.path.abspath(configs_path)), v)
        if k == 'anchors' or k == 'class_names':
            configs[k] = ast.literal_eval(v)
    return configs


if __name__ == '__main__':
    import argparse
    import configparser

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', dest='mode', help='Either train or test', default='test')
    args = parser.parse_args()
    c = vars(args)
    c['mode'] = 'train'
    main(c)
