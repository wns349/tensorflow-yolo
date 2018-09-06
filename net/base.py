import os
import xml.etree.ElementTree as ET

import cv2
import imgaug as ia
import numpy as np
import sklearn.cluster
import tensorflow as tf
from imgaug import augmenters as iaa
from tqdm import tqdm

COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255)]

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


def load_weights(layers, weights):
    variables = {v.op.name: v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="yolo")}
    ops = []
    read = 0
    for layer in layers:
        for variable_name in layer.variable_names:
            var = variables[variable_name]
            tokens = var.name.split("/")
            size = np.prod(var.shape.as_list())
            shape = var.shape.as_list()
            if "kernel" in tokens[-1]:
                shape = [shape[3], shape[2], shape[0], shape[1]]
            value = np.reshape(weights[read: read + size], shape)
            if "kernel" in tokens[-1]:
                value = np.transpose(value, (2, 3, 1, 0))
            ops.append(tf.assign(var, value, validate_shape=True))
            read += size

    print("Weights ready ({}/{} read)".format(read, len(weights)))

    return ops


def run_kmeans(data, num_anchors, tolerate, verbose=False):
    km = sklearn.cluster.KMeans(n_clusters=num_anchors, tol=tolerate, verbose=verbose)
    km.fit(data)
    return km.cluster_centers_


def load_checkpoint_by_path(saver, sess, checkpoint_path):
    try:
        saver.restore(sess, checkpoint_path)
        return True
    except Exception as e:
        print("Failed to load {}: {}".format(checkpoint_path, str(e)))
        return False


def load_image_paths(path_to_img_dir):
    return [os.path.join(os.path.abspath(path_to_img_dir), f) for f in os.listdir(path_to_img_dir)
            if any(f.lower().endswith(ext) for ext in ["jpg", "bmp", "png", "gif"])]


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

    # For debug only
    """
    from copy import deepcopy
    tmp = deepcopy(net_image)
    for obj in new_objects:
        tl = int(obj[0]), int(obj[1])
        br = int(obj[2]), int(obj[3])
        tmp = np.ascontiguousarray(tmp[:, :, ::-1], dtype=np.uint8)
        print(tl, br, tmp.shape)
        print(tmp)
        cv2.rectangle(tmp, tl, br, (0, 255, 0), thickness=3)
        print(tl, br, tmp.shape)
    cv2.imshow("image", tmp)
    cv2.waitKey(0)
    """

    net_image = net_image / 255.

    return net_image, new_objects


def generate_test_batch(img_paths, batch_size, input_shape):
    total_batches = int(np.ceil(len(img_paths) / batch_size))

    for b in range(total_batches):
        images = []
        paths = []
        for i in range(min(batch_size, len(img_paths) - b * batch_size)):
            image, _ = preprocess_image(img_paths[b * batch_size + i], input_shape, augment_prob=0.)
            images.append(np.expand_dims(image, axis=0))
            paths.append(img_paths[b * batch_size + i])
        yield np.concatenate(images, axis=0), paths


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


def non_maximum_suppression(boxes, iou_threshold):
    if len(boxes) == 0:
        return []

    boxes.sort(key=lambda box: box.prob, reverse=True)  # sort by prob(confidence)
    new_boxes = [boxes[0]]  # add first element
    for i in range(1, len(boxes)):
        overlapping = False
        for new_box in new_boxes:
            if iou_score(new_box, boxes[i]) >= iou_threshold:
                overlapping = True
                break
        if not overlapping:
            new_boxes.append(boxes[i])
    return new_boxes


def draw_boxes(path_to_img, boxes, class_names):
    image = cv2.imread(path_to_img)
    assert image is not None
    h, w = image.shape[0:2]
    for box in boxes:
        tl = np.maximum(box.get_top_left(h, w), 0)
        br = np.maximum(box.get_bottom_right(h, w), 0)
        tl = (int(tl[0]), int(tl[1]))
        br = (int(br[0]), int(br[1]))

        cv2.rectangle(image, tl, br, COLORS[box.class_idx % len(COLORS)], thickness=3)
        class_name = class_names[box.class_idx]
        cv2.putText(image, "{} {:.3f}".format(class_name, box.prob), (tl[0], tl[1] - 10), \
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[box.class_idx % len(COLORS)], thickness=1)
    return image


def save_image(image, out_path):
    cv2.imwrite(out_path, image)


def load_checkpoint(saver, sess, checkpoint_dir, checkpoint_prefix):
    metas = [f for f in os.listdir(checkpoint_dir) if f.startswith(checkpoint_prefix) and f.endswith(".meta")]

    if len(metas) == 0:
        return -1

    try:
        metas.sort(reverse=True)
        checkpoint_path = os.path.join(checkpoint_dir, os.path.splitext(metas[0])[0])
        name = os.path.splitext(os.path.basename(checkpoint_path))[0]
        step = int(name.split("-")[1])
        saver.restore(sess, checkpoint_path)
        return step
    except ValueError as e:
        print("Failed to load checkpoint: {}".format(str(e)))
        return -1


def save_checkpoint(saver, sess, checkpoint_dir, checkpoint_prefix, step):
    out_path = os.path.join(checkpoint_dir, "{}-{}".format(checkpoint_prefix, step))
    saver.save(sess, out_path)
    print("Checkpoint saved to {}".format(out_path))


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
