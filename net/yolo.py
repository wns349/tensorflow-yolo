import os
import xml.etree.ElementTree as ET

import cv2
import imgaug as ia
import numpy as np
import sklearn.cluster
from imgaug import augmenters as iaa
from tqdm import tqdm

COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255)]

# for img aug
sometimes = lambda aug: iaa.Sometimes(0.5, aug)
seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5),  # horizontally flip 50% of all images
        iaa.Flipud(0.2),  # vertically flip 20% of all images
        # crop images by -5% to 10% of their height/width
        sometimes(iaa.CropAndPad(
            percent=(-0.05, 0.1),
            pad_mode=ia.ALL,
            pad_cval=(0, 255)
        )),
        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent (per axis)
            rotate=(-45, 45),  # rotate by -45 to +45 degrees
            shear=(-16, 16),  # shear by -16 to +16 degrees
            order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 5),
                   [
                       sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
                       # convert images into their superpixel representation
                       iaa.OneOf([
                           iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                           iaa.AverageBlur(k=(2, 7)),  # blur image using local means with kernel sizes between 2 and 7
                           iaa.MedianBlur(k=(3, 11)),
                           # blur image using local medians with kernel sizes between 2 and 7
                       ]),
                       iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                       iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
                       # search either for all edges or for directed edges,
                       # blend the result with the original image using a blobby mask
                       iaa.SimplexNoiseAlpha(iaa.OneOf([
                           iaa.EdgeDetect(alpha=(0.5, 1.0)),
                           iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                       ])),
                       iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                       # add gaussian noise to images
                       iaa.OneOf([
                           iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                           iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                       ]),
                       iaa.Invert(0.05, per_channel=True),  # invert color channels
                       iaa.Add((-10, 10), per_channel=0.5),
                       # change brightness of images (by -10 to 10 of original value)
                       iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
                       # either change the brightness of the whole image (sometimes
                       # per channel) or change the brightness of subareas
                       iaa.OneOf([
                           iaa.Multiply((0.5, 1.5), per_channel=0.5),
                           iaa.FrequencyNoiseAlpha(
                               exponent=(-4, 0),
                               first=iaa.Multiply((0.5, 1.5), per_channel=True),
                               second=iaa.ContrastNormalization((0.5, 2.0))
                           )
                       ]),
                       iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                       iaa.Grayscale(alpha=(0.0, 1.0)),
                       sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
                       # move pixels locally around (with random strengths)
                       sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),  # sometimes move parts of the image around
                       sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                   ],
                   random_order=True
                   )
    ],
    random_order=True
)


def run_kmeans(data, num_anchors, tolerate, verbose=False):
    km = sklearn.cluster.KMeans(n_clusters=num_anchors, tol=tolerate, verbose=verbose)
    km.fit(data)
    return km.cluster_centers_


def load_checkpoint_by_path(saver, sess, checkpoint_path):
    try:
        saver.restore(sess, checkpoint_path)
        return True
    except ValueError as e:
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
            xmin = obj[0] / image.shape[1] * new_shape[1]  # xmin
            ymin = obj[1] / image.shape[0] * new_shape[0]  # ymin
            xmax = obj[2] / image.shape[1] * new_shape[1]  # xmin
            ymax = obj[3] / image.shape[0] * new_shape[0]  # ymin
            new_objects.append((xmin, ymin, xmax, ymax, obj[4]))

        if np.random.uniform() <= augment_prob:
            net_image, new_objects = augment_image(net_image, new_objects, new_shape)

    return net_image, new_objects


def generate_test_batch(img_paths, batch_size, input_shape):
    total_batches = int(np.ceil(len(img_paths) / batch_size))

    for b in range(total_batches):
        images = []
        paths = []
        for i in range(min(batch_size, len(img_paths) - b * batch_size)):
            image, _ = preprocess_image(img_paths[b * batch_size + i], input_shape)
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
        tl = box.get_top_left(h, w)
        br = box.get_bottom_right(h, w)
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
        return np.maximum((self.x - self.w / 2.) * w, 0.), np.maximum((self.y - self.h / 2.) * h, 0.)

    def get_bottom_right(self, h=1., w=1.):
        return np.minimum((self.x + self.w / 2.) * w, w), np.minimum((self.y + self.h / 2.) * h, h)
