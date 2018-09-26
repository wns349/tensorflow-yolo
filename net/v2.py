import random

import numpy as np
import tensorflow as tf

from . import base
from .layers import conv2d_bn_act, input_layer, max_pool2d, route, reorg


@staticmethod
def create_full_network(anchors, class_names, is_training, scope="yolo", input_shape=(416, 416, 3)):
    num_anchors = len(anchors)
    num_classes = len(class_names)
    conv2d_bn_act.reset()
    tf.reset_default_graph()
    layers = []
    with tf.variable_scope(scope):
        layers.append(input_layer([None, input_shape[0], input_shape[1], input_shape[2]], "input"))

        for filter_size in (32, 64):
            layers.append(conv2d_bn_act(layers[-1].out, filter_size, 3, stride=1, is_training=is_training, scope=scope))
            layers.append(max_pool2d(layers[-1].out, 2, stride=2))

        for filter_size in (128, 256):
            layers.append(conv2d_bn_act(layers[-1].out, filter_size, 3, stride=1, is_training=is_training, scope=scope))
            layers.append(
                conv2d_bn_act(layers[-1].out, filter_size // 2, 1, stride=1, is_training=is_training, scope=scope))
            layers.append(conv2d_bn_act(layers[-1].out, filter_size, 3, stride=1, is_training=is_training, scope=scope))
            layers.append(max_pool2d(layers[-1].out, 2, stride=2))

        layers.append(conv2d_bn_act(layers[-1].out, 512, 3, stride=1, is_training=is_training, scope=scope))
        layers.append(conv2d_bn_act(layers[-1].out, 256, 1, stride=1, is_training=is_training, scope=scope))
        layers.append(conv2d_bn_act(layers[-1].out, 512, 3, stride=1, is_training=is_training, scope=scope))
        layers.append(conv2d_bn_act(layers[-1].out, 256, 1, stride=1, is_training=is_training, scope=scope))
        layers.append(conv2d_bn_act(layers[-1].out, 512, 3, stride=1, is_training=is_training, scope=scope))
        layers.append(max_pool2d(layers[-1].out, 2, stride=2))

        layers.append(conv2d_bn_act(layers[-1].out, 1024, 3, stride=1, is_training=is_training, scope=scope))
        layers.append(conv2d_bn_act(layers[-1].out, 512, 1, stride=1, is_training=is_training, scope=scope))
        layers.append(conv2d_bn_act(layers[-1].out, 1024, 3, stride=1, is_training=is_training, scope=scope))
        layers.append(conv2d_bn_act(layers[-1].out, 512, 1, stride=1, is_training=is_training, scope=scope))
        layers.append(conv2d_bn_act(layers[-1].out, 1024, 3, stride=1, is_training=is_training, scope=scope))

        layers.append(conv2d_bn_act(layers[-1].out, 1024, 3, stride=1, is_training=is_training, scope=scope))
        layers.append(conv2d_bn_act(layers[-1].out, 1024, 3, stride=1, is_training=is_training, scope=scope))
        layers.append(route([layers[-9].out]))
        layers.append(conv2d_bn_act(layers[-1].out, 64, 1, stride=1, is_training=is_training, scope=scope))
        layers.append(reorg(layers[-1].out, 2))
        layers.append(route([layers[-1].out, layers[-4].out]))
        layers.append(conv2d_bn_act(layers[-1].out, 1024, 3, stride=1, is_training=is_training, scope=scope))

        layers.append(conv2d_bn_act(layers[-1].out, num_anchors * (5 + num_classes), 1, 1,
                                    use_batch_normalization=False,
                                    activation_fn="linear",
                                    is_training=is_training,
                                    scope=scope))

        # rename last layer for convenience
        layers[-1].out = tf.identity(layers[-1].out, "output")
    return layers


@staticmethod
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
    print("Found {} weight values.".format(len(weights)))
    return base.load_weights(layers, weights)


@staticmethod
def find_bounding_boxes(net_out, net, threshold, iou_threshold, anchors, class_names):
    net_out = np.reshape(net_out,
                         [-1, net_out.shape[1], net_out.shape[2], len(anchors), (5 + len(class_names))])
    net_boxes = []
    for out in net_out:
        bounding_boxes = _find_bounding_boxes(out, anchors, threshold)
        net_boxes.append(base.non_maximum_suppression(bounding_boxes, iou_threshold))
    return net_boxes


def _find_bounding_boxes(out, anchors, threshold):
    h, w = out.shape[0:2]
    no_b = len(anchors)
    bboxes = []
    # TODO: maybe use matrix operation instead of for loops?
    for cy in range(h):
        for cw in range(w):
            for b in range(no_b):
                # calculate p(class|obj)
                prob_obj = base.sigmoid(out[cy, cw, b, 4])
                prob_classes = base.softmax(out[cy, cw, b, 5:])
                class_idx = np.argmax(prob_classes)
                class_prob = prob_classes[class_idx]
                p = prob_obj * class_prob
                if p < threshold:  # if lower than threshold, pass
                    continue

                coords = out[cy, cw, b, 0:4]
                bbox = base.BoundingBox()
                bbox.x = (base.sigmoid(coords[0]) + cw) / w
                bbox.y = (base.sigmoid(coords[1]) + cy) / h
                bbox.w = (anchors[b][0] * np.exp(coords[2])) / w
                bbox.h = (anchors[b][1] * np.exp(coords[3])) / h
                bbox.class_idx = class_idx
                bbox.prob = p
                bboxes.append(bbox)
    return bboxes


@staticmethod
def create_loss_fn(batch_size, net, anchors, class_names):
    net_out = net[-1].out
    h, w = net_out.get_shape().as_list()[1:3]
    b, c = len(anchors), len(class_names)

    cell_h = tf.tile(tf.range(h), [w])
    cell_w = tf.tile(tf.expand_dims(tf.range(w), 0), [h, 1])
    cell_w = tf.reshape(tf.transpose(cell_w), [-1])
    cell_hw = tf.stack([cell_h, cell_w], 1)
    cell_hw = tf.reshape(cell_hw, [-1, h, w, 1, 2])
    cell_hw = tf.tile(cell_hw, [1, 1, 1, b, 1])
    cell_hw = tf.to_float(cell_hw)

    anchor_tensor = np.reshape(anchors, [1, 1, 1, b, 2])

    pred = tf.reshape(net_out, [-1, h, w, b, 5 + c])
    pred_xy = tf.sigmoid(pred[..., 0:2]) + cell_hw
    pred_wh = tf.exp(pred[..., 2:4]) * anchor_tensor
    pred_obj = tf.sigmoid(pred[..., 4:5])
    pred_class = pred[..., 5:]

    # ground truth
    gt = tf.placeholder(dtype=np.float32, shape=[None, h, w, b, 5 + c])
    gt_ij = tf.placeholder(dtype=np.float32, shape=[None, h, w, b])
    gt_i = tf.placeholder(dtype=np.float32, shape=[None, h, w])
    placeholders = {
        "gt": gt, "gt_ij": gt_ij, "gt_i": gt_i
    }

    gt_xy = gt[..., 0:2]
    gt_wh = gt[..., 2:4]
    gt_obj = gt[..., 4:5]
    gt_class = tf.argmax(gt[..., 5:], axis=-1)

    mask_ij = tf.expand_dims(gt_ij, axis=-1)
    mask_i = tf.expand_dims(gt_i, axis=-1)

    loss_xy = 1. * tf.reduce_sum(mask_ij * tf.square(gt_xy - pred_xy)) / batch_size
    loss_wh = 1. * tf.reduce_sum(mask_ij * tf.square(tf.sqrt(gt_wh) - tf.sqrt(pred_wh))) / batch_size
    loss_obj = 5. * tf.reduce_sum(mask_ij * tf.square(gt_obj - pred_obj)) / batch_size
    loss_noobj = 1. * tf.reduce_sum((1 - mask_ij) * tf.square(gt_obj - pred_obj)) / batch_size
    loss_class = 1. * tf.reduce_sum(
        mask_i * tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt_class, logits=pred_class))

    loss = loss_xy + loss_wh + loss_obj + loss_noobj + loss_class

    with tf.name_scope("losses"):
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("loss_xy", loss_xy)
        tf.summary.scalar("loss_wh", loss_wh)
        tf.summary.scalar("loss_obj", loss_obj)
        tf.summary.scalar("loss_noobj", loss_noobj)
        tf.summary.scalar("loss_class", loss_class)

    return loss, placeholders


@staticmethod
def create_train_optimizer(loss_fn, learning_rate):
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        return tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_fn)


@staticmethod
def make_batch(net, annotations, batch_size, anchors, class_names, augment_prob):
    if len(annotations) == 0:
        yield [], {}
    elif len(annotations) < batch_size:
        batch_size = len(annotations)  # change batch_size

    total_batches = int(np.ceil(len(annotations) / batch_size))
    if len(annotations) % batch_size > 0:
        annotations.extend(annotations[0:batch_size - len(annotations) % batch_size])

    random.shuffle(annotations)

    input_h, input_w, input_c = net[0].out.get_shape().as_list()[1:4]
    for batch in range(total_batches):
        net_images = []
        net_placeholders = {}
        for i in range(batch_size):
            img_path, img_objects = annotations[batch * batch_size + i]
            net_image, objects = base.preprocess_image(img_path, (input_h, input_w, input_c), img_objects, augment_prob)
            net_placeholder = _make_ground_truths(net, objects, anchors, class_names)

            net_images.append(np.expand_dims(net_image, axis=0))
            for k in net_placeholder:
                new_value = net_placeholder[k]
                if k in net_placeholders:
                    old_value = net_placeholders.get(k)
                else:
                    old_value = np.zeros((0,) + new_value.shape)
                net_placeholders[k] = np.concatenate([old_value, [new_value]])

        yield np.concatenate(net_images, axis=0), net_placeholders


def _make_ground_truths(net, objects, anchors, class_names):
    input_h, input_w = net[0].out.get_shape().as_list()[1:3]
    output_h, output_w = net[-1].out.get_shape().as_list()[1:3]
    gts = np.zeros(shape=[output_h, output_w, len(anchors), 5 + len(class_names)])
    gt_ij = np.zeros(shape=[output_h, output_w, len(anchors)])
    gt_i = np.zeros(shape=[output_h, output_w])

    boxes = []
    for obj in objects:
        # xmin, ymin, xmax, ymax, name
        bx = (obj[0] + obj[2]) * .5 / input_w * output_w
        by = (obj[1] + obj[3]) * .5 / input_h * output_h
        bw = (obj[2] - obj[0]) / input_w * output_w
        bh = (obj[3] - obj[1]) / input_h * output_h
        cx = int(np.floor(bx))
        cy = int(np.floor(by))

        boxes.append(base.BoundingBox(bx, by, bw, bh, cx, cy, class_names.index(obj[4])))

    best_boxes = {}
    for box in boxes:
        key = "{}_{}".format(box.cx, box.cy)
        if key in best_boxes:
            best_box, best_iou, best_anchor_idx = best_boxes[key]
        else:
            best_box, best_iou, best_anchor_idx = None, -1, -1

        _box = base.BoundingBox(output_w / 2., output_h / 2., box.w, box.h)
        for idx, anchor in enumerate(anchors):
            _anchor_box = base.BoundingBox(output_w / 2., output_h / 2., anchor[0], anchor[1])
            iou = base.iou_score(_box, _anchor_box)
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

        gt_ij[best_box.cy, best_box.cx, best_anchor_idx] = 1
        gt_i[best_box.cy, best_box.cx] = 1

    return {
        "gt": gts,
        "gt_ij": gt_ij,
        "gt_i": gt_i
    }


@staticmethod
def generate_anchors(params):
    num_anchors = int(params["num_anchors"])
    image_dir = params["image_dir"]
    annotation_dir = params["annotation_dir"]
    tolerate = float(params["tolerate"])
    stride = int(params["stride"])
    input_w = int(params["input_w"])
    input_h = int(params["input_h"])

    annotations = base.parse_annotations(annotation_dir, image_dir, normalize=True)
    print("{} annotations found.".format(len(annotations)))
    class_names = set()
    data = []
    for annotation in annotations:
        obj = annotation[1]
        for o in obj:
            w = float(o[2] - o[0])
            h = float(o[3] - o[1])
            data.append([w, h])
            class_names.add(o[-1])
    anchors = base.run_kmeans(data, num_anchors, tolerate)
    anchors = [[a[0] * input_w / stride, a[1] * input_h / stride] for a in anchors]
    anchors = np.reshape(anchors, [-1])

    return anchors, class_names
