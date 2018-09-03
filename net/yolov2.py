import os
import random

import numpy as np
import tensorflow as tf

from . import yolo
from .layers import conv2d_bn_act, input_layer, max_pool2d, route, reorg


def _create_full_network(num_anchors, num_classes, is_training, scope="yolo", input_shape=(416, 416, 3)):
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


def _load_weights(layers, weights_path):
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
    if read != len(weights):
        print("(warning) read count and total count do not match. Possibly an incorrect weights file.")

    return ops


def _find_bounding_boxes(out, anchors, threshold):
    h, w = out.shape[0:2]
    no_b = len(anchors)
    bboxes = []
    # TODO: maybe use matrix operation instead of for loops?
    for cy in range(h):
        for cw in range(w):
            for b in range(no_b):
                # calculate p(class|obj)
                prob_obj = yolo.sigmoid(out[cy, cw, b, 4])
                prob_classes = yolo.softmax(out[cy, cw, b, 5:])
                class_idx = np.argmax(prob_classes)
                class_prob = prob_classes[class_idx]
                p = prob_obj * class_prob
                if p < threshold:  # if lower than threshold, pass
                    continue

                coords = out[cy, cw, b, 0:4]
                bbox = yolo.BoundingBox()
                bbox.x = (yolo.sigmoid(coords[0]) + cw) / w
                bbox.y = (yolo.sigmoid(coords[1]) + cy) / h
                bbox.w = (anchors[b][0] * np.exp(coords[2])) / w
                bbox.h = (anchors[b][1] * np.exp(coords[3])) / h
                bbox.class_idx = class_idx
                bbox.prob = p
                bboxes.append(bbox)
    return bboxes


def _create_loss_fn(batch_size, net, anchors, class_names):
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


def _create_train_optimizer(loss_fn, learning_rate):
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        return tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_fn)


def _make_batch(net, annotations, batch_size, anchors, class_names, augment_prob):
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
            net_image, objects = yolo.preprocess_image(img_path, (input_h, input_w, input_c), img_objects, augment_prob)
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

    stride_h = input_h / output_h
    stride_w = input_w / output_w

    boxes = []
    for obj in objects:
        # xmin, ymin, xmax, ymax, name
        bx = (obj[0] + obj[2]) * .5 / input_w * output_w
        by = (obj[1] + obj[3]) * .5 / input_h * output_h
        bw = (obj[2] - obj[0]) / input_w * output_w
        bh = (obj[3] - obj[1]) / input_h * output_h
        cx = int(bx // stride_w)
        cy = int(by // stride_h)

        boxes.append(yolo.BoundingBox(bx, by, bw, bh, cx, cy, class_names.index(obj[4])))

    best_boxes = {}
    for box in boxes:
        key = "{}_{}".format(box.cx, box.cy)
        if key in best_boxes:
            best_box, best_iou, best_anchor_idx = best_boxes[key]
        else:
            best_box, best_iou, best_anchor_idx = None, -1, -1

        _box = yolo.BoundingBox(0, 0, box.w, box.h)
        for idx, anchor in enumerate(anchors):
            _anchor_box = yolo.BoundingBox(0, 0, anchor[0], anchor[1])
            iou = yolo.iou_score(_box, _anchor_box)
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
        gts[best_box.cy, best_box.cx, best_anchor_idx, 4] = best_iou
        gts[best_box.cy, best_box.cx, best_anchor_idx, 5 + best_box.class_idx] = 1

        gt_ij[best_box.cy, best_box.cx, best_anchor_idx] = 1
        gt_i[best_box.cy, best_box.cx] = 1

    return {
        "gt": gts,
        "gt_ij": gt_ij,
        "gt_i": gt_i
    }


def generate_anchors(params):
    num_anchors = int(params["num_anchors"])
    image_dir = params["image_dir"]
    annotation_dir = params["annotation_dir"]
    tolerate = float(params["tolerate"])
    stride = int(params["stride"])
    input_w = int(params["input_w"])
    input_h = int(params["input_h"])

    annotations = yolo.parse_annotations(annotation_dir, image_dir, normalize=True)
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
    anchors = yolo.run_kmeans(data, num_anchors, tolerate)
    anchors = [[a[0] * input_w / stride, a[1] * input_h / stride] for a in anchors]
    anchors = np.reshape(anchors, [-1])

    return anchors, class_names


def test(params):
    image_dir = params["image_dir"]
    out_dir = params["out_dir"]
    batch_size = int(params["batch_size"])
    threshold = float(params["threshold"])
    iou_threshold = float(params["iou_threshold"])
    anchors = np.reshape(params["anchors"], [-1, 2])
    class_names = params["class_names"]
    input_h = int(params["input_h"])
    input_w = int(params["input_w"])
    input_c = int(params["input_c"])
    checkpoint_path = params["checkpoint_path"]
    pretrained_weights_path = params["pretrained_weights_path"]

    # load images
    image_paths = yolo.load_image_paths(image_dir)
    if len(image_paths) == 0:
        print("No test images found in {}".format(image_dir))
        return

    # build network
    net = _create_full_network(len(anchors), len(class_names), False, input_shape=(input_h, input_w, input_c))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # load checkpoint
        saver = tf.train.Saver()
        if not yolo.load_checkpoint_by_path(saver, sess, checkpoint_path):
            # load pre-trained weights
            ops = _load_weights(net, pretrained_weights_path)
            sess.run(ops)
            print("Pre-trained weights loaded.")
        else:
            print("Checkpoint {} restored.".format(checkpoint_path))

        input_shape = tuple(net[0].out.get_shape().as_list()[1:4])
        test_batches = yolo.generate_test_batch(image_paths, batch_size, input_shape)
        for x_batch, paths in test_batches:  # run batch
            net_out = sess.run(net[-1].out, feed_dict={net[0].out: x_batch})

            # post-process
            net_boxes = []
            net_out = np.reshape(net_out,
                                 [-1, net_out.shape[1], net_out.shape[2], len(anchors), (5 + len(class_names))])
            for out in net_out:
                bounding_boxes = _find_bounding_boxes(out, anchors, threshold)
                net_boxes.append(yolo.non_maximum_suppression(bounding_boxes, iou_threshold))

            for boxes, path in zip(net_boxes, paths):
                # draw box on image
                new_img = yolo.draw_boxes(path, boxes, class_names)
                # write to file
                file_name, file_ext = os.path.splitext(os.path.basename(path))
                out_path = os.path.join(out_dir, "{}_out{}".format(file_name, file_ext))
                yolo.save_image(new_img, out_path)
                print("{}: Found {} objects. Saved to {}".format(file_name, len(boxes), out_path))
        print("Done")


def train(params):
    train_image_dir = params["image_dir"]
    train_annotation_dir = params["annotation_dir"]
    val_image_dir = params["val_image_dir"]
    val_annotation_dir = params["val_annotation_dir"]
    batch_size = int(params["batch_size"])
    learning_rate = float(params["learning_rate"])
    augment_prob = float(params["augment_probability"])
    checkpoint_prefix = params["checkpoint_prefix"]
    checkpoint_dir = params["checkpoint_dir"]
    checkpoint_step = int(params["checkpoint_step"])
    pretrained_weights_path = params["pretrained_weights_path"]
    tensorboard_log_dir = params["tensorboard_log_dir"]
    anchors = np.reshape(params["anchors"], [-1, 2])
    class_names = params["class_names"]
    input_h = int(params["input_h"])
    input_w = int(params["input_w"])
    input_c = int(params["input_c"])
    epochs = int(params["epochs"])
    max_step = int(params["max_step"])

    # prepare data
    train_annotations = yolo.parse_annotations(train_annotation_dir, train_image_dir)
    assert len(train_annotations) > 0
    val_annotations = yolo.parse_annotations(val_annotation_dir, val_image_dir)

    # build network
    net = _create_full_network(len(anchors), len(class_names), True, input_shape=(input_h, input_w, input_c))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # create loss function
        loss, placeholders = _create_loss_fn(batch_size, net, anchors, class_names)
        train_op = _create_train_optimizer(loss, learning_rate)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            tf_summary = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(os.path.join(tensorboard_log_dir, "train"), sess.graph)
            val_writer = tf.summary.FileWriter(os.path.join(tensorboard_log_dir, "validation"), sess.graph)
            sess.run(tf.global_variables_initializer())

            # load pre-trained weights/checkpoint
            step = yolo.load_checkpoint(saver, sess, checkpoint_dir, checkpoint_prefix)
            if step < 0:
                # load pre-trained weights
                ops = _load_weights(net, pretrained_weights_path)
                sess.run(ops)
                print("Pre-trained weights loaded.")
                step = 0
                yolo.save_checkpoint(saver, sess, checkpoint_dir, checkpoint_prefix, step)
            else:
                print("Checkpoint restored. step:{}".format(step))

            # make train batch
            train_loss_mva = None
            for epoch in range(1, epochs + 1):
                if 0 <= max_step < step:
                    break
                batches = _make_batch(net, train_annotations, batch_size, anchors, class_names, augment_prob)
                for feed_images, feed_gts in batches:
                    step += 1
                    if 0 <= max_step < step:
                        break
                    feed_dict = {placeholders[key]: feed_gts[key] for key in placeholders}
                    feed_dict[net[0].out] = feed_images
                    _, summary, train_loss = sess.run([train_op, tf_summary, loss], feed_dict=feed_dict)

                    train_writer.add_summary(summary, step)
                    train_writer.flush()
                    train_loss_mva = train_loss_mva * 0.9 + train_loss * 0.1 if train_loss_mva is not None else train_loss
                    print("step {} ({}/{}): {} (moving average: {})".format(step, epoch, epochs, train_loss,
                                                                            train_loss_mva))
                    if step > 0 and step % checkpoint_step == 0:
                        yolo.save_checkpoint(saver, sess, checkpoint_dir, checkpoint_prefix, step)

                        # validation
                        val_batches = _make_batch(net, val_annotations, batch_size, anchors, class_names, 0)
                        val_total = 0
                        val_count = 0
                        for val_feed_images, val_feed_gts in val_batches:
                            val_feed_dict = {placeholders[key]: val_feed_gts[key] for key in placeholders}
                            val_feed_dict[net[0].out] = val_feed_images
                            val_loss = sess.run(loss, feed_dict=val_feed_dict)
                            val_total += float(val_loss)
                            val_count += 1
                        val_loss = val_total / val_count
                        val_summary = tf.Summary(value=[
                            tf.Summary.Value(tag="loss", simple_value=val_loss),
                        ])
                        val_writer.add_summary(val_summary)
                        val_writer.flush()
                        print("validation loss: {}".format(val_loss))
                print("Epoch ({}/{}) completed.".format(epoch, epochs + 1))
    print("Done")
