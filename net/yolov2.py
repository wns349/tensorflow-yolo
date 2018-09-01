import os
import random

import numpy as np
import tensorflow as tf
import sklearn.cluster

from . import yolo
from .layers import conv2d_bn_act, input_layer, max_pool2d, route, reorg


def create_full_network(num_anchors, num_classes, is_training, scope="yolo", input_shape=(416, 416, 3)):
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


def find_bounding_boxes(out, anchors, threshold):
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


def run_kmeans(data, num_anchors, tolerate, verbose=False):
    km = sklearn.cluster.KMeans(n_clusters=num_anchors, tol=tolerate, verbose=verbose)
    km.fit(data)
    return km.cluster_centers_


class YoloV2(object):
    def __init__(self):
        super().__init__()

    def generate_anchors(self, params):
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
        anchors = run_kmeans(data, num_anchors, tolerate)
        anchors = [[a[0] * input_w / stride, a[1] * input_h / stride] for a in anchors]
        anchors = np.reshape(anchors, [-1])

        return anchors, class_names

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
            input_shape = tuple(self.net[0].out.get_shape().as_list()[1:3])

            test_batches = yolo.generate_test_batch(test_img_paths, batch_size, input_shape)
            for x_batch, paths in test_batches:
                net_out = sess.run(self.net[-1].out, feed_dict={self.net[0].out: x_batch})
                net_boxes = self.postprocess(net_out, threshold, iou_threshold)
                for boxes, path in zip(net_boxes, paths):
                    # draw box on image
                    new_img = yolo.draw_boxes(path, boxes, self.names)
                    # write to file
                    file_name, file_ext = os.path.splitext(os.path.basename(path))
                    out_path = os.path.join(out_dir, "{}_out{}".format(file_name, file_ext))
                    yolo.save_image(new_img, out_path)
                    print("{}: Found {} objects. Saved to {}".format(file_name, len(boxes), out_path))

    def postprocess(self, net_out, threshold, iou_threshold):
        results = []
        out_shape = net_out.shape
        net_out = np.reshape(net_out, [-1, out_shape[1], out_shape[2], len(self.anchors), (5 + len(self.names))])
        for out in net_out:
            bounding_boxes = find_bounding_boxes(out, self.anchors, threshold)
            results.append(yolo.non_maximum_suppression(bounding_boxes, iou_threshold))
        return results

    def train(self,
              train_image_dir,
              train_annotation_dir,
              val_image_dir,
              val_annotation_dir,
              batch_size,
              epochs,
              learning_rate,
              augment_probability,
              pretrained_weights_path,
              checkpoint_step,
              checkpoint_dir,
              tensorboard_log_dir):
        train_annotations = yolo.parse_annotations(train_annotation_dir, train_image_dir)
        assert len(train_annotations) > 0
        val_annotations = yolo.parse_annotations(val_annotation_dir, val_image_dir)

        loss, placeholders = self._create_loss_fn(batch_size)
        train_op = self._create_train_optimizer(loss, learning_rate)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            tf_summary = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(os.path.join(tensorboard_log_dir, "train"), sess.graph)
            val_writer = tf.summary.FileWriter(os.path.join(tensorboard_log_dir, "validation"), sess.graph)
            sess.run(tf.global_variables_initializer())

            # load pretrained weights/checkpoint
            step = yolo.load_checkpoint(saver, sess, checkpoint_dir)
            if step == 0 and pretrained_weights_path is not None:
                print("Running pretrained weights")
                weight_ops = load_weights(self.net, pretrained_weights_path)
                sess.run(weight_ops)

            # make train batch
            train_loss_mva = None
            for epoch in range(1, epochs + 1):
                batches = self._make_batch(train_annotations, batch_size)
                for feed_images, feed_gts in batches:
                    step += 1
                    feed_dict = {placeholders[key]: feed_gts[key] for key in placeholders}
                    feed_dict[self.net[0].out] = feed_images
                    _, summary, train_loss = sess.run([train_op, tf_summary, loss], feed_dict=feed_dict)

                    train_writer.add_summary(summary, step)
                    train_loss_mva = train_loss_mva * 0.9 + train_loss * 0.1 if train_loss_mva is not None else train_loss
                    print("step {} ({}/{}): {} (moving average: {})".format(step, epoch, epochs, train_loss,
                                                                            train_loss_mva))
                    if step > 0 and step % checkpoint_step == 0:
                        yolo.save_checkpoint(saver, sess, checkpoint_dir, step)

    def _make_ground_truths(self, objects):
        input_h, input_w = self.net[0].out.get_shape().as_list()[1:3]
        output_h, output_w = self.net[-1].out.get_shape().as_list()[1:3]
        gts = np.zeros(shape=[output_h, output_w, len(self.anchors), 5 + len(self.names)])
        gt_ij = np.zeros(shape=[output_h, output_w, len(self.anchors)])
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

            boxes.append(yolo.BoundingBox(bx, by, bw, bh, cx, cy, self.names.index(obj[4])))

        best_boxes = {}
        for box in boxes:
            key = "{}_{}".format(box.cx, box.cy)
            if key in best_boxes:
                best_box, best_iou, best_anchor_idx = best_boxes[key]
            else:
                best_box, best_iou, best_anchor_idx = None, -1, -1

            _box = yolo.BoundingBox(0, 0, box.w, box.h)
            for idx, anchor in enumerate(self.anchors):
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

    def _make_batch(self, annotations, batch_size):
        if len(annotations) == 0:
            yield [], {}
        elif len(annotations) < batch_size:
            batch_size = len(annotations)  # change batch_size
            # annotations = annotations * int(batch_size / len(annotations))  # repeat

        total_batches = int(np.ceil(len(annotations) / batch_size))
        if len(annotations) % batch_size > 0:
            annotations.extend(annotations[0:batch_size - len(annotations) % batch_size])

        random.shuffle(annotations)

        input_h, input_w = self.net[0].out.get_shape().as_list()[1:3]
        for batch in range(total_batches):
            net_images = []
            net_placeholders = {}
            for i in range(batch_size):
                img_path, img_objects = annotations[batch * batch_size + i]
                # TODO: augmentation?
                net_image, objects = yolo.preprocess_image(img_path, (input_h, input_w), img_objects)
                net_placeholder = self._make_ground_truths(objects)

                net_images.append(np.expand_dims(net_image, axis=0))
                for k in net_placeholder:
                    new_value = net_placeholder[k]
                    if k in net_placeholders:
                        old_value = net_placeholders.get(k)
                    else:
                        old_value = np.zeros((0,) + new_value.shape)
                    net_placeholders[k] = np.concatenate([old_value, [new_value]])

            yield np.concatenate(net_images, axis=0), net_placeholders

    def _create_train_optimizer(self, loss_fn, learning_rate):
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            return tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_fn)

    def _create_loss_fn(self, batch_size):
        net_out = self.net[-1].out
        h, w = net_out.get_shape().as_list()[1:3]

        cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(h), [w]), (1, h, w, 1, 1)))
        cell_y = tf.transpose(cell_x, (0, 2, 1, 3, 4))
        cell_xy = tf.tile(tf.concat([cell_x, cell_y], -1), [batch_size, 1, 1, len(self.anchors), 1])
        _anchors = np.reshape(self.anchors, [1, 1, 1, len(self.anchors), 2])

        pred = tf.reshape(net_out, [-1, h, w, len(self.anchors), 5 + len(self.names)])
        pred_xy = tf.sigmoid(pred[..., 0:2]) + cell_xy
        pred_wh = tf.exp(pred[..., 2:4]) * _anchors
        pred_obj = tf.expand_dims(tf.sigmoid(pred[..., 4]), axis=-1)
        pred_class = pred[..., 5:]

        # ground truth
        gt = tf.placeholder(dtype=np.float32, shape=[None, h, w, len(self.anchors), 5 + len(self.names)])
        gt_ij = tf.placeholder(dtype=np.float32, shape=[None, h, w, len(self.anchors)])
        gt_i = tf.placeholder(dtype=np.float32, shape=[None, h, w])
        placeholders = {
            "gt": gt, "gt_ij": gt_ij, "gt_i": gt_i
        }

        gt_xy = gt[..., 0:2]
        gt_wh = gt[..., 2:4]
        gt_obj = tf.expand_dims(gt[..., 4], axis=-1)
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


if __name__ == "__main__":
    o = YoloV2()
    o.initialize({
        "anchors": [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828],
        "names": ["tower"]
    }, True)
    o.train("./resource/eiffel/train/", "./resource/eiffel/train/",
            "./resource/eiffel/val/", "./resource/eiffel/val/",
            2, 1000, 1e-5, 0.3, "./bin/yolov2.weights", 5, "./out/", "./log/")

    print("Done")
