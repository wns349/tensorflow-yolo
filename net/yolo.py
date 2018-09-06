import abc
import os

import numpy as np
import tensorflow as tf

from . import base
from . import v2


class Yolo(object):
    @abc.abstractmethod
    def create_network(self, num_anchors, num_classes, is_training, scope="yolo", input_shape=(416, 416, 3)):
        raise NotImplementedError()

    @abc.abstractmethod
    def load_weights(self, layers, weights_path):
        raise NotImplementedError()

    @abc.abstractmethod
    def find_bounding_boxes(self, out, anchors, threshold):
        raise NotImplementedError()

    @abc.abstractmethod
    def generate_anchors(self, params):
        raise NotImplementedError()

    @abc.abstractmethod
    def create_loss_fn(self, batch_size, net, anchors, class_names):
        raise NotImplementedError()

    @abc.abstractmethod
    def make_batch(self, net, annotations, batch_size, anchors, class_names, augment_prob):
        raise NotImplementedError()

    @abc.abstractmethod
    def create_train_optimizer(self, loss_fn, learning_rate):
        raise NotImplementedError()

    def test(self, params):
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
        cpu_only = params["cpu_only"].lower() == "true"

        # load images
        image_paths = base.load_image_paths(image_dir)
        if len(image_paths) == 0:
            print("No test images found in {}".format(image_dir))
            return

        # build network
        net = self.create_network(len(anchors), len(class_names), False, input_shape=(input_h, input_w, input_c))

        if cpu_only:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # load checkpoint
            saver = tf.train.Saver()
            if not base.load_checkpoint_by_path(saver, sess, checkpoint_path):
                # load pre-trained weights
                ops = self.load_weights(net, pretrained_weights_path)
                sess.run(ops)
                print("Pre-trained weights loaded.")
            else:
                print("Checkpoint {} restored.".format(checkpoint_path))

            input_shape = tuple(net[0].out.get_shape().as_list()[1:4])
            test_batches = base.generate_test_batch(image_paths, batch_size, input_shape)
            for x_batch, paths in test_batches:  # run batch
                net_out = sess.run(net[-1].out, feed_dict={net[0].out: x_batch})

                # post-process
                net_boxes = []
                net_out = np.reshape(net_out,
                                     [-1, net_out.shape[1], net_out.shape[2], len(anchors), (5 + len(class_names))])
                for out in net_out:
                    bounding_boxes = self.find_bounding_boxes(out, anchors, threshold)
                    net_boxes.append(base.non_maximum_suppression(bounding_boxes, iou_threshold))

                for boxes, path in zip(net_boxes, paths):
                    # draw box on image
                    new_img = base.draw_boxes(path, boxes, class_names)
                    # write to file
                    file_name, file_ext = os.path.splitext(os.path.basename(path))
                    out_path = os.path.join(out_dir, "{}_out{}".format(file_name, file_ext))
                    base.save_image(new_img, out_path)
                    print("{}: Found {} objects. Saved to {}".format(file_name, len(boxes), out_path))
            print("Done")

    def train(self, params):
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
        cpu_only = params["cpu_only"].lower() == "true"

        # prepare data
        train_annotations = base.parse_annotations(train_annotation_dir, train_image_dir)
        assert len(train_annotations) > 0
        val_annotations = base.parse_annotations(val_annotation_dir, val_image_dir)

        # build network
        net = self.create_network(len(anchors), len(class_names), True, input_shape=(input_h, input_w, input_c))

        if cpu_only:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # create loss function
            loss, placeholders = self.create_loss_fn(batch_size, net, anchors, class_names)
            train_op = self.create_train_optimizer(loss, learning_rate)
            saver = tf.train.Saver()
            with tf.Session() as sess:
                tf_summary = tf.summary.merge_all()
                train_writer = tf.summary.FileWriter(os.path.join(tensorboard_log_dir, "train"), sess.graph)
                val_writer = tf.summary.FileWriter(os.path.join(tensorboard_log_dir, "validation"), sess.graph)
                sess.run(tf.global_variables_initializer())

                # load pre-trained weights/checkpoint
                step = base.load_checkpoint(saver, sess, checkpoint_dir, checkpoint_prefix)
                if step < 0:
                    # load pre-trained weights
                    ops = self.load_weights(net, pretrained_weights_path)
                    sess.run(ops)
                    print("Pre-trained weights loaded.")
                    step = 0
                    base.save_checkpoint(saver, sess, checkpoint_dir, checkpoint_prefix, step)
                else:
                    print("Checkpoint restored. step:{}".format(step))

                # make train batch
                train_loss_mva = None
                for epoch in range(1, epochs + 1):
                    if 0 <= max_step < step:
                        break
                    batches = self.make_batch(net, train_annotations, batch_size, anchors, class_names, augment_prob)
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
                            base.save_checkpoint(saver, sess, checkpoint_dir, checkpoint_prefix, step)

                            # validation
                            val_batches = self.make_batch(net, val_annotations, batch_size, anchors, class_names, 0)
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
                    print("Epoch ({}/{}) completed.".format(epoch, epochs))
        print("Done")


class YoloV2(Yolo):
    create_network = v2.create_full_network
    load_weights = v2.load_weights
    find_bounding_boxes = v2.find_bounding_boxes
    generate_anchors = v2.generate_anchors
    make_batch = v2.make_batch
    create_train_optimizer = v2.create_train_optimizer
    create_loss_fn = v2.create_loss_fn
