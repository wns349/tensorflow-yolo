import cv2
import numpy as np
import tensorflow as tf

from inference import preprocess, postprocess
from layers import input_layer, conv2d_bn_act, max_pool2d, reorg, route
from yolo import load_weights as _load_weights


def create_full_network(num_anchors, num_classes, is_training=False, scope="yolo", input_shape=(416, 416, 3)):
    layers = []
    with tf.variable_scope(scope):
        layers.append(input_layer([None, input_shape[0], input_shape[1], input_shape[2]], "input"))

        for filter_size in (32, 64):
            layers.append(conv2d_bn_act(layers[-1].out, filter_size, 3, stride=1, is_training=is_training))
            layers.append(max_pool2d(layers[-1].out, 2, stride=2))

        for filter_size in (128, 256):
            layers.append(conv2d_bn_act(layers[-1].out, filter_size, 3, stride=1, is_training=is_training))
            layers.append(conv2d_bn_act(layers[-1].out, filter_size // 2, 1, stride=1, is_training=is_training))
            layers.append(conv2d_bn_act(layers[-1].out, filter_size, 3, stride=1, is_training=is_training))
            layers.append(max_pool2d(layers[-1].out, 2, stride=2))

        layers.append(conv2d_bn_act(layers[-1].out, 512, 3, stride=1, is_training=is_training))
        layers.append(conv2d_bn_act(layers[-1].out, 256, 1, stride=1, is_training=is_training))
        layers.append(conv2d_bn_act(layers[-1].out, 512, 3, stride=1, is_training=is_training))
        layers.append(conv2d_bn_act(layers[-1].out, 256, 1, stride=1, is_training=is_training))
        layers.append(conv2d_bn_act(layers[-1].out, 512, 3, stride=1, is_training=is_training))
        layers.append(max_pool2d(layers[-1].out, 2, stride=2))

        layers.append(conv2d_bn_act(layers[-1].out, 1024, 3, stride=1, is_training=is_training))
        layers.append(conv2d_bn_act(layers[-1].out, 512, 1, stride=1, is_training=is_training))
        layers.append(conv2d_bn_act(layers[-1].out, 1024, 3, stride=1, is_training=is_training))
        layers.append(conv2d_bn_act(layers[-1].out, 512, 1, stride=1, is_training=is_training))
        layers.append(conv2d_bn_act(layers[-1].out, 1024, 3, stride=1, is_training=is_training))

        layers.append(conv2d_bn_act(layers[-1].out, 1024, 3, stride=1, is_training=is_training))
        layers.append(conv2d_bn_act(layers[-1].out, 1024, 3, stride=1, is_training=is_training))
        layers.append(route([layers[-9].out]))
        layers.append(conv2d_bn_act(layers[-1].out, 64, 1, stride=1, is_training=is_training))
        layers.append(reorg(layers[-1].out, 2))
        layers.append(route([layers[-1].out, layers[-4].out]))
        layers.append(conv2d_bn_act(layers[-1].out, 1024, 3, stride=1, is_training=is_training))

        layers.append(conv2d_bn_act(layers[-1].out, num_anchors * (5 + num_classes), 1, 1,
                                    use_batch_normalization=False,
                                    activation_fn="linear",
                                    is_training=is_training))

        # rename last layer for convenience
        layers[-1].out = tf.identity(layers[-1].out, "output")
    return layers


def create_tiny_voc_network(num_anchors, num_classes, is_training=False, scope="yolo", input_shape=(416, 416, 3)):
    layers = []
    with tf.variable_scope(scope):
        layers.append(input_layer([None, input_shape[0], input_shape[1], input_shape[2]], "input"))

        for filter_size, pool_stride in zip([16, 32, 64, 128, 256, 512], [2, 2, 2, 2, 2, 1]):
            layers.append(conv2d_bn_act(layers[-1].out, filter_size, 3, is_training=is_training))
            layers.append(max_pool2d(layers[-1].out, 2, stride=pool_stride))

        for _ in range(2):
            layers.append(conv2d_bn_act(layers[-1].out, 1024, 3, is_training=is_training))

        layers.append(conv2d_bn_act(layers[-1].out, num_anchors * (5 + num_classes), 1, 1,
                                    use_batch_normalization=False,
                                    activation_fn="linear",
                                    is_training=is_training))

        # rename last layer for convenience
        layers[-1].out = tf.identity(layers[-1].out, "output")
    return layers


def create_tiny_network(num_anchors, num_classes, is_training=False, scope="yolo", input_shape=(416, 416, 3)):
    layers = []
    with tf.variable_scope(scope):
        layers.append(input_layer([None, input_shape[0], input_shape[1], input_shape[2]], "input"))

        for filter_size, pool_stride in zip([16, 32, 64, 128, 256, 512], [2, 2, 2, 2, 2, 1]):
            layers.append(conv2d_bn_act(layers[-1].out, filter_size, 3, is_training=is_training))
            layers.append(max_pool2d(layers[-1].out, 2, stride=pool_stride))

        layers.append(conv2d_bn_act(layers[-1].out, 1024, 3, is_training=is_training))
        layers.append(conv2d_bn_act(layers[-1].out, 512, 3, is_training=is_training))

        layers.append(conv2d_bn_act(layers[-1].out, num_anchors * (5 + num_classes), 1, 1,
                                    use_batch_normalization=False,
                                    activation_fn="linear",
                                    is_training=is_training))

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

    return _load_weights(layers, weights)


def create_loss_function(pred, batch_size, pred_h, pred_w, anchors, num_classes):
    no_b = len(anchors) // 2

    # reshape to [N, H, W, B, x,y,w,h,p(obj),C]
    pred = tf.reshape(pred, [-1, pred_h, pred_w, no_b, 5 + num_classes])

    # x, y prediction
    tile = tf.to_float(tf.tile(tf.range(pred_w), [pred_h]))  # make H x W tile
    cell_x = tf.reshape(tile, [1, pred_h, pred_w, 1, 1])  # [N, H, W, B, C]
    cell_y = tf.transpose(cell_x, [0, 2, 1, 3, 4])  # Flip H, W
    cell_xy = tf.concat([cell_x, cell_y], axis=-1)  # [1, H, W, 1, 2]
    cell_xy = tf.tile(cell_xy, [batch_size, 1, 1, no_b, 1])
    pred_xy = tf.sigmoid(pred[..., 0:2]) + cell_xy  # bx, by in paper

    # w, h prediction
    reshaped_anchors = np.reshape(anchors, [1, 1, 1, no_b, 2])
    pred_wh = reshaped_anchors * tf.exp(pred[..., 2:4])  # bw, bh in paper

    # objectness prediction
    pred_obj = tf.sigmoid(pred[..., 4])  # Pr(object) * IOU(b, object)
    pred_obj = tf.expand_dims(pred_obj, axis=-1)

    # class prediction
    pred_class = pred[..., 5:]
    # ----
    # create placeholders for ground truth
    truth = tf.placeholder(tf.float32, [None, pred_h, pred_w, no_b, 5 + num_classes])

    # x, y truth
    true_xy = truth[..., 0:2]

    # w, h truth
    true_wh = truth[..., 2:4]

    # objectness truth = IOU(pred, truth)
    truth_min = true_xy - true_wh / 2.
    truth_max = true_xy + true_wh / 2.
    truth_area = true_wh[..., 0] * true_wh[..., 1]
    pred_min = pred_xy - pred_wh / 2.
    pred_max = pred_xy + pred_wh / 2.
    pred_area = pred_wh[..., 0] * pred_wh[..., 1]
    intersection_min = tf.maximum(truth_min, pred_min)
    intersection_max = tf.minimum(truth_max, pred_max)
    intersection_wh = tf.maximum(intersection_max - intersection_min, 0.)
    intersection_area = intersection_wh[..., 0] * intersection_wh[..., 1]
    union_area = truth_area + pred_area - intersection_area
    true_obj = truth[..., 4] * tf.truediv(intersection_area, union_area)  # Pr(object) * IOU(pred, truth)
    true_obj = tf.expand_dims(true_obj, axis=-1)

    # class truth
    true_class = tf.argmax(truth[..., 5:], axis=-1)
    # ----
    # create mask placeholder
    mask_obj = tf.placeholder(tf.float32, [None, pred_h, pred_w, no_b])
    mask_noobj = tf.placeholder(tf.float32, [None, pred_h, pred_w, no_b])

    mask_obj = tf.expand_dims(mask_obj, axis=-1)  # [N, H, W, B, 1]
    mask_noobj = tf.expand_dims(mask_noobj, axis=-1)  # [N, H, W, B, 1]

    # ---
    # calculate loss
    lambda_coord = 5.
    lambda_noobj = .5
    loss_xy = lambda_coord * tf.reduce_sum(tf.square(true_xy - pred_xy) * mask_obj) / batch_size
    loss_wh = lambda_coord * tf.reduce_sum(tf.square(tf.sqrt(true_wh) - tf.sqrt(pred_wh)) * mask_obj) / batch_size
    loss_obj = tf.reduce_sum(tf.square(true_obj - pred_obj) * mask_obj) / batch_size
    loss_noobj = lambda_noobj * tf.reduce_sum(tf.square(true_obj - pred_obj) * mask_noobj) / batch_size
    softmax_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_class, logits=pred_class)
    loss_class = tf.reduce_sum(lambda_coord * softmax_class) / batch_size

    loss = (loss_xy + loss_wh + loss_obj + loss_noobj + loss_class)

    # tensorboard
    with tf.name_scope("losses"):
        tf.summary.scalar("loss_xy", loss_xy)
        tf.summary.scalar("loss_wh", loss_wh)
        tf.summary.scalar("loss_obj", loss_obj)
        tf.summary.scalar("loss_noobj", loss_noobj)
        tf.summary.scalar("loss_class", loss_class)

    # Make placeholder for ground truth
    placeholders = {
        "ground_truth": truth,
        "ground_truth_mask_obj": mask_obj,
        "ground_truth_mask_noobj": mask_noobj
    }
    return loss, placeholders


def predict(params):
    test_image_path = params["test_image"]
    anchors = params["anchors"]
    class_names = params["class_names"]
    weights = params["weights"]
    threshold = float(params["threshold"])
    iou_threshold = float(params["iou_threshold"])
    builder = params["builder"]
    weights_loader = params["loader"]
    scope = params["scope"]
    cb = params["result_callback"] if "result_callback" in params else None

    # create network
    layers = builder(anchors, class_names, False, scope)
    layer_in = layers[0].out
    layer_in_h, layer_in_w = layer_in.shape.as_list()[1:3]
    layer_out = layers[-1].out
    layer_out_h, layer_out_w = layer_out.shape.as_list()[1:3]

    # run prediction
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ops = weights_loader(layers, weights)
        print("Initializing weights...")
        sess.run(ops)

        print("Test image: {}".format(test_image_path))
        img = cv2.imread(test_image_path)
        net_img = preprocess(img, (layer_in_h, layer_in_w))
        net_img = np.expand_dims(net_img, axis=0)  # since only a single image is evaluated here

        net_out = sess.run(layer_out, feed_dict={layer_in: net_img})
        net_out = np.reshape(net_out, [-1, layer_out_h, layer_out_w, len(anchors) // 2, 5 + len(class_names)])

        results = postprocess(net_out, anchors, threshold, iou_threshold, layer_out_h, layer_out_w)
        result = results[0]  # only a single image

        if cb is not None:
            cb(result)


if __name__ == "__main__":
    with open("./resource/voc.names", "r") as f:
        v_names = [l.strip() for l in f.readlines()]
    with open("./resource/yolov2-coco.anchors", "r") as f:
        v_anchors = [float(t) for t in f.readline().split(",")]
    layers = create_full_network(v_anchors, v_names)
    net_out = layers[-1].out
    net_h, net_w = net_out.get_shape().as_list()[1:3]
    loss, placeholders = create_loss_function(net_out, 1, net_h, net_w, v_anchors, v_names)

    # ops = load_weights(layers, "./bin/yolov2.weights")
    # with tf.Session() as sess:
    #     sess.run(ops)
    print(loss, placeholders)
    print("Done!")
