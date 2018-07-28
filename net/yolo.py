import os

import numpy as np
import tensorflow as tf


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
            if "weights" in tokens[-1]:
                shape = [shape[3], shape[2], shape[0], shape[1]]
            value = np.reshape(weights[read: read + size], shape)
            if "weights" in tokens[-1]:
                value = np.transpose(value, (2, 3, 1, 0))
            ops.append(tf.assign(var, value, validate_shape=True))
            read += size

    print("Weights ready ({}/{} read)".format(read, len(weights)))
    if read != len(weights):
        print("(warning) read count and total count do not match. Possibly an incorrect weights file.")

    return ops


def load_image_paths(path_to_img_dir):
    return [os.path.join(os.path.abspath(path_to_img_dir), f) for f in os.listdir(path_to_img_dir)
            if any(f.lower().endswith(ext) for ext in ["jpg", "bmp", "png", "gif"])]


def generate_test_batch(img_paths, batch_size):
    total_batches = np.ceil(len(img_paths) / batch_size)

    print(total_batches)
    return None

class Yolo(object):
    def __init__(self):
        pass

