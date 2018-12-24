import numpy as np

from keras_net import darknet_yolov2
from tensorflow.python import keras

K = keras.backend
layers = keras.layers


def load_darknet_weights(model, variable_placeholder, weights_path):
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

    variables = {v.op.name: v for v in model.variables}
    read = 0
    for variable_name in variable_placeholder:
        var = variables[variable_name]
        tokens = var.name.split("/")
        size = np.prod(var.shape.as_list())
        shape = var.shape.as_list()
        if "kernel" in tokens[-1]:
            shape = [shape[3], shape[2], shape[0], shape[1]]
        value = np.reshape(weights[read: read + size], shape)
        if "kernel" in tokens[-1]:
            value = np.transpose(value, (2, 3, 1, 0))
        K.set_value(var, value)
        read += size

    print("Weights ({}/{} read)".format(read, len(weights)))


if __name__ == '__main__':
    input_layer = layers.Input(shape=[416, 416, 3])
    model = darknet_yolov2.build_model(input_tensor=input_layer)
    print(model.summary())
    variable_placeholder = [v.op.name for v in model.variables]
    variable_placeholder.sort(key=lambda v: (v.split('/')[1], v.split('/')[-1]))
    print("variables: ", variable_placeholder)
    load_darknet_weights(model, variable_placeholder, "../bin/yolov2.weights")
    model.save_weights("./checkpoints/full-yolov2-coco")
