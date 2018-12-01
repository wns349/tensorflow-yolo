import tensorflow as tf
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import InputLayer, Conv2D, BatchNormalization, LeakyReLU, MaxPool2D, Lambda, \
    Concatenate, Input
from tensorflow.python.keras.utils import plot_model


def build_model(num_anchors=5, num_classes=80, scope="yolo"):
    with tf.variable_scope(scope):
        # define helper methods
        def conv2d_bn_leaky(_prev, _filter_size, _kernel_size, _strides):
            _prev = Conv2D(_filter_size,
                           kernel_size=_kernel_size,
                           strides=_strides,
                           padding="SAME",
                           use_bias=False,
                           activation=None)(_prev)
            _prev = BatchNormalization(momentum=0.9, epsilon=1e-5)(_prev)
            _prev = LeakyReLU(alpha=0.1)(_prev)
            return _prev

        def space_to_depth_x2(x):
            return tf.space_to_depth(x, block_size=2)

        input_layer = Input(shape=[416, 416, 3])
        x = input_layer
        for f in (32, 64):
            x = conv2d_bn_leaky(x, f, 3, 1)
            x = MaxPool2D(2, strides=2)(x)

        for f in (128, 256):
            x = conv2d_bn_leaky(x, f, 3, 1)
            x = conv2d_bn_leaky(x, f // 2, 1, 1)
            x = conv2d_bn_leaky(x, f, 3, 1)
            x = MaxPool2D(2, strides=2)(x)

        for _ in range(2):
            x = conv2d_bn_leaky(x, 512, 3, 1)
            x = conv2d_bn_leaky(x, 256, 1, 1)
        x = conv2d_bn_leaky(x, 512, 3, 1)
        skip = x
        x = MaxPool2D(2, strides=2)(x)

        for _ in range(2):
            x = conv2d_bn_leaky(x, 1024, 3, 1)
            x = conv2d_bn_leaky(x, 512, 1, 1)
        for _ in range(3):
            x = conv2d_bn_leaky(x, 1024, 3, 1)

        # reorg
        skip = conv2d_bn_leaky(skip, 64, 1, 1)
        skip = Lambda(space_to_depth_x2)(skip)

        x = Concatenate()([skip, x])

        x = conv2d_bn_leaky(x, 1024, 3, 1)

        # final layer
        x = Conv2D(num_anchors * (5 + num_classes),
                   kernel_size=1,
                   strides=1,
                   padding="SAME",
                   use_bias=False,
                   activation="linear")(x)

        return Model([input_layer], [x])


def analyze_model(model):
    print(model.trainable_variables)


if __name__ == '__main__':
    model = build_model()
    # print(model.summary())
    analyze_model(model)
    # plot_model(model, "out.png", show_shapes=True)
