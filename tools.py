import tensorflow as tf
import numpy as np

def conv(_input, name, width, stride, out_depth, transpose=False):
    with tf.variable_scope(name):
        tf.summary.histogram("in", _input)

        input_shape = _input.get_shape().as_list()
        in_depth = input_shape[-1]
        if transpose:
            conv_shape = [width, width, out_depth, in_depth]
        else:
            conv_shape = [width, width, in_depth, out_depth]
        n = width * width * out_depth
        conv_w = tf.get_variable("w", conv_shape, initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))
        conv_b = tf.get_variable("b", out_depth, initializer=tf.random_normal_initializer(0))
        
        tf.add_to_collection("l2_losses", tf.nn.l2_loss(conv_w))

        if transpose:
            output_shape = [tf.shape(_input)[0], input_shape[1] * stride, input_shape[2] * stride, out_depth]
            _input = tf.nn.conv2d_transpose(_input, conv_w, output_shape, [1, stride, stride, 1], padding='SAME')
            _input = tf.reshape(_input, (-1, input_shape[1] * stride, input_shape[2] * stride, out_depth), name="act")
        else:
            _input = tf.nn.conv2d(_input, conv_w, [1, stride, stride, 1], padding='SAME', name="act")

        tf.summary.histogram("out", _input)
        tf.summary.histogram("w", conv_w)
        tf.summary.histogram("b", conv_b)

        return (_input, conv_b)

def maxpool(_input, name, width, stride):
    with tf.variable_scope(name):
        return tf.nn.max_pool(_input, ksize=[1, width, width, 1], strides=[1, stride, stride, 1], padding='SAME')

def batch_norm(_input, name, is_train):
    normed = tf.contrib.layers.batch_norm(_input, center=True, scale=False, decay=0.9, epsilon=1e-5, is_training=is_train, updates_collections="update_bn", fused=True, scope=name)
    with tf.variable_scope(name, reuse=True):
        tf.summary.histogram("bn", normed)
        return normed