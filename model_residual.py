import tensorflow as tf
import numpy as np
from tools import *

modelName = 'ResNet'

input_image = tf.placeholder(tf.float32, shape=(None, 1500, 1500, 3))
label_image = tf.placeholder(tf.float32, shape=(None, 1500, 1500, 1))

is_train = tf.placeholder_with_default(True, ())
global_step = tf.Variable(0, trainable=False)


norm_coef = 0.001

keep_prob = tf.cond(is_train, lambda: tf.identity(1.0), lambda: tf.identity(1.0))

learning_rate = tf.train.exponential_decay(0.1, global_step, 1500, 0.5, staircase=True)
tf.summary.scalar('learning_rate', learning_rate)

layer, bias= conv(input_image, "conv1", width=7, stride=2, out_depth=32)
layer = batch_norm(layer, 'bn1', is_train)
layer = tf.nn.relu(layer + bias)

temp = layer

layer, bias = conv(layer, "conv2", width=3, stride=1, out_depth=32)
layer = batch_norm(layer, 'bn2', is_train)
layer = tf.nn.relu(layer + bias)

layer, bias = conv(layer, "conv3", width=3, stride=1, out_depth=32)
layer = batch_norm(layer, 'bn3', is_train)
layer = tf.nn.relu(temp + layer + bias)

temp = layer

layer, bias = conv(layer, "conv4", width=3, stride=1, out_depth=32)
layer = batch_norm(layer, 'bn4', is_train)
layer = tf.nn.relu(layer + bias)

layer, bias = conv(layer, "conv5", width=3, stride=1, out_depth=32)
layer = batch_norm(layer, 'bn5', is_train)
layer = tf.nn.relu(temp + layer + bias)

temp = layer

layer, bias = conv(layer, "conv6", width=3, stride=1, out_depth=32)
layer = batch_norm(layer, 'bn6', is_train)
layer = tf.nn.relu(layer + bias)

layer, bias = conv(layer, "conv7", width=3, stride=1, out_depth=32)
layer = batch_norm(layer, 'bn7', is_train)
layer = tf.nn.relu(temp + layer + bias)

lay_k = layer

layer, bias = conv(lay_k, "conv8", width=1, stride=2, out_depth=64)
layer = batch_norm(layer, 'bn8', is_train)
layer = tf.nn.relu(layer + bias)

temp = layer

layer, bias = conv(lay_k, "conv9", width=7, stride=2, out_depth=64)
layer = batch_norm(layer, 'bn9', is_train)
layer = tf.nn.relu(layer + bias)

layer, bias = conv(layer, "conv10", width=3, stride=1, out_depth=64)
layer = batch_norm(layer, 'bn10', is_train)
layer = tf.nn.relu(temp + layer + bias)

temp = layer

layer, bias = conv(layer, "conv11", width=3, stride=1, out_depth=64)
layer = batch_norm(layer, 'bn11', is_train)
layer = tf.nn.relu(layer + bias)

layer, bias = conv(layer, "conv12", width=3, stride=1, out_depth=64)
layer = batch_norm(layer, 'bn12', is_train)
layer = tf.nn.relu(temp + layer + bias)

temp = layer

layer, bias = conv(layer, "conv13", width=3, stride=1, out_depth=64)
layer = batch_norm(layer, 'bn13', is_train)
layer = tf.nn.relu(layer + bias)

layer, bias = conv(layer, "conv14", width=3, stride=1, out_depth=64)
layer = batch_norm(layer, 'bn14', is_train)
layer = tf.nn.relu(temp + layer + bias)

temp = layer

layer, bias = conv(layer, "conv15", width=3, stride=1, out_depth=64)
layer = batch_norm(layer, 'bn15', is_train)
layer = tf.nn.relu(layer + bias)

layer, bias = conv(layer, "conv16", width=3, stride=1, out_depth=64)
layer = batch_norm(layer, 'bn16', is_train)
layer = tf.nn.relu(temp + layer + bias)


layer = tf.nn.dropout(layer, keep_prob, name="dropout")

layer, bias = conv(layer, "conv17", width=16, stride=2, out_depth=16, transpose=True)
layer = batch_norm(layer, 'bn17', is_train)
layer = tf.nn.relu(layer + bias)

layer, bias = conv(layer, "conv18", width=16, stride=2, out_depth=1, transpose=True)
layer = layer + bias

result = tf.nn.sigmoid(layer)

update_ops = tf.get_collection("update_bn")

error = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=layer, labels=label_image))
tf.summary.scalar('loss', error)
test_summary = tf.summary.scalar('test_loss',  tf.cond(is_train, lambda: tf.identity(0.0), lambda: tf.identity(error)))

full_error = error + norm_coef * tf.reduce_sum(tf.get_collection("l2_losses"))
tf.summary.scalar('full_error', full_error)

with tf.control_dependencies(update_ops):
    train = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(full_error, global_step=global_step)

summary = tf.summary.merge_all()
