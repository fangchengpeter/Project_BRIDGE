# CNN model for MNIST
# conv--pool--conv--pool--fc--fc

import tensorflow as tf
import numpy as np


class CNN:
    def __init__(self, stepsize=1e-4):
        self.W_conv1 = tf.Variable(tf.random.truncated_normal([5, 5, 1, 32], stddev=0.1))
        self.b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))

        self.W_conv2 = tf.Variable(tf.random.truncated_normal([5, 5, 32, 64], stddev=0.1))
        self.b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))

        self.W_fc1 = tf.Variable(tf.random.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
        self.b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))

        self.W_fc2 = tf.Variable(tf.random.truncated_normal([1024, 10], stddev=0.1))
        self.b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))

        self.layers = [
            self.W_conv1, self.b_conv1,
            self.W_conv2, self.b_conv2,
            self.W_fc1, self.b_fc1,
            self.W_fc2, self.b_fc2,
        ]
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=stepsize)

    def __call__(self, x, training=False):
        x = tf.cast(x, tf.float32)
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        h_conv1 = tf.nn.relu(
            tf.nn.conv2d(x_image, self.W_conv1, strides=[1, 1, 1, 1], padding='SAME') + self.b_conv1)
        h_pool1 = tf.nn.max_pool2d(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        h_conv2 = tf.nn.relu(
            tf.nn.conv2d(h_pool1, self.W_conv2, strides=[1, 1, 1, 1], padding='SAME') + self.b_conv2)
        h_pool2 = tf.nn.max_pool2d(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, self.W_fc1) + self.b_fc1)
        if training:
            h_fc1 = tf.nn.dropout(h_fc1, rate=0.5)
        return tf.matmul(h_fc1, self.W_fc2) + self.b_fc2

    def compute_loss(self, x, y_, training=False):
        logits = self(x, training=training)
        return tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.cast(y_, tf.float32), logits=logits))

    def accuracy_eval(self, x, y_):
        logits = self(x, training=False)
        correct = tf.equal(tf.argmax(logits, 1), tf.argmax(tf.cast(y_, tf.float32), 1))
        return tf.reduce_mean(tf.cast(correct, tf.float32)).numpy()

    def get_gradient(self, x, y_, training=True):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x, y_, training=training)
        return [g.numpy() for g in tape.gradient(loss, self.layers)]

    def apply_ext_gradient(self, grad):
        self.optimizer.apply_gradients(zip(
            [tf.constant(g, dtype=tf.float32) for g in grad], self.layers))

    def train_step_fn(self, x, y_):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x, y_, training=True)
        grads = tape.gradient(loss, self.layers)
        self.optimizer.apply_gradients(zip(grads, self.layers))

    def weights(self):
        return [v.numpy() for v in self.layers]

    def assign(self, weight):
        for var, val in zip(self.layers, weight):
            var.assign(val)
