import tensorflow as tf
import numpy as np


class linear_classifier:
    '''
    Learning a linear classifier using a one-layer neural network (TF2 eager mode)
    '''
    def __init__(self, stepsize=1e-4, sigma2=0.1, adam=False):
        self.W = tf.Variable(tf.random.truncated_normal([3072, 10], stddev=sigma2))
        self.b = tf.Variable(tf.constant(0.1, shape=[10]))
        self.layers = [self.W, self.b]
        if adam:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=stepsize)
        else:
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=stepsize)

    def __call__(self, x):
        x = tf.cast(x, tf.float32)
        return tf.matmul(x, self.W) + self.b

    def compute_loss(self, x, y_):
        logits = self(x)
        return tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.cast(y_, tf.float32), logits=logits))

    def accuracy_eval(self, x, y_):
        logits = self(x)
        correct = tf.equal(tf.argmax(logits, 1), tf.argmax(tf.cast(y_, tf.float32), 1))
        return tf.reduce_mean(tf.cast(correct, tf.float32)).numpy()

    def train_step_fn(self, x, y_, stepsize):
        self.optimizer.learning_rate = stepsize
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x, y_)
        grads = tape.gradient(loss, self.layers)
        self.optimizer.apply_gradients(zip(grads, self.layers))

    def get_gradient_w(self, x, y_):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x, y_)
        return tape.gradient(loss, self.W).numpy()

    def get_gradient_b(self, x, y_):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x, y_)
        return tape.gradient(loss, self.b).numpy()

    def apply_gradient_w(self, grad_w, stepsize):
        self.optimizer.learning_rate = stepsize
        self.optimizer.apply_gradients(
            [(tf.constant(grad_w, dtype=tf.float32), self.W)])

    def apply_gradient_b(self, grad_b, stepsize):
        self.optimizer.learning_rate = stepsize
        self.optimizer.apply_gradients(
            [(tf.constant(grad_b, dtype=tf.float32), self.b)])

    def weights(self):
        '''
        Return list containing weight matrix and offset vector [W, b]
        '''
        return [self.W.numpy(), self.b.numpy()]

    def assign(self, weight):
        self.W.assign(weight[0])
        self.b.assign(weight[1])
