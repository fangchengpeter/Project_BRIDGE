import numpy as np
import tensorflow as tf

def Byz_random(data, label, model, sess):
    grad = sess.run(model.gradient, feed_dict={model.x: data, model.y_: label, model.keep_prob: 0.5})
    Byz_grad = [(np.random.random(pair[0].shape) * 50) for pair in grad]
    return Byz_grad

def Byz_flip(data, label, model, sess):
    data = (-1) * np.array(data)
    grad = sess.run(model.gradient, feed_dict={model.x: data, model.y_: label, model.keep_prob: 0.5})
    Byz_grad = [pair[0] for pair in grad]
    return Byz_grad

def Byz_random_label(data, label, model, sess):
    np.random.shuffle(label)
    grad = sess.run(model.gradient, feed_dict={model.x: data, model.y_: label, model.keep_prob: 0.5})
    Byz_grad = [pair[0] for pair in grad]
    return Byz_grad

def nonfaulty(data, label, model, sess):
    grad = sess.run(model.gradient, feed_dict={model.x: data, model.y_: label, model.keep_prob: 0.5})
    Byz_grad = [pair[0] for pair in grad]
    return Byz_grad
    
    
Byzantine_strategy = {'random': Byz_random,
                     'flip': Byz_flip,
                     'random_label': Byz_random_label,
                     'nonfaulty': nonfaulty}


