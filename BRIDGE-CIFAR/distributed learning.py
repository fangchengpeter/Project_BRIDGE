# Distributed learning

# Specify M agents, N local samples, dataset, failure type, screening type, stepsize, batchsize, T iterations,
# b Byzantine agents

import numpy as np
from pre_proc import data_prep
from CNN_model import CNN
import tensorflow as tf
import time
from screening_method import Byzantine_algs
from Byzantine_strategy import Byzantine_strategy
import pickle


class experiment_parameters:
    def __init__(self, agents, dataset, localsize_N, iteration, batchsize=0, stepsize=1e-4,
                 screen=False, b=0, Byzantine='random'):
        self.dataset = dataset
        self.M = agents
        self.T = iteration
        self.screen = screen
        self.b = b
        self.stepsize = stepsize
        self.batchsize = batchsize
        self.N = localsize_N
        self.Byzantine = Byzantine
        if self.batchsize > self.N:
            raise ValueError("Batch size is large than local datasize")


def data_distribution(dataset, nodes_M, sample_N):  # samples will be distributed evenly to each node
    distributed_data, test_data, test_label = data_prep(dataset, nodes_M, nodes_M * sample_N, one_hot=False)
    return distributed_data, test_data, test_label


def server_to_agent(server, nodes):
    weights = server.weights()
    for node in nodes:
        node.assign(weights)


def agent_calculation(node, models, sets, batchsize):
    data, label = sets.next_batch(node, batchsize)
    w_local = models[node]
    return w_local.get_gradient(
        np.array(data, dtype=np.float32),
        np.array(label, dtype=np.float32),
        training=True
    )


def Byzantine_gradient(strategy, node, models, sets, batchsize):
    data, label = sets.next_batch(node, batchsize)
    w_local = models[node]
    return Byzantine_strategy[strategy](data, label, w_local)


def screen(grad, b, method):
    screened = Byzantine_algs[method](grad, b)
    return screened


def acc_test(model, t_data, t_label):
    return model.accuracy_eval(
        np.array(t_data, dtype=np.float32),
        np.array(t_label, dtype=np.float32)
    )


def server_update(model, gradient, b, screening=False):
    if screening:
        grad = screen(gradient, b, screening)
    else:
        grad = np.mean(gradient, axis=0)
    model.apply_ext_gradient(grad)
    return model


if __name__ == "__main__":
    para = experiment_parameters(agents=25, dataset='CIFAR', localsize_N=2400, iteration=5000,
                                 batchsize=250, stepsize=1e-4, screen='dimensionwise_trimmed_mean', b=1, Byzantine='random')
    local_set, test_data, test_label = data_distribution(para.dataset, para.M, para.N)

    tf.random.set_seed(42)

    w_server = CNN(stepsize=para.stepsize)
    w_nodes = [CNN() for _ in range(para.M)]

    rec = []
    beginning = time.time()
    for iteration in range(para.T):
        # Communication
        server_to_agent(w_server, w_nodes)
        # local gradient calculation
        gradient = []
        # Byzantine nodes are the first b nodes
        for node in range(para.b):
            gradient.append(Byzantine_gradient(para.Byzantine, node, w_nodes, local_set, para.batchsize))
        for node in range(para.b, para.M):
            gradient.append(agent_calculation(node, w_nodes, local_set, para.batchsize))
        # Server update using Adam
        server_update(w_server, gradient, para.b, screening=para.screen)
        if iteration % 1 == 0:
            # test over all test data
            accuracy = acc_test(w_server, test_data, test_label)
            rec.append(accuracy)
            print(accuracy)
            print(iteration)
            print(time.time() - beginning)

    with open('result_BRIDGE_b0_5.pickle', 'wb') as handle:
        pickle.dump(rec, handle)
