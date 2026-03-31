# Decentralized nonconvex learning with CNN (BRIDGE)
#
# Each node maintains its own CNN model and communicates only with graph
# neighbors (peer-to-peer). There is no central parameter server.
# Specify the number of Byzantine nodes, whether they actually send faulty
# values, and the screening method to defend against them.
# Run this 10 times with monte_trial 0-9 for independent trials.

import numpy as np
from pre_proc import data_prep
from CNN_model import CNN
from DecLearning import DecLearning
import tensorflow as tf
import time
import pickle
import random
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("monte_trial",
                    help="A number between 0 and 9 to indicate which Monte Carlo trial to run",
                    type=int)
parser.add_argument("-b", "--byzantine",
                    help="Maximum number of Byzantine nodes to defend against; defaults to 0",
                    type=int)
parser.add_argument("-gb", "--goByzantine",
                    help="Boolean to indicate if Byzantine nodes actually send faulty values",
                    type=bool)
parser.add_argument("-s", "--screening",
                    help="Screening method (BRIDGE, Median, Krum, Bulyan); default is no screening",
                    choices=['BRIDGE', 'Median', 'Krum', 'Bulyan'],
                    type=str)

args = parser.parse_args()

monte_trial = args.monte_trial
b = args.byzantine if args.byzantine else 0
goByzantine = args.goByzantine if args.goByzantine else False

min_neighbor = 0
if args.screening:
    screen_method = args.screening
    dec_method = screen_method
    if screen_method == 'Bulyan':
        min_neighbor = 4 * b + 1
else:
    screen_method = None
    dec_method = 'DGD'

print(f'Starting Monte Carlo trial {monte_trial}')
start = time.time()

# Directory to store results
os.makedirs(f'./result/{dec_method}', exist_ok=True)

# Set random seeds for reproducibility
np.random.seed(30 + monte_trial)
random.seed(a=30 + monte_trial)
tf.random.set_seed(30 + monte_trial)

# Network and training parameters
num_nodes = 20
batchsize = 120
stepsize = 1e-4
T = 5000

para = DecLearning(dataset='MNIST', nodes=num_nodes, byzantine=b, total_samples=60000)
para.gen_graph(min_neigh=min_neighbor, con_rate=50)
local_set, test_data, test_label = data_prep(para.dataset, para.M, para.N, one_hot=True)
neighbors = para.get_neighbor()

# Each node has its own CNN — no central server
w_nodes = [CNN(stepsize=stepsize) for _ in range(num_nodes)]

save = []

for iteration in range(T):
    # Peer-to-peer weight sharing with optional Byzantine screening
    para.communication_multilayer(w_nodes, neighbors, b=para.b,
                                  goByzantine=goByzantine,
                                  screenMethod=screen_method)

    # Evaluate accuracy at each node and report the mean
    accuracy = [para.acc_test(node, test_data, test_label) for node in w_nodes]
    mean_acc = np.mean(accuracy)
    print(f'Iteration {iteration}: mean accuracy = {mean_acc:.4f}')
    save.append(mean_acc)

    # Local mini-batch gradient step at every node
    para.node_update_cnn(w_nodes, local_set, batchsize)

# Save per-iteration accuracy and final weights
wb = [node.weights() for node in w_nodes]
end = time.time()

if b != 0 and goByzantine:
    filename = f'./result/{dec_method}/result_{dec_method}_b{b}_{monte_trial}.pickle'
else:
    filename = f'./result/{dec_method}/result_{num_nodes}_{dec_method}_b{b}_faultless_{monte_trial}.pickle'

print(f'Monte Carlo {monte_trial} done! Time elapsed: {end - start:.1f}s')

with open(filename, 'wb') as handle:
    pickle.dump(save, handle)
    pickle.dump(wb, handle)
