# BRIDGE-codebase for reproducibility of the journal paper published in IEEE-TSIPN
Byzantine-resilient Decentralized Machine Learning: Codebase for experiment in journal paper BRIDGE published in IEEE-TSIPN.
## Table of Contents
<!-- MarkdownTOC -->
- [General Information](#introduction)
- [BRIDGE Experiments](#bridge)
- [ByRDiE Experiments](#byrdie)
- [Plotting](#plotting)
- [Contributors](#contributors)
<!-- /MarkdownTOC -->

<a name="introduction"></a>
# General Information
This repo provides implementations of **Byzantine-resilient Distributed Coordinate Descent for Decentralized Learning (ByRDiE)**, **Byzantine-resilient Decentralized Gradient Descent (BRIDGE)**, and different variants of the BRIDGE algorithm. In addition, it includes code to implement decentralized machine learning in the presence of Byzantine (malicious) nodes. The codebase in particular can be used to reproduce the decentralized learning experiments reported in the IEEE journal paper named "BRIDGE: Byzantine-resilient Decentralized Gradient Descent" which is available at (https://ieeexplore.ieee.org/document/9815556).
## License and Citation
The code in this repo is being released under the GNU General Public License v3.0; please refer to the [LICENSE](./LICENSE) file in the repo for detailed legalese pertaining to the license. In particular, if you use any part of this code then you must cite both the original papers as well as this codebase as follows:
**Paper Citations:** 
- C. Fang, Z. Yang and W. U. Bajwa, "BRIDGE: Byzantine-Resilient Decentralized Gradient Descent," in IEEE Transactions on Signal and Information Processing over Networks, vol. 8, pp. 610-626, 2022, doi: 10.1109/TSIPN.2022.3188456. (https://ieeexplore.ieee.org/document/9815556)
## Summary of Experiments
The codebase uses implementations of ByRDiE, BRIDGE, and BRIDGE variants to generate results for Byzantine-resilient decentralized learning. The generated results correspond to experiments in which we simulate a decentralized network that trains a multiclass classifier on the [MNSIT dataset](http://yann.lecun.com/exdb/mnist/) and [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) using a one-layer/convolutional neural network that is implemented in TensorFlow. The network consists of fifty nodes, with each node assigned one thousand and two hundred training samples from the MNIST/CIFAR10 dataset.  The codebase provides nine sets of experiments (network connection probability is set to 0.5 and the dataset is i.i.d. across the nodes in the network unless stated otherwise):
1. Train one-layer neural network with MNIST dataset using Distributed Gradient Descent (DGD), BRIDGE, and three variants of BRIDGE, namely, BRIDGE--Median, BRIDGE--Krum, and BRIDGE--Bulyan, with the Byzantine-resilient algorithms defending against at most one Byzantine nodes while zero nodes actually undergo Byzantine failure. This is the convex faultless setting and the code produces a plot identical to Figure 1 in the paper (Fang et al., 2022) in this case.
2. Train one-layer neural network with MNIST dataset using Distributed Gradient Descent (DGD), BRIDGE, ByRDiE, and three variants of BRIDGE, namely, BRIDGE--Median, BRIDGE--Krum and BRIDGE--Bulyan, with the Byzantine-resilient algorithms defending against at most two/four Byzantine nodes and exactly two/four nodes undergo Byzantine failure. This is the convex faulty setting and the code produces a plot identical to Figure 2 and Figure 4 in the paper (Fang et al., 2022) in this case.
3. Train one-layer neural network with MNIST dataset using BRIDGE, and two variants of BRIDGE, namely, BRIDGE--Median and BRIDGE--Krum, with the Byzantine-resilient algorithms defending against at most six/twelve/eighteen/twenty-four Byzantine nodes and exactly six/twelve/eighteen/twenty-four nodes undergo Byzantine failure. This is the convex faulty setting with network connection probability being set to 0.5/0.75/1.0 and the code produces a plot identical to Figure 3 in the paper (Fang et al., 2022) in this case.
4. Train convolutional neural network with MNIST dataset using Distributed Gradient Descent (DGD), BRIDGE, and three variants of BRIDGE, namely, BRIDGE--Median, BRIDGE--Krum and BRIDGE--Bulyan, with the Byzantine-resilient algorithms defending against at most one Byzantine nodes while zero nodes actually undergo Byzantine failure. This is the nonconvex faultless setting and the code produces a plot identical to Figure 6 in the paper (Fang et al., 2022) in this case.
5. Train convolutional neural network with MNIST dataset using Distributed Gradient Descent (DGD), BRIDGE, and three variants of BRIDGE, namely, BRIDGE--Median, BRIDGE--Krum and BRIDGE--Bulyan, with the Byzantine-resilient algorithms defending against at most two/fourByzantine nodes and exactly two/four nodes undergo Byzantine failure. This is the nonconvex faulty setting and the code produces a plot identical to Figure 7 in the paper (Fang et al., 2022) in this case.
6. Train one-layer neural network with MNIST dataset comparing BRIDGE with BRDSO (J. Peng, W. Li, and Q. Ling, Byzantine-robust decentralized stochastic optimization over static and time-varying networks, Signal Processing, Volume 183,2021,108020, ISSN 0165-1684, https://doi.org/10.1016/j.sigpro.2021.108020.) with the Byzantine-resilient algorithms defending against at most one/two/four Byzantine nodes while zero/two/four nodes actually undergo Byzantine failure. This is the convex extreme non-i.i.d. faultless/faulty setting and the code produces a plot identical to Figure S.1 in the supplementary material of the paper (Fang et al., 2022) in this case.
7. Train a one-layer neural network with MNIST dataset comparing BRIDGE with BRDSO with the Byzantine-resilient algorithms defending against at most one/two/four Byzantine nodes while zero/two/four nodes actually undergo Byzantine failure. This is the convex moderate non-i.i.d. faultless/faulty setting and the code produces a plot identical to Figure S.2 in the supplementary material of the paper (Fang et al., 2022) in this case.
8. Train a one-layer neural network with CIFAR dataset using BRIDGE, and two variants of BRIDGE, namely, BRIDGE--Median and BRIDGE--Krum, with the Byzantine-resilient algorithms defending against at most one/two/four/six Byzantine nodes while zero/two/four/six nodes actually undergo Byzantine failure. This is the convex faultless/faulty setting and the code produces a plot identical to Figure 5 in the paper (Fang et al., 2022) in this case.
9. Train convolutional neural network with MNIST dataset using BRIDGE, and two variants of BRIDGE, namely, BRIDGE--Median and BRIDGE--Krum, with the Byzantine-resilient algorithms defending against at most one/two/four Byzantine nodes and zero/two/four nodes undergo Byzantine failure. This is the nonconvex faultless/faulty setting and the code produces a plot identical to Figure 8 in the paper (Fang et al., 2022) in this case. (Figure 8 also includes training CIFAR dataset in a centralized setting to compare the results.)

For experiments in both the faultless and the faulty setting with the MNIST dataset where the number of Byzantine nodes is less than four, we ran ten Monte Carlo trials in parallel and averaged the classification accuracy before plotting.

For experiments in both the faultless and the faulty setting with the CIFAR10 dataset and the faulty setting with the MNIST dataset where the number of Byzantine nodes is more than 4, we ran three Monte Carlo trials in parallel and averaged the classification accuracy before plotting.
## Summary of Code
The `dec_BRIDGE.py`, `dec_ByRDiE.py`, and `decentralized learning.py` serve as the "driver" or "main" files where we set up the experiments and call the necessary functions to learn the machine learning model in a decentralized manner. The actual implementations of the various screening methods (ByRDiE, BRIDGE, and variants of BRIDGE) are carried out in the `DecLearning.py` module. While these specific implementations are written for the particular case of training with neural networks using TensorFlow, the core of these implementations can be easily adapted for other machine learning problems.
## Computing Environment
All of our computational experiments were carried out on a Linux high-performance computing (HPC) cluster provided by the Rutgers Office of Advanced Research Computing; specifically, all of our experiments were run on 2x Intel Xeon Platinum 8358 (Ice Lake) Processors (48MB cache, 2.60GHz):
- 3200 MHz DDR4 memory 32-core processors (64 cores/node)
- 16x16GB DIMMS (256GB/node)
- 480GB SSD onboard drive
- 10GigE and Infiniband HDR (100Gb/s) adapters

## Requirements and Dependencies
This code is written in Python and uses TensorFlow.  To reproduce the environment with the necessary dependencies needed for running the code in this repo, we recommend that the users create a `conda` environment using the `environment.yml` YAML file that is provided in the repo. Assuming the conda management system is installed on the user's system, this can be done using the following:
```
$ conda env create -f environment.yml
```
In the case users don't have conda installed on their system, they should check out the `environment.yml` file for the appropriate version of Python as well as the necessary dependencies with their respective versions needed to run the code in the repo.
## Data
The MNIST dataset used in our experiments can be found in the `./BRIDGE-MNIST/data` directory. The `./BRIDGE-MNIST/data/MNIST/raw` directory contains the raw MNIST data, as available from [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/), while the `./BRIDGE-MNIST/data/MNIST_read.py` script reads the data into `numpy` arrays that are then *pickled* for use in the experiments. The pickled numpy arrays are already available in the `./BRIDGE-MNIST/data/MNIST/pickled` directory, so there is no need to rerun our script in order to perform the experiments.

The CIFAR dataset used in our experiments is in the `./BRIDGE-CIFAR/data` directory, as available from [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html). It contains 5 batches of training data and 1 batch of test data that are passed through `./BRIDGE-CIFAR/CIFAR10_read.py` script and ready to be used each time we run `dec_BRIDGE.py`,`dec_ByRDiE.py` and `decentralized learning.py` in `./BRIDGE-CIFAR/` directory.
<a name="bridge"></a>
# BRIDGE Experiments
- Convex setting:
We performed decentralized learning using BRIDGE and some of its variants based on distributed learning screening methods, namely Median, Krum, and Bulyan. To train the one-layer neural network on MNIST with BRIDGE or its variants, run the `dec_BRIDGE.py` script in `./BRIDGE-MNIST` directory and on the CIFAR10 dataset, run the `dec_BRIDGE.py` script in `./BRIDGE-CIFAR` directory. When no screening method is selected, training is done with distributed gradient descent (DGD) without screening. 

```
usage: dec_BRIDGE.py [-h] [-b BYZANTINE] [-gb GOBYZANTINE]
                     [-s {BRIDGE,Median,Krum,Bulyan}]
                     monte_trial

positional arguments:
  monte_trial           A number between 0 and 9 indicates which
  					  Monte Carlo trial to run

optional arguments:
  -h, --help            Show this help message and exit
  -b BYZANTINE, --byzantine BYZANTINE
                        Maximum number of Byzantine nodes to defend
                        against; if none then it defaults to 0
  -gb GOBYZANTINE, --goByzantine GOBYZANTINE
                        Boolean to indicate if the specified number of
                        Byzantine nodes actually send out faulty values
  -s {BRIDGE,Median,Krum,Bulyan}, --screening {BRIDGE,Median,Krum,Bulyan}
                        Screening method to use (BRIDGE, Median,Krum, Bulyan);
                        default is distributed gradient descent without screening
```

- Nonconvex setting:
To train the convolutional neural network on MNIST with BRIDGE or its variants, run the `distributed learning.py` script in `./BRIDGE-MNIST` directory, and on the CIFAR10 dataset, run the `distributed learning.py` script in `./BRIDGE-CIFAR` directory. When no screening method is selected, training is done with distributed gradient descent (DGD) without screening. Unlike convex setting, changing arguments/parameters of BRIDGE and its variant is done by updating the main function in line 85.

- Non-i.i.d. setting with MNIST dataset:
To train the one-layer neural network with the MNIST dataset, follow the same manner as described above. For moderated non-i.i.d. setting (one node contains only two sets of differently labeled data evenly), comment line 27 and uncomment line 14 and line 26 in `dist_data.py` script in either `./BRIDGE-MNIST` directory. For extreme non-i.i.d. setting (one nodes contains data that are labelled as the same number), comment line 27 and uncomment line 15.

### Examples

1) BRIDGE defending against at most two Byzantine nodes with no faulty nodes in the network (faultless setting) with MNIST dataset in the convex setting.

```
$ cd BRIDGE-MNIST
$ python dec_BRIDGE.py 0 -b=2 -s=BRIDGE
```
2) BRIDGE-M defending against at most two Byzantine nodes with exactly two faulty nodes in the network (faulty setting) with CIFAR10 dataset in the convex setting.

```
$ cd BRIDGE-CIFAR
$ python dec_BRIDGE.py 0 -b=2 -gb=True -s=Median
```

3) BRIDGE-B defending against at most four Byzantine nodes with exactly four faulty nodes in the network (faulty setting) with MNIST dataset and nonconvex setting.
```
$ cd BRIDGE-MNIST
$ python decentralized learning.py
with experiment_parameters(agents=50, dataset='MNIST', localsize_N=1200, iteration=1000,batchsize=1200, stepsize=2e-5, screen='bulyan', b=4, Byzantine='random') at line 85 in `distributed learning.py` script
```
4) BRIDGE defending against at most two Byzantine nodes with exactly two faulty nodes in the network (faulty setting) with MNIST dataset, convex and extreme non-i.i.d. setting.
```
$ cd BRIDGE-MNIST
$ python dec_BRIDGE.py 0 -b=2 -gb=True -s=BRIDGE
with self.dist_data ,self.dist_label = self.data_redistribute(data, label) at line 17 in `dist_data.py` script
```
<a name="byrdie"></a>
## ByRDiE Experiments
We performed decentralized learning using ByRDiE, in the presence of actual Byzantine nodes with the MNIST dataset and in the convex setting. The code can be found at [https://github.com/INSPIRE-Lab-US/Byzantine-resilient-distributed-learning](https://github.com/INSPIRE-Lab-US/Byzantine-resilient-distributed-learning).
To train the one-layer neural network on MNIST with ByRDiE, run the `dec_ByRDiE.py` script. 

```
usage: dec_ByRDiE.py [-h] [-b BYZANTINE] [-gb GOBYZANTINE] monte_trial

positional arguments:
  monte_trial           A number between 0 and 9 indicates which
  					  Monte Carlo trial to run

optional arguments:
  -h, --help            Show this help message and exit
  -b BYZANTINE, --byzantine BYZANTINE
                        Maximum number of Byzantine nodes to defend
                        against; if none then it defaults to 0
  -gb GOBYZANTINE, --goByzantine GOBYZANTINE
                        Boolean to indicate if the specified number of
                        Byzantine nodes send out faulty values
```

### Examples

1) ByRDiE defending against at most two Byzantine nodes with exactly two faulty nodes in the network (faulty setting).

```
$ python dec_ByRDiE.py 0 -b=2 -gb=True
```
The paper uses three Monte Carlo trials and averages the classification accuracy to produce Figure 4 in the paper (Fang et al., 2022).

<a name="plotting"></a>
## BRDSO Experiments
The implementation of BRDSO in the non-i.i.d. setting is from the codebase available at https://github.com/pengj97/Byzantine-robust-decentralized-stochastic-optimization.
## BRIDGE Experiments with other datasets
If one would like to produce results of BRIDGE with datasets other than MNIST and CIFAR-10, please use `Byzantine_strategy.py`,` DecLearning.py`, `dec_BRIDGE.py`,`dist_data.py`,` distributed learning.py` and `screening_method.py`, along with `linear_classifier.py` for convex setting or `CNN_model.py` for nonconvex setting in the BRIDGE-MNIST folder.
# Plotting
All results generated by `dec_BRIDGE.py` and `dec_ByRDiE.py` get saved in `./result` folder while results generated by `distributed learning.py` get saved in the same folder where you run the script. After running three/ten independent trials for each Byzantine-resilient decentralized method as described above, run different `plotxx.py` scripts to generate the plots that are identical to the Figures in the paper (Fang et al., 2022). Detailed procedures are introduced below:
1) Run `plot.py` script in `./BRIDGE-MNIST` directory to generate figures identical to Figure 1 and Figure 2 in the paper (Fang et al., 2022).

2) Run `plot50.py` , `plot75.py` and `plot100.py` scripts in `./BRIDGE-MNIST` directory to generate figures identical to Figure 3 in the paper (Fang et al., 2022).

3) Run `plotbvb.py` script in `./BRIDGE-MNIST` directory to generate figures identical to Figure 4 in the paper (Fang et al., 2022).

4) Run `plotnc.py` script in `./BRIDGE-MNIST` directory to generate figures identical to Figure 6 and Figure 7 in the paper (Fang et al., 2022).

6) Run `plotnoniid.py` script in `./BRIDGE-MNIST` directory using different file names as input to generate figures identical to Figure S.1 and Figure S.2 in the supplementary material of the paper (Fang et al., 2022).

7) Run `plot.py` script in `./BRIDGE-CIFAR` directory to generate figures identical to Figure 5 in the paper (Fang et al., 2022).

8) Run `plotnc.py` script in `./BRIDGE-CIFAR` directory to generate figures identical to Figure 8 in the paper (Fang et al., 2022).

# Contributors
The algorithmic implementations and experiments were originally developed by the authors of the papers listed above:
- [Cheng Fang](https://ieeexplore.ieee.org/author/37089460236)
- [Zhixiong Yang](https://www.linkedin.com/in/zhixiong-yang-67139152/)
- [Waheed U. Bajwa](http://www.inspirelab.us/)

The reproducibility of this codebase and publicizing of it was made possible by:
- [Cheng Fang](https://ieeexplore.ieee.org/author/37089460236)
- [Joseph Shenouda](https://joeshenouda.github.io/)
- [Waheed U. Bajwa](http://www.inspirelab.us/)
