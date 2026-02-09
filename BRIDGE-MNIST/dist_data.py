# read and preprocess data
import numpy as np
import random
import pickle
import os

class dis_data:
    '''
    Encapsulation of our training data split across multiple nodes
    '''
    def __init__(self, data, label, nodes, shuffle=False, index=None, one_hot=True):
        self.size = len(data)
        self.nodes = nodes
        #self.all_data ,self.all_label = self.data_redistribute(data, label)
        #self.dist_data ,self.dist_label = self.data_redistribute(data, label)
        self.all_data  = data
        self.all_label = label
        if index:
            self.index = index
        else:
            self.index = list(range(self.size))
        if shuffle:
            self.shuffle()
      
        #self.dist_data, self.dist_label = self.distribute(nodes)
        #self.dist_data, self.dist_label = self.nv5distribute(nodes, self.all_data, self.all_label)
        self.dist_data, self.dist_label = self.distribute(nodes,self.all_data, self.all_label)
        
        if one_hot:
            new_label = []
            for node in self.dist_label:
                new_label.append(_one_hot(node))
            self.dist_label = new_label
        #print(f'{self.all_data[1][1]}')
    def shuffle(self):
        '''
        Shuffle the training data and labels, updates the member variables and returns the shuffled data and labels vectors
        '''
        random.shuffle(self.index)
        new_data = []
        new_label = []
        for ind in self.index:
            new_data.append(self.all_data[ind])
            new_label.append(self.all_label[ind])
        new_data = np.asarray(new_data)
        new_label = np.asarray(new_label)
        return new_data, new_label
    def data_redistribute(self, image, label):
        """
        Rearrange the samples in the order of label
        :param image: image, shape(10, 784)
        :param label: label, scalar
        """
        number_sample = len(label)
        im = [[] for _ in range(10)]
        la = [[] for _ in range(10)]
        for i in range(number_sample):
            im[label[i]].append(image[i])
            la[label[i]].append(label[i])
        data_image = []
        data_label = []
        for i in range(10):
            for j in range(len(la[i])):
                data_image.append(im[i][j])
                data_label.append(la[i][j])

        return data_image, data_label
    def distribute(self,nodes,data,label):
        '''
        Evenly distribute the training data across the nodes
        '''
        remainder = self.size % nodes
        frac = int(self.size/nodes)
        
        
        dist_data = []
        dist_label = []
        for n in range(nodes):
            if n == 0:
                dist_data.append(data[0 : frac + remainder])
                dist_label.append(label[0 : frac + remainder])
            else:                
                dist_data.append(data[n * frac : (n + 1) * frac])
                dist_label.append(label[n * frac : (n + 1) * frac])
        dist_data = np.asarray(dist_data,dtype=object)
        dist_label = np.asarray(dist_label,dtype=object)
        return dist_data, dist_label
    def nvdistribute(self,nodes):
        frac = int(self.size/nodes)
        hfrac= int(frac*0.5)
        dpl= int(self.size/10)
        hdpl=int(dpl*0.5)
        dist_data = []
        dist_label = []
        for n in range(nodes):
          dist_data.append(self.all_data[hfrac*n : (n+1)*hfrac] )
          dist_label.append(self.all_label[hfrac*n : (n+1)*hfrac] )
          for j in range(10):          
            dist_data.append(self.all_data[(j*dpl+hdpl+60*n):(j*dpl+hdpl+(n+1)*60)])
            dist_label.append(self.all_label[(j*dpl+hdpl+60*n):(j*dpl+hdpl+(n+1)*60)])
        for k in range(nodes):
          for p in range(10):
            dist_data[11*k].extend(dist_data[11*k+p+1])
            dist_label[11*k].extend(dist_label[11*k+p+1])
        dist_data = np.asarray(dist_data,dtype=object)
        dist_label = np.asarray(dist_label,dtype=object)
        return dist_data, dist_label
    def nv5distribute(self,nodes,data,label):
        remainder = self.size % nodes
        frac = int(self.size/nodes)
        hfrac= int(frac*0.5)
        dpl= int(self.size/10)
        hdpl=int(dpl*0.5)
        dist_data = []
        dist_label = []
        
        for p in range (10):
          for n in range(10):
            for k in range(hfrac):
              dist_data.append(data[(dpl*n+p*hfrac+k)])
              dist_label.append(label[(dpl*n+p*hfrac+k)])
            #dist_data.append(self.all_data[(dpl*n+p*hfrac) : (n*dpl+(p+1)*hfrac)])
            #dist_label.append(self.all_label[(dpl*n+p*hfrac) : (n*dpl+(p+1)*hfrac)])
        
        #for i in range (50):
          #dis_data[2*i].extend(dis_data[2*i+1])
          #dis_label[2*i].extend(dis_label[2*i+1])
   
        dist_data = np.asarray(dist_data,dtype=object)
        dist_label = np.asarray(dist_label,dtype=object)
        return dist_data, dist_label
    def next_batch(self, node, size):
        '''
        Return next batch of distributed training samples with labels
        '''
        l = len(self.dist_data[node])
        sample = []
        label = []
        for _ in range(size):
            index = random.randint(0, l-1)
            sample.append(self.dist_data[node][index])
            label.append(self.dist_label[node][index])
        return sample, label

def data_prep(dataset, nodes, size=0, one_hot=False):
    '''
    Distribute training data across nodes and return test data w/ labels
    '''
    if dataset == 'MNIST':
        train_data, train_label, test_data, test_label = mnist_read_pickled()
        if one_hot:
            test_label = _one_hot(test_label)
    elif dataset == 'CIFAR':
        with open('cifar_dataset.pickle', 'rb') as handle:
            (train_data, train_label, test_data, test_label) = pickle.load(handle)
        train_data, test_data = train_data / 255.0, test_data / 255.0
        train_label = _one_hot(train_label)
        test_label = _one_hot(test_label)
    else:
        raise NameError("Cannot find %s dataset") % (dataset)
    
    if size:
        train_data = train_data[:size]
        train_label = train_label[:size]
        
    full_data = dis_data(train_data, train_label, nodes, shuffle = False, one_hot=one_hot)
    return full_data, test_data, test_label

def _one_hot(label):
    l_oh = []
    for i in label:
        new_l = [0] * 10
        new_l[int(i)] = 1
        l_oh.append(new_l)
    return l_oh

def mnist_read_pickled(path = './data/MNIST/pickled'):
    '''
    Read pickled MNIST train/test data

    Args:
        path: Path to pickled numpy arrays (default: ./data/MNIST/pickled)
    Return:
        train_data
        train_labels
        test_data
        test_labels
    '''
    with open(os.path.join(path, 'train_data.pickle'), 'rb') as handle:
        train_data = pickle.load(handle)
    with open(os.path.join(path, 'train_labels.pickle'), 'rb') as handle:
        train_labels = pickle.load(handle)
    
    with open(os.path.join(path, 'test_data.pickle'), 'rb') as handle:
        test_data = pickle.load(handle)
    with open(os.path.join(path, 'test_labels.pickle'), 'rb') as handle:
        test_labels = pickle.load(handle)
    
    return train_data, train_labels, test_data, test_labels
