import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

byrdie_b2 = []
bridge_b2 = []
with open(f'./result/ByRDiE/result_ByRDiE_b2_2.pickle', 'rb') as handle:
        byrdie_b2.append(pickle.load(handle))
with open(f'./result/BRIDGE/result_BRIDGE_b2_0.pickle','rb') as handle:
        bridge_b2.append(pickle.load(handle))

bridge_b2=np.array([np.array(xi) for xi in bridge_b2],dtype=object)
smooth_byrdie_b2 = np.mean(byrdie_b2, axis=0)
smooth_byrdie_b2 = np.mean(smooth_byrdie_b2, axis=1)
smooth_byrdie_b2 = smooth_byrdie_b2[:200]
smooth_bridge_b2 = np.mean(bridge_b2, axis=0)

for i in range (39000):
  smooth_bridge_b2= np.append(smooth_bridge_b2,smooth_bridge_b2[999])


scalar_comms = [n for n in range(40000)]
byrdie_axis = []
for t in range(100):
    for p in range(39):
        byrdie_axis.append(t * 7840 + (p+1) * 200)
    for p in range(10, 11):
        byrdie_axis.append((t+1) * 7840 + p)
plt.subplot(1,1,1)
plt.plot(byrdie_axis[:200], smooth_byrdie_b2*100, markevery=50, marker='p')
plt.plot(scalar_comms, smooth_bridge_b2*100, markevery=10000, marker='p', color='g')
plt.ylim((5,95))
plt.ylabel('Average classification accuracy (%)')
plt.xlabel('Number of communication iterations')
#plt.title('Faulty b=2 setting')
plt.legend(['ByRDiE','BRIDGE-T'], loc='right')
plt.savefig('./result/plot_bvb.png', bboxinches='tight')
