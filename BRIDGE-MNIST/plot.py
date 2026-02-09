'''
Plots 10 independent trials of each algorithm and saves to a file  './result/plot_dec.png'
'''
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

dgd_b0 = []
dgd_b2 = []
dgd_b4 = []

#byrdie_b1_faultless = []
#byrdie_b2 = []

bridge_b1 = []
bridge_b2 = []
bridge_b4 = []

median_b1 = []
median_b2 = []
median_b4 = []

krum_b1 = []
krum_b2 = []
krum_b4 = []

bulyan_b1 = []
bulyan_b2 = []
bulyan_b4 = []

#krum_b4_faultless = []
#krum_b4 = []

#krum_b3_faultless = []
#krum_b3 = []



for monte in range(10):
    with open(f'./result/DGD/result_50_DGD_b0_faultless_{monte}.pickle', 'rb') as handle:
        dgd_b0.append(pickle.load(handle))
    with open(f'./result/DGD/result_DGD_b2_{monte}.pickle', 'rb') as handle:
        dgd_b2.append(pickle.load(handle))
    with open(f'./result/DGD/result_DGD_b4_{monte}.pickle', 'rb') as handle:
        dgd_b4.append(pickle.load(handle))
        
    #with open(f'./result/ByRDiE/result_ByRDiE_b2_faultless_{monte}.pickle', 'rb') as handle:
        #byrdie_b2_faultless.append(pickle.load(handle))
    #with open(f'./result/ByRDiE/result_ByRDiE_b2_{monte}.pickle', 'rb') as handle:
        #byrdie_b2.append(pickle.load(handle))
    
    with open(f'./result/BRIDGE/result_BRIDGE_b1_{monte}.pickle','rb') as handle:
        bridge_b1.append(pickle.load(handle))
    with open(f'./result/BRIDGE/result_BRIDGE_b2_{monte}.pickle','rb') as handle:
        bridge_b2.append(pickle.load(handle))
    with open(f'./result/BRIDGE/result_BRIDGE_b4_{monte}.pickle','rb') as handle:
        bridge_b4.append(pickle.load(handle))
    
    with open(f'./result/Median/result_Median_b1_{monte}.pickle','rb') as handle:
        median_b1.append(pickle.load(handle))
    with open(f'./result/Median/result_Median_b2_{monte}.pickle','rb') as handle:
        median_b2.append(pickle.load(handle))
    with open(f'./result/Median/result_Median_b4_{monte}.pickle','rb') as handle:
        median_b4.append(pickle.load(handle))

    with open(f'./result/Krum/result_Krum_b1_{monte}.pickle','rb') as handle:
        krum_b1.append(pickle.load(handle))
    with open(f'./result/Krum/result_Krum_b2_{monte}.pickle','rb') as handle:
        krum_b2.append(pickle.load(handle))
    with open(f'./result/Krum/result_Krum_b4_{monte}.pickle','rb') as handle:
        krum_b4.append(pickle.load(handle))
        
    with open(f'./result/Bulyan/result_Bulyan_b1_{monte}.pickle','rb') as handle:
        bulyan_b1.append(pickle.load(handle))
    with open(f'./result/Bulyan/result_Bulyan_b2_{monte}.pickle','rb') as handle:
        bulyan_b2.append(pickle.load(handle))
    with open(f'./result/Bulyan/result_Bulyan_b4_{monte}.pickle','rb') as handle:
        bulyan_b4.append(pickle.load(handle))
        
dgd_b0=np.array([np.array(xi) for xi in dgd_b0],dtype=object)
dgd_b2=np.array([np.array(xi) for xi in dgd_b2],dtype=object)
dgd_b4=np.array([np.array(xi) for xi in dgd_b4],dtype=object)

bridge_b1=np.array([np.array(xi) for xi in bridge_b1],dtype=object)
bridge_b2=np.array([np.array(xi) for xi in bridge_b2],dtype=object)
bridge_b4=np.array([np.array(xi) for xi in bridge_b4],dtype=object)

median_b1=np.array([np.array(xi) for xi in median_b1],dtype=object)
median_b2=np.array([np.array(xi) for xi in median_b2],dtype=object)
median_b4=np.array([np.array(xi) for xi in median_b4],dtype=object)

krum_b1=np.array([np.array(xi) for xi in krum_b1],dtype=object)
krum_b2=np.array([np.array(xi) for xi in krum_b2],dtype=object)
krum_b4=np.array([np.array(xi) for xi in krum_b4],dtype=object)

bulyan_b1=np.array([np.array(xi) for xi in bulyan_b1],dtype=object)
bulyan_b2=np.array([np.array(xi) for xi in bulyan_b2],dtype=object)
bulyan_b4=np.array([np.array(xi) for xi in bulyan_b4],dtype=object)

#dgd_b2 = np.array(dgd_b2, dtype=object)
#for i in range(10):
# print(len(dgd_b2[i]))

 
smooth_dgd_b0 = np.mean(dgd_b0, axis=0)
smooth_dgd_b2 = np.mean(dgd_b2, axis=0)
smooth_dgd_b4 = np.mean(dgd_b4, axis=0)

#smooth_byrdie_b1_faultless = np.mean(byrdie_b1_faultless, axis=0)
#smooth_byrdie_b2 = np.mean(byrdie_b2, axis=0)
#smooth_byrdie_b2_FL = np.mean(smooth_byrdie_b2_faultless, axis=1)
#smooth_byrdie_b2 = np.mean(smooth_byrdie_b2, axis=1)

smooth_bridge_b1 = np.mean(bridge_b1, axis=0)
smooth_bridge_b2 = np.mean(bridge_b2, axis=0)
smooth_bridge_b4 = np.mean(bridge_b4, axis=0)

smooth_median_b1 = np.mean(median_b1, axis=0)
smooth_median_b2 = np.mean(median_b2, axis=0)
smooth_median_b4 = np.mean(median_b4, axis=0)

smooth_krum_b1 = np.mean(krum_b1, axis=0)
smooth_krum_b2 = np.mean(krum_b2, axis=0)
smooth_krum_b4 = np.mean(krum_b4, axis=0)

smooth_bulyan_b1 = np.mean(bulyan_b1, axis=0)
smooth_bulyan_b2 = np.mean(bulyan_b2, axis=0)
smooth_bulyan_b4 = np.mean(bulyan_b4, axis=0)

scalar_comms = [n for n in range(1000)]

#byrdie_axis = []
#for t in range(100):
 #   for p in range(39):
  #      byrdie_axis.append(t * 7840 + (p+1) * 200)
  #  for p in range(10, 11):
   #     byrdie_axis.append((t+1) * 7840 + p)

plot_faultless = plt.figure(figsize=(8,6))

plt.plot(scalar_comms, smooth_dgd_b0*100, markevery=50, marker='v')
#plt.plot(byrdie_axis[:3960], smooth_byrdie_b2_FL*100, markevery=200, marker='.')
plt.plot(scalar_comms, smooth_bridge_b1*100, markevery=50, marker='p', color='g')
plt.plot(scalar_comms, smooth_median_b1*100, markevery=50, marker='s', color='r')
plt.plot(scalar_comms, smooth_krum_b1*100, markevery=50, marker='s', color='m')
plt.plot(scalar_comms, smooth_bulyan_b1*100, markevery=50, marker='x')
print(smooth_bridge_b1[999]*100)
print(smooth_median_b1[999]*100)
print(smooth_krum_b1[999]*100)
print(smooth_bulyan_b1[999]*100)
plt.ylim((5,90))
plt.ylabel('Average classification accuracy (%)', fontsize= 15)
plt.xlabel('Number of iterations', fontsize= 15)
#plt.title('Faultless setting')

plt.legend(['DGD','BRIDGE-T (faultless, b=1)','BRIDGE-M (faultless, b=1)','BRIDGE-K (faultless, b=1)','BRIDGE-B (faultless, b=1)'], loc='right', fontsize= 15)

plt.savefig('./result/plot_convex1.png', bbox_inches='tight')

plot_faultless = plt.figure(figsize=(8,6))
plt.plot(scalar_comms, smooth_dgd_b2*100, markevery=50, marker='v')
#plt.plot(byrdie_axis[:3960], smooth_byrdie_b2*100, markevery=200, marker='.')
plt.plot(scalar_comms, smooth_bridge_b2*100, markevery=50, marker='p', color='g')
plt.plot(scalar_comms, smooth_median_b2*100, markevery=50, marker='s', color='r')
plt.plot(scalar_comms, smooth_krum_b2*100, markevery=50, marker='s', color='m')
plt.plot(scalar_comms, smooth_bulyan_b2*100, markevery=50, marker='x')
print(smooth_bridge_b2[999]*100)
print(smooth_median_b2[999]*100)
print(smooth_krum_b2[999]*100)
print(smooth_bulyan_b2[999]*100)

plt.ylim((5,90))
plt.ylabel('Average classification accuracy (%)', fontsize= 15)
plt.xlabel('Number of iterations', fontsize= 15)
#plt.title('Faulty b=2 setting')
plt.legend(['DGD (b=2)','BRIDGE-T (b=2)','BRIDGE-M (b=2)','BRIDGE-K (b=2)','BRIDGE-B (b=2)'], loc='lower right', fontsize= 15)
plt.savefig('./result/plot_convex2.png', bbox_inches='tight')

plot_faultless = plt.figure(figsize=(8,6))
plt.plot(scalar_comms, smooth_dgd_b4*100, markevery=50, marker='v')
#plt.plot(byrdie_axis[:3960], smooth_byrdie_b2_FL*100, markevery=200, marker='.')
plt.plot(scalar_comms, smooth_bridge_b4*100, markevery=50, marker='p', color='g')
plt.plot(scalar_comms, smooth_median_b4*100, markevery=50, marker='s', color='r')
plt.plot(scalar_comms, smooth_krum_b4*100, markevery=50, marker='s', color='m')
plt.plot(scalar_comms, smooth_bulyan_b4*100, markevery=50, marker='x')
print(smooth_bridge_b4[999]*100)
print(smooth_median_b4[999]*100)
print(smooth_krum_b4[999]*100)
print(smooth_bulyan_b4[999]*100)
plt.ylim((5,90))
plt.ylabel('Average classification accuracy (%)', fontsize= 15)
plt.xlabel('Number of iterations', fontsize= 15)
#plt.title('Faulty b=4 setting')
plt.legend(['DGD (b=4)','BRIDGE-T (b=4)','BRIDGE-M (b=4)','BRIDGE-K (b=4)','BRIDGE-B (b=4)'], loc='right', fontsize= 11)
plt.savefig('./result/plot_convex3.png', bbox_inches='tight')