'''
Plots 10 independent trials of each algorithm and saves to a file  './result/plot_dec.png'
'''
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt




bridge_b1 = []
bridge_b2 = []
bridge_b4 = []






for monte in range(4,5):

    
    with open(f'./resultnoniid/BRIDGE/result_BRIDGE_b1_{monte}.pickle','rb') as handle:
        bridge_b1.append(pickle.load(handle))
    with open(f'./resultnoniid/BRIDGE/result_BRIDGE_b2_{monte}.pickle','rb') as handle:
        bridge_b2.append(pickle.load(handle))
    with open(f'./resultnoniid/BRIDGE/result_BRIDGE_b4_{monte}.pickle','rb') as handle:
        bridge_b4.append(pickle.load(handle))

        


bridge_b1=np.array([np.array(xi) for xi in bridge_b1],dtype=object)
bridge_b2=np.array([np.array(xi) for xi in bridge_b2],dtype=object)
bridge_b4=np.array([np.array(xi) for xi in bridge_b4],dtype=object)




 

smooth_bridge_b1 = np.mean(bridge_b1, axis=0)
smooth_bridge_b2 = np.mean(bridge_b2, axis=0)
smooth_bridge_b4 = np.mean(bridge_b4, axis=0)

acc_listb0 = []
acc_listb2 = []
acc_listb4 = []

with open(f'./resultnoniid/lingqing/results-exnoniidb0.pkl','rb') as f:
  acc_listb0 =pickle.load(f)
  
  



with open(f'./resultnoniid/lingqing/results-exnoniidb2.pkl','rb') as f:
  acc_listb2 =pickle.load(f)
 
  
  

with open(f'./resultnoniid/lingqing/results-exnoniidb4.pkl','rb') as f:
  acc_listb4 =pickle.load(f)
  
  
acc_listb0=np.array(acc_listb0)
acc_listb2=np.array(acc_listb2)
acc_listb4=np.array(acc_listb4)


scalar_comms = [n for n in range(1000)]



plot_faultless = plt.figure(figsize=(8,6))


plt.plot(scalar_comms, smooth_bridge_b1*100, markevery=50, marker='p', color='g')
plt.plot(scalar_comms, smooth_bridge_b2*100, markevery=50, marker='s', color='r')
plt.plot(scalar_comms, smooth_bridge_b4*100, markevery=50, marker='s', color='m')
plt.plot(scalar_comms, acc_listb0*100, markevery=50, marker='p', color='b')
plt.plot(scalar_comms, acc_listb2*100, markevery=50, marker='p', color='c')
plt.plot(scalar_comms, acc_listb4*100, markevery=50, marker='p', color='y')

plt.ylim((5,90))
plt.ylabel('Average classification accuracy (%)', fontsize=15)
plt.xlabel('Number of iterations', fontsize=15)
#plt.title('Moderate non-i.i.d. setting')
plt.legend(['BRIDGE-T (faultless, b=1)','BRIDGE-T (b=2)','BRIDGE-T (b=4)','BRDSO (faultless, b=1)','BRDSO (b=2)','BRDSO (b=4)'], loc='right', fontsize=15)

plt.savefig('./result/plot_exnoniid.png', bboxinches='tight')

