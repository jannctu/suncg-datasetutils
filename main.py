from __future__ import print_function
import os
from suncg import SUNCG
import matplotlib.pyplot as plt
path = 'C:/Users/Jan/PycharmProjects/SUNCG/'


TARGET_SHAPE = (256, 256)
da = SUNCG(path_to=path)

file_ids, cnts, sgmnts, depths = da.get_train()

#print(cnts[0].shape)
#plt.imshow(cnts[0].reshape(480,640))
#plt.show()

#files = os.listdir(path)
#mi = 1000
#ma = 0
#for name in files:
    #print(name)
	#files2 = os.listdir(path + name)
	#n = len(files2)
	#print(type(n))
	#quit()
	#if (n < mi): 
	#	mi = n
	
	#if (n > ma): 
	#	ma = n


#print(mi)		
#print(ma)		