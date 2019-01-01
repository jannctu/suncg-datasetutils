from __future__ import print_function
import os
from suncg import SUNCG
import matplotlib.pyplot as plt
path = '/media/commlab/TenTB/home/jan/TenTB/SUNCG/'


TARGET_SHAPE = (240,320)
da = SUNCG(path_to=path,target_size=TARGET_SHAPE,batch_size=1)

da.buildFlistDepth('flistID.txt')
#file_ids, cnts, sgmnts, depths, hha,images = da.get_train()

#print(sgmnts[0].shape)
#plt.imshow(hha[2])
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