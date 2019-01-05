from __future__ import print_function
import numpy as np
import os
#from suncg import SUNCG
import matplotlib.pyplot as plt
from utils import inBatchIndex
from scipy.misc import imread, imsave


path = '/media/commlab/TenTB/home/jan/TenTB/SUNCG/'

import sys
sys.path.insert(0, '/media/commlab/TenTB/home/jan/TenTB/tensorflow-data-augmentation/')
from data_aug import *
from read_gt import *

#TARGET_SHAPE = (240,320)
#da = SUNCG(path_to=path,target_size=TARGET_SHAPE,batch_size=1)

#da.buildFlistDepth('flistID.txt')
#file_ids, cnts, sgmnts, depths, hha,images = da.get_train()

lines = [line.rstrip('\n') for line in open(path + 'flistID.txt')]
normal_paths = []
for i in range(10000):
    normal_paths.append(path + 'normals_png/' + lines[i] + '.png')

normal_paths = np.array(normal_paths)

counter = 0
batch_size = 5
IMAGE_SIZE = 256
CH = 3

for ii in range(10000/batch_size):
    print(ii)
    batchID = inBatchIndex(ii, batch_size)
    #gt = load_ground_truth(gt_paths[batchID], (256, 256))
    images = load_images(normal_paths[batchID])
    resize_imgs = tf_resize_images(images, IMAGE_SIZE, CH)
    salt_pepper_noise_imgs = add_salt_pepper_noise(resize_imgs, 0.2, 0.004)
    gaussian_noise_imgs = add_gaussian_noise(resize_imgs)

    for iii in range(batch_size):
        # SAVE ALL IMAGES

        # Save resize images
        if not os.path.exists(path + 'normal256/' + lines[counter].split('/')[0]):
            os.makedirs(path + 'normal256/' + lines[counter].split('/')[0])
        fname = path + 'normal256/' + lines[counter] + '.png'
        print(fname)
        imsave(fname, resize_imgs[iii])
        # Save normal salt images
        if not os.path.exists(path + 'normals256_saltpepper/' + lines[counter].split('/')[0]):
            os.makedirs(path + 'normals256_saltpepper/' + lines[counter].split('/')[0])
        fname = path + 'normals256_saltpepper/' + lines[counter] + '.png'
        print(fname)
        imsave(fname, salt_pepper_noise_imgs[iii])
        # Save normal gauss images
        if not os.path.exists(path + 'normals256_gauss/' + lines[counter].split('/')[0]):
            os.makedirs(path + 'normals256_gauss/' + lines[counter].split('/')[0])
        fname = path + 'normals256_gauss/' + lines[counter] + '.png'
        print(fname)
        imsave(fname, gaussian_noise_imgs[iii])
        counter = counter + 1

'''
lines = [line.rstrip('\n') for line in open(path + 'flistID.txt')]
gt_paths = []
normal_paths = []
for i in range(10000):
    gt_paths.append(path + 'gt/' + lines[i] + '.mat')
    normal_paths.append(path + 'normals_png/' + lines[i] + '.png')
    #print(gt_paths[i])
    #print(normal_paths[i])
#print(gt_paths[1000])
normal_paths = np.array(normal_paths)
gt_paths = np.array(gt_paths)

counter = 0
batch_size = 5
#for ii in range(10000/batch_size):
for ii in range(338, 2000, 1):
    print(ii)
    batchID = inBatchIndex(ii,batch_size)
    gt = load_ground_truth(gt_paths[batchID],(256,256))

    for iii in range(batch_size):
        if not os.path.exists(path + 'gt256/' + lines[counter].split('/')[0]):
            os.makedirs(path + 'gt256/' + lines[counter].split('/')[0])
        fname = path + 'gt256/' + lines[counter] + '.png'
        print(fname)
        imsave(fname, gt[iii].reshape(256, 256))
        counter = counter + 1
'''

#plt.imshow(gt[0].reshape(256,256))
#plt.show()

#imsave('test.png',gt[0].reshape(256,256))
