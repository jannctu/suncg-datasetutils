import os
from scipy.io import loadmat
from scipy.misc import imread
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import sys
sys.path.insert(0, '/media/commlab/TenTB/jan/rgbd-processor-python/')
from camera import processCamMat
from getHHAImg import *


class SUNCG(object):

    def __init__(self, path_to=None, target_size=None, masks_to_binary=True, batch_size=10):
        if not path_to:
            print(path_to)
        else:
            self.SUNCG_BASE = path_to

        print(self.SUNCG_BASE)
        self.target_size = target_size
        self.last_ID = 0
        self.batch_size = batch_size
        self.masks_to_binary = masks_to_binary
        camAddr = '/media/commlab/TenTB/jan/rgbd-processor-python/imgs/intrinsics.txt'
        with open(camAddr, 'r') as camf:
            self.cameraMatrix = processCamMat(camf.readlines())


        #self.layout = os.listdir(self.SUNCG_BASE + 'renderings_depth/')
        self.layout = os.listdir(self.SUNCG_BASE + 'gt/')
        self.IMAGE_TRAIN_PATH = os.path.join(self.SUNCG_BASE, 'renderings_ldr/')
        self.DEPTH_TRAIN_PATH = os.path.join(self.SUNCG_BASE, 'renderings_depth/')
        self.NORMAL_TRAIN_PATH = os.path.join(self.SUNCG_BASE, 'normals/')
        self.HHA_TRAIN_PATH = os.path.join(self.SUNCG_BASE, 'hha/')
        # self.TEST_PATH = os.path.join(self.BSDS_BASE, 'renderings_depth/data/images/test/')
        # self.VALID_PATH = os.path.join(self.BSDS_BASE, 'BSDS500/data/images/val/')
        self.GROUND_TRUTH_TRAIN = os.path.join(self.SUNCG_BASE, 'gt/')

    # self.GROUND_TRUTH_TEST = os.path.join(self.BSDS_BASE, 'BSDS500/data/groundTruth/test/')
    # self.GROUND_TRUTH_VALID = os.path.join(self.BSDS_BASE, 'BSDS500/data/groundTruth/val/')

    def buildFlistDepth(self,fpath):
        layouts = self.layout
        f = open(fpath, "w")

        for layout in layouts:
            flists = os.listdir(self.IMAGE_TRAIN_PATH + layout + '/')
            for fn in flists:
                f.write(layout+'/'+fn.split('_')[0]+"\n")
                print(layout+'/'+fn.split('_')[0])
        f.close()


    def load_ground_truth(self, layouts):
        file_id = []
        cnts = []
        sgmnts = []

        for layout in layouts:
            flist = os.listdir(self.GROUND_TRUTH_TRAIN + layout + '/')
            # print(flist[0])
            # quit()
            file_name = flist[0].split('.')[0]
            gt = loadmat(self.GROUND_TRUTH_TRAIN + layout + '/' + file_name + '.mat')
            gt = gt['groundTruth'][0]
            for annotator in gt:
                contours = annotator[0][0][1]  # 1-> contours
                segments = annotator[0][0][0]  # 0 -> segments
                # print(np.unique(contours))
                # plt.imshow(contours)
                # plt.show()

                if self.target_size:
                    contours = resize(contours.astype(float), output_shape=self.target_size)
                    segments = resize(segments, output_shape=self.target_size)

                if self.masks_to_binary:
                    contours[contours > 0] = 1

                file_id.append(layout + '_' + file_name)
                cnts.append(contours)
                sgmnts.append(segments)

        cnts = np.concatenate([np.expand_dims(a, 0) for a in cnts])
        sgmnts = np.concatenate([np.expand_dims(a, 0) for a in sgmnts])
        cnts = cnts[..., np.newaxis]
        sgmnts = sgmnts[..., np.newaxis]
        return file_id, cnts, sgmnts

    def sixteen_to_eight(self,im):
        im = np.float32(im)
        # quit()
        im *= (255.0 / float(im.max()))
        return (im).astype('uint8')

    def get_train(self):
        sequence = np.linspace(self.last_ID, self.last_ID + self.batch_size - 1, num=(self.batch_size))
        sequence = sequence.astype(np.int64)
        # print(sequence)
        layouts = np.array(self.layout)
        l = layouts[sequence]
        # print(l)

        file_ids, cnts, sgmnts = self.load_ground_truth(l)
        depth_paths = []
        images_paths = []
        for f_id in file_ids:
            depth_paths.append(self.DEPTH_TRAIN_PATH + f_id.split('_')[0] + '/'+f_id.split('_')[1]+'_depth.png')
            images_paths.append(self.IMAGE_TRAIN_PATH + f_id.split('_')[0] + '/' + f_id.split('_')[1] + '_mlt.png')

        depths,hha = self.load_depths(depth_paths,False)
        #depths = []
        #hha = []
        #images = self.load_images(images_paths)
        images = []
        self.last_ID = self.last_ID + self.batch_size
        return file_ids, cnts, sgmnts, depths,hha, images

    def load_images(self,image_paths):
        processed_images = []
        for ip in image_paths:
            im = imread(ip)
            if self.target_size:
                im = resize(im, output_shape=self.target_size)

            processed_images.append(np.expand_dims(im, 0))
        processed_images = np.concatenate(processed_images)
        return processed_images

    def load_depths(self, depth_paths,hhaF=False):
        processed_depths = []
        processed_hha = []
        for dp in depth_paths:
            im = imread(dp)
            if self.target_size:
                im = resize(im, output_shape=self.target_size)

            if hhaF:
                missingMask = (im == 0)
                HHA = getHHAImg(im, missingMask, self.cameraMatrix)
                processed_hha.append(np.expand_dims(HHA, 0))

            im = self.sixteen_to_eight(im)
            processed_depths.append(np.expand_dims(np.expand_dims(im,2), 0))
            #processed_depths.append(np.expand_dims(im, 0))

        # print(processed_depths[0].shape)
        # plt.imshow(processed_depths[0].reshape(480,640))
        # plt.show()
        processed_depths = np.concatenate(processed_depths)
        if hhaF:
            processed_hha = np.concatenate(processed_hha)

        return processed_depths, processed_hha

    def get_lastID(self):
        return self.last_ID

    def set_lastID(self, par):
        self.last_ID = par

    def get_layout(self):
        return self.layout