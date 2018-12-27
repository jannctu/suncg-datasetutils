import os
from scipy.io import loadmat
from scipy.misc import imread

import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize


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
        self.layout = os.listdir(self.SUNCG_BASE + 'gt/')
        self.IMAGE_TRAIN_PATH = os.path.join(self.SUNCG_BASE, 'renderings_ldr/')
        self.DEPTH_TRAIN_PATH = os.path.join(self.SUNCG_BASE, 'renderings_depth/')
        # self.NORMAL_TRAIN_PATH = os.path.join(self.SUNCG_BASE, 'normals/')
        # self.TEST_PATH = os.path.join(self.BSDS_BASE, 'renderings_depth/data/images/test/')
        # self.VALID_PATH = os.path.join(self.BSDS_BASE, 'BSDS500/data/images/val/')
        self.GROUND_TRUTH_TRAIN = os.path.join(self.SUNCG_BASE, 'gt/')

    # self.GROUND_TRUTH_TEST = os.path.join(self.BSDS_BASE, 'BSDS500/data/groundTruth/test/')
    # self.GROUND_TRUTH_VALID = os.path.join(self.BSDS_BASE, 'BSDS500/data/groundTruth/val/')

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

    def get_train(self):
        sequence = np.linspace(self.last_ID, self.last_ID + self.batch_size - 1, num=(self.batch_size))
        sequence = sequence.astype(np.int64)
        # print(sequence)
        layouts = np.array(self.layout)
        l = layouts[sequence]
        # print(l)

        file_ids, cnts, sgmnts = self.load_ground_truth(l)
        depth_paths = []
        for f_id in file_ids:
            # depth_paths.append(self.DEPTH_TRAIN_PATH + f_id.split('_')[0] + '/'+f_id.split('_')[1]+'_depth.png')
            depth_paths.append(self.IMAGE_TRAIN_PATH + f_id.split('_')[0] + '/' + f_id.split('_')[1] + '_mlt.png')

        depths = self.load_depths(depth_paths)
        self.last_ID = self.last_ID + self.batch_size
        return file_ids, cnts, sgmnts, depths

    def load_depths(self, depth_paths):
        processed_depths = []
        for dp in depth_paths:
            im = imread(dp)
            if self.target_size:
                im = resize(im, output_shape=self.target_size)

            # processed_depths.append(np.expand_dims(np.expand_dims(im,2), 0))
            processed_depths.append(np.expand_dims(im, 0))

        # print(processed_depths[0].shape)

        # plt.imshow(processed_depths[0].reshape(480,640))
        # plt.show()

        processed_depths = np.concatenate(processed_depths)

        return processed_depths

    def get_lastID(self):
        return self.last_ID

    def set_lastID(self, par):
        self.last_ID = par

    def get_layout(self):
        return self.layout