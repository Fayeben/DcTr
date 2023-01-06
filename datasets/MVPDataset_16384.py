import torch
import numpy as np
import torch.utils.data as data
import h5py
import os
import torch.utils.data as data
import numpy as np
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import data_transforms
from .io import IO
import random
import os
import json
from .build import DATASETS
from utils.logger import *

@DATASETS.register_module()
class MVP_16384(data.Dataset):
    def __init__(self, config):
        if config.subset=="train":
            self.file_path = '/home/feiben/Point_Diffusion_Refinement/pointnet2/mvp_dataloader/data/mvp_dataset/mvp_train_input.h5'
            self.file_path_gt = '/home/feiben/Point_Diffusion_Refinement/pointnet2/mvp_dataloader/data/mvp_dataset/mvp_train_gt_16384pts.h5'
        elif config.subset=="val":
            self.file_path = '/home/feiben/Point_Diffusion_Refinement/pointnet2/mvp_dataloader/data/mvp_dataset/mvp_test_input.h5'
            self.file_path_gt = '/home/feiben/Point_Diffusion_Refinement/pointnet2/mvp_dataloader/data/mvp_dataset/mvp_test_gt_16384pts.h5'
        # the hidden test set below is only used for workshop competition
        elif config.subset=="test":
            self.file_path = '/home/feiben/Point_Diffusion_Refinement/pointnet2/mvp_dataloader/data/mvp_dataset/mvp_test_input.h5'
            self.file_path_gt = '/home/feiben/Point_Diffusion_Refinement/pointnet2/mvp_dataloader/data/mvp_dataset/mvp_test_gt_16384pts.h5'
        else:
            raise ValueError("ValueError prefix should be [train/val/test] ")

        self.prefix = config.subset

        input_file = h5py.File(self.file_path, 'r')
        input_file_gt = h5py.File(self.file_path_gt, 'r')
        self.input_data = np.array(input_file['incomplete_pcds'][()])
        self.novel_input_data = np.array((input_file['novel_incomplete_pcds'][()]))
        self.input_data = np.concatenate((self.input_data, self.novel_input_data), axis=0)
        self.novel_labels = np.array((input_file['novel_labels'][()]))
        self.labels = np.array((input_file['labels'][()]))
        print(self.input_data.shape)

        # if config.subset is not "test":
        self.gt_data = np.array(input_file_gt['complete_pcds'][()])
        self.novel_gt_data = np.array((input_file_gt['novel_complete_pcds'][()]))
        
        self.gt_data = np.concatenate((self.gt_data, self.novel_gt_data), axis=0)
        self.labels = np.concatenate((self.labels, self.novel_labels), axis=0)
        print(self.gt_data.shape, self.labels.shape)

        input_file.close()
        self.len = self.input_data.shape[0]
        cat_name = ['airplane', 'cabinet', 'car', 'chair', 'lamp', 'sofa', 'table', 'vessel',
                    'bed', 'bench', 'bookshelf', 'bus', 'guitar', 'motorbike', 'pistol', 'skateboard']
        self.cat_name = [cat_name[int(i)] for i in np.unique(self.labels)]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        partial = torch.from_numpy((self.input_data[index]))

        # if self.prefix is not "test":
        complete = torch.from_numpy((self.gt_data[index // 26]))
        label = (self.labels[index])

        cat_name1 = ['02691156', '02933112', '02958343', '03001627', '03636649', '04256520', '04379243', '04530566',
                    '02818832', '02828884', '02871439', '02924116', '03467517', '03790512', '03948459', '04225987']
        cat_name2 = cat_name1[int(label)]
        return cat_name2, label, (partial, complete)
        # else:
        #     return partial

