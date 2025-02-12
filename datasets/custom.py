import random
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from Register import Registers
from datasets.base import ImagePathDataset
from datasets.utils import get_image_paths_from_dir
from PIL import Image
import cv2
import os


@Registers.datasets.register_with_name('custom_aligned')
class CustomAlignedDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        image_paths_ori = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, f'{stage}/B'))
        image_paths_cond = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, f'{stage}/A'))
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal

        self.imgs_ori = ImagePathDataset(image_paths_ori, self.image_size, flip=self.flip, to_normal=self.to_normal)
        self.imgs_cond = ImagePathDataset(image_paths_cond, self.image_size, flip=self.flip, to_normal=self.to_normal)

    def __len__(self):
        return len(self.imgs_ori)

    def __getitem__(self, i):
        return self.imgs_ori[i], self.imgs_cond[i]

##########
# synthrad dataset
# (for using original nifti file)
##########
    
import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, random_split

@Registers.datasets.register_with_name('synthrad_dataset')
class SynthradDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.brain_dir = dataset_config.dataset_path  
        self.ratio = (0.8, 0.1, 0.1)  # (train, val, test)
        self.stage = stage
        self.axis = 2
        self.randomseed = 42
        self.image_size = 128  

        # Load CBCT, CT data
        self.brain_files = [f for f in os.listdir(self.brain_dir) if f != 'overview']

        # fix random seed
        np.random.seed(self.randomseed)
        torch.manual_seed(self.randomseed)

        # Train/Val/Test Split
        total_cases = len(self.brain_files)
        train_size = int(self.ratio[0] * total_cases)
        val_size = int(self.ratio[1] * total_cases)
        test_size = total_cases - train_size - val_size

        shuffled_files = np.random.permutation(self.brain_files)  
        self.train_files = shuffled_files[:train_size]
        self.val_files = shuffled_files[train_size:train_size + val_size]
        self.test_files = shuffled_files[train_size + val_size:]

        # 현재 stage에 맞는 데이터셋 선택
        if self.stage == 'train':
            self.brain_files = self.train_files
        elif self.stage == 'val':
            self.brain_files = self.val_files
        elif self.stage == 'test':
            self.brain_files = self.test_files
        else:
            raise ValueError(f"Invalid stage: {self.stage}")

    def __len__(self):
        return len(self.brain_files) * 60 #60 slices per case
    
    def __getitem__(self, idx):
        sample_idx = idx // 60  # case ID
        slice_idx = idx % 60    # 해당 환자의 몇 번째 슬라이스인지

        sample_folder = self.brain_files[sample_idx]
        sample_dir = os.path.join(self.brain_dir, sample_folder)

        cbct_path = os.path.join(sample_dir, 'cbct.nii.gz')
        ct_path = os.path.join(sample_dir, 'ct.nii.gz')

        cbct_image = nib.load(cbct_path).get_fdata()
        ct_image = nib.load(ct_path).get_fdata()

        cbct_slice = np.take(cbct_image, slice_idx, axis=self.axis)
        ct_slice = np.take(ct_image, slice_idx, axis=self.axis)
       
        # transform : resize, totensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.image_size, self.image_size))
        ])
        
        cbct_slice = transform(cbct_slice)
        ct_slice = transform(ct_slice)
        
        return cbct_slice, ct_slice
    
