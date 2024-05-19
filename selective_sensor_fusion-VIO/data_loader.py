import torch.utils.data as data
import numpy as np
from PIL import Image
from pathlib import Path
import cv2

class Args:
    def __init__(self):
        self.data = 'E:\\SLAM\\Dataset\\KITTI_RAW_Synced'
        self.sequence_length = 3
        self.rotation_mode = 'euler'
        self.workers = 4
        self.epochs = 100
        self.epoch_size = 0
        self.batch_size = 64
        self.lr = 1e-4
        self.momentum = 0.9
        self.beta = 0.999
        self.weight_decay = 0
        self.print_freq = 50
        self.log_summary = 'progress_log_summary.csv'

class KITTI_Loader(data.Dataset):

    def __init__(self, root, train=0, sequence_length=3, transform=None):
        self.root = Path(root)
        if train == 0:
            scene_list_path = self.root / 'train.txt'
        if train == 1:
            scene_list_path = self.root/'val.txt'
        if train == 2:
            scene_list_path = self.root/'test.txt'

        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.transform = transform
        self.sequence_length = sequence_length

        if (train == 0) or (train == 1):
            self.crawl_folders(sequence_length)
        else:
            self.crawl_test_folders()

    def crawl_folders(self, sequence_length):
        sequence_set = []
        demi_length = (sequence_length-1)//2
        shifts = list(range(-demi_length, demi_length + 1))
        for scene in self.scenes:
            print(scene)
            # load data from left camera
            imgs_l = sorted(scene.glob('*.jpg'))
            imus_l = sorted(scene.glob('*.txt'))
            poses_l = np.genfromtxt(scene / 'poses.txt').astype(np.float64).reshape(-1, 3, 4)

            if len(imgs_l) < sequence_length:
                continue

            for i in range(demi_length, len(imgs_l) - demi_length):
                sample = {'imgs': [], 'poses': [], 'imus': []}
                for j in shifts:
                    sample['imgs'].append(imgs_l[i + j])
                    sample['poses'].append(poses_l[i + j, :, :])
                    sample['imus'].append(np.genfromtxt(imus_l[i + j]).astype(np.float32).reshape(-1, 6))
                sequence_set.append(sample)

        self.samples = sequence_set

    def crawl_test_folders(self):

        sequence_set = []
        for scene in self.scenes:

            # load data from left camera
            imgs_l = sorted(scene.glob('*.jpg'))
            imus_l = sorted(scene.glob('*.txt'))
            poses_l = np.genfromtxt(scene / 'poses.txt').astype(np.float64).reshape(-1, 3, 4)

            sample = {'imgs': [], 'poses': [], 'imus': []}

            for i in range(len(imgs_l)):
                sample['imgs'].append(imgs_l[i])
                sample['poses'].append(poses_l[i, :, :])
                sample['imus'].append(np.genfromtxt(imus_l[i]).astype(np.float32).reshape(-1, 6))
            sequence_set.append(sample)

        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]

        imgs = [cv2.imread(str(img)).astype(np.float32) for img in sample['imgs']]
        poses = [pose for pose in sample['poses']]
        imus = [imu for imu in sample['imus']]

        if self.transform is not None:
            imgs = self.transform(imgs)

        return imgs, imus, poses

    def __len__(self):
        return len(self.samples)