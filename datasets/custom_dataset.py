import torch
from torch.utils.data import Dataset
import os
import numpy as np
import random
from scipy.io import loadmat, savemat

class Landmarker(Dataset):

    def __init__(self, split):
        assert split in ('train', 'val')
        self.split = split
        vox1_train_path = 'datasets/motion_wav2lip_train'
        self.vox1_train_file_paths = sorted([os.path.join(vox1_train_path, f) for f in os.listdir(vox1_train_path)])
        vox1_val_path = 'datasets/motion_wav2lip_val'
        self.vox1_val_file_paths = sorted([os.path.join(vox1_val_path, f) for f in os.listdir(vox1_val_path)])

        if self.split == 'train':
            self.file_paths = self.vox1_train_file_paths
        else:
            self.file_paths = self.vox1_val_file_paths

        self.norm_info = dict(
            np.load(r'dataset_process/norm.npz'))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        fp = self.file_paths[idx]
        arr = loadmat(fp)

        yaw = (arr['yaw']-self.norm_info['yaw_mean'])/self.norm_info['yaw_std']
        pitch = (arr['pitch']-self.norm_info['pitch_mean'])/self.norm_info['pitch_std']
        roll = (arr['roll']-self.norm_info['roll_mean'])/self.norm_info['roll_std']
        t = (arr['t']-self.norm_info['t_mean'])/self.norm_info['t_std']
        exp = (arr['exp'].reshape([-1, 63])-self.norm_info['exp_mean'])/self.norm_info['exp_std']
        scale = (arr['scale']-self.norm_info['scale_mean'])/self.norm_info['scale_std']
        standard_kp = (arr['kp'].reshape([-1, 63])-self.norm_info['kp_mean'])/self.norm_info['kp_std']
        kps = np.concatenate([scale, yaw, pitch, roll, t, exp], -1)

        aud_feat = arr['aud_feat']

        sequence_name = os.path.split(fp)[-1]

        seq_nums = 64
        kps = np.expand_dims(kps, -1)
        kps = np.concatenate([kps, kps, kps], -1)

        min_size = min(kps.shape[0]-1, aud_feat.shape[0]- 2)
        kps = kps[:min_size]
        aud_feat = aud_feat[:min_size]

        num_frame = len(kps)

        rand_frame_idx = random.randint(0, num_frame-1-seq_nums)

        ref_keypoint = torch.Tensor(kps[rand_frame_idx])
        ref_standard_kp = torch.cat([torch.zeros([7, ]), torch.Tensor(standard_kp[rand_frame_idx])], 0)

        sequence_keypoints = torch.Tensor(kps[rand_frame_idx+1:rand_frame_idx+1+seq_nums])

        aud_feat = torch.Tensor(aud_feat[rand_frame_idx+1:rand_frame_idx+seq_nums+1])

        data= {'frame_number': rand_frame_idx,
                'sequence_name': sequence_name,
                'sequence_category': 0,
                'sequence_keypoints': sequence_keypoints,
                'ref_keypoint': ref_keypoint,
                'ori_keypoint': ref_standard_kp,
                'aud_feat': aud_feat}
        return data

