import glob
import os
from scipy.io import loadmat, savemat
import numpy as np

mat_path = 'datasets/motion_wav2lip_train'
mats = sorted(glob.glob(os.path.join(mat_path, '*.mat')))

all_yaw = []
all_pitch = []
all_roll = []
all_t = []
all_exp = []
all_scale = []
all_kp = []
i =0
for mat in mats:
    i+=1
    print(i)
    yaw = loadmat(os.path.join(mat))['yaw']
    pitch = loadmat(os.path.join(mat))['pitch']
    roll = loadmat(os.path.join(mat))['roll']
    t = loadmat(os.path.join(mat))['t']
    exp = loadmat(os.path.join(mat))['exp'].reshape([-1, 63])
    scale = loadmat(os.path.join(mat))['scale']
    kp = loadmat(os.path.join(mat))['kp'].reshape([-1, 63])

    all_yaw.append(yaw)
    all_pitch.append(pitch)
    all_roll.append(roll)
    all_t.append(t)
    all_exp.append(exp)
    all_scale.append(scale)
    all_kp.append(kp)

all_yaw = np.concatenate(all_yaw, 0)
yaw_mean = np.mean(all_yaw, axis=0)
yaw_std = np.std(all_yaw, axis=0)
yaw_std[yaw_std == 0] = 1

all_pitch = np.concatenate(all_pitch, 0)
pitch_mean = np.mean(all_pitch, axis=0)
pitch_std = np.std(all_pitch, axis=0)
pitch_std[pitch_std == 0] = 1

all_roll = np.concatenate(all_roll, 0)
roll_mean = np.mean(all_roll, axis=0)
roll_std = np.std(all_roll, axis=0)
roll_std[roll_std == 0] = 1

all_t = np.concatenate(all_t, 0)
t_mean = np.mean(all_t, axis=0)
t_std = np.std(all_t, axis=0)
t_std[t_std == 0] = 1

all_exp = np.concatenate(all_exp, 0)
exp_mean = np.mean(all_exp, axis=0)
exp_std = np.std(all_exp, axis=0)
exp_std[exp_std == 0] = 1

all_scale = np.concatenate(all_scale, 0)
scale_mean = np.mean(all_scale, axis=0)
scale_std = np.std(all_scale, axis=0)
scale_std[scale_std == 0] = 1

all_kp = np.concatenate(all_kp, 0)
kp_mean = np.mean(all_kp, axis=0)
kp_std = np.std(all_kp, axis=0)
kp_std[kp_std == 0] = 1

np.savez('norm.npz', yaw_mean=yaw_mean, yaw_std=yaw_std, pitch_mean=pitch_mean, pitch_std=pitch_std,
         roll_mean=roll_mean, roll_std=roll_std, t_mean=t_mean, t_std=t_std, exp_mean=exp_mean, exp_std=exp_std,
         scale_mean=scale_mean, scale_std=scale_std, kp_mean=kp_mean, kp_std=kp_std)
