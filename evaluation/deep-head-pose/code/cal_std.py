import numpy as np
import os
import glob

npy_root = 'evaluation/deep-head-pose/code/output'
npy_paths = glob.glob(os.path.join("npy_root", '*.npy'))

yaw_std_list = []
pitch_std_list = []
roll_std_list = []
for npy in npy_paths:
    data = np.load(npy)
    yaw_values = data[:, 0, :]
    pitch_values = data[:, 1, :]
    roll_values = data[:, 2, :]

    yaw_std = np.mean(np.std(yaw_values, axis=0))
    pitch_std = np.mean(np.std(pitch_values, axis=0))
    roll_std = np.mean(np.std(roll_values, axis=0))

    yaw_std_list.append(yaw_std)
    pitch_std_list.append(pitch_std)
    roll_std_list.append(roll_std)

y = np.mean(yaw_std_list)
p = np.mean(pitch_std_list)
r = np.mean(roll_std_list)
print('Standard deviation of Yaw:', y)
print('Standard deviation of Pitch:', p)
print('Standard deviation of Roll:', r)
print('Overall standard deviation:', (y+p+r)/3)


