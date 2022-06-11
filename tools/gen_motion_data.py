import os
import numpy as np
from numpy.lib.format import open_memmap

sets = {
    'train', 'val'
}
# 'ntu/xview', 'ntu/xsub',  'kinetics'
datasets = {
    'ntu_300/xview', 'ntu_300/xsub'#,  'kinetics'
}

parts = {
    'joint', 'bone'
}
from tqdm import tqdm

for dataset in datasets:
    for set in sets:
        for part in parts:
            print(dataset, set, part)
            data = np.load('../data/{}/{}_data_{}.npy'.format(dataset, set, part))
            N, C, T, V, M = data.shape
            fp_sp = open_memmap(
                '../data/{}/{}_data_{}_motion.npy'.format(dataset, set, part),
                dtype='float32',
                mode='w+',
                shape=(N, 3, T, V, M))
            for t in tqdm(range(T - 1)):
 			
                fp_sp[:, :, t, :, :] = data[:, :, t + 1, :, :] - data[:, :, t, :, :]
                if dataset != 'kinetics':
                    fp_sp[:, 2, t, :, :] = data[:, 2, t + 1, :, :]
            fp_sp[:, :, T - 1, :, :] = 0
