import os
import glob
from dataset import common
import numpy as np
import torch.utils.data as data
import torch

class SRData(data.Dataset):
    def __init__(self, opt, train=True):
        self.opt = opt
        self.train = train
        self.scale = self.opt['scale']
        self.arr = np.fromfile('arr.dat', dtype=int)  # (3200,)
        self.arr_resume = np.arange(1400, 1600)
        np.random.shuffle(self.arr_resume)

        data_range = [r.split('-') for r in self.opt['datasets']['data_range'].split('/')]  # ex: self.data_range: 1-400/401-432
        data_range = data_range[0] if train else data_range[1]
        self.begin, self.end = list(map(lambda x: int(x), data_range))

        self._set_filesystem(self.opt['datasets']['dir_data_root'])  # opt.dir_data_root: /home/xxx/xxx/data/

        self.images_hr_clean, self.images_lr_noisy, self.images_hr_noisy, self.images_lr_clean = self._scan()  # get file list

        if train:
            n_patches = self.opt['datasets']['batch_size'] * self.opt['datasets']['test_every']  # opt.batch_size: 16, opt.test_every: 1000
            self.repeat = max(n_patches // len(self.images_hr_clean), 1) if len(self.images_hr_clean) > 0 else 0

    # Below functions as used to prepare images
    def _scan(self):
        names_hr_clean = sorted(glob.glob(os.path.join(self.opt['datasets']['dir_hr_clean'], '*.dat')))
        names_lr_noisy = sorted(glob.glob(os.path.join(self.opt['datasets']['dir_lr_noisy'], '*.dat')))
        names_hr_noisy = sorted(glob.glob(os.path.join(self.opt['datasets']['dir_hr_noisy'], '*.dat')))
        names_lr_clean = sorted(glob.glob(os.path.join(self.opt['datasets']['dir_lr_clean'], '*.dat')))

        names_hr_clean = np.array(names_hr_clean)[self.arr]  # shuffle
        names_hr_clean = names_hr_clean[self.begin - 1: self.end]

        names_lr_noisy = np.array(names_lr_noisy)[self.arr]
        names_lr_noisy = names_lr_noisy[self.begin - 1: self.end]

        names_hr_noisy = np.array(names_hr_noisy)[self.arr]
        names_hr_noisy = names_hr_noisy[self.begin - 1: self.end]

        names_lr_clean = np.array(names_lr_clean)[self.arr]
        names_lr_clean = names_lr_clean[self.begin - 1: self.end]

        return names_hr_clean, names_lr_noisy, names_hr_noisy, names_lr_clean

    def _set_filesystem(self, dir_data_root):
        self.root_path = dir_data_root
        self.dir_hr_clean = self.opt['datasets']['dir_hr_clean']
        self.dir_lr_noisy = self.opt['datasets']['dir_lr_noisy']
        self.dir_hr_noisy = self.opt['datasets']['dir_hr_noisy']
        self.dir_lr_clean = self.opt['datasets']['dir_lr_clean']

    def __getitem__(self, idx):
        lr_noisy, lr_clean, hr_noisy, hr_clean, filename = self._load_file(idx)

        pair, params = common.normal(lr_noisy, lr_clean, hr_noisy, hr_clean)
        lr_noisy_normal, lr_clean_normal, hr_noisy_normal, hr_clean_normal = pair[0], pair[1], pair[2], pair[3]

        lr_noisy_normal = np.expand_dims(lr_noisy_normal, axis=0)
        lr_clean_normal = np.expand_dims(lr_clean_normal, axis=0)
        hr_noisy_normal = np.expand_dims(hr_noisy_normal, axis=0)
        hr_clean_normal = np.expand_dims(hr_clean_normal, axis=0)

        lr_noisy_normal = torch.from_numpy(lr_noisy_normal).float()
        lr_clean_normal = torch.from_numpy(lr_clean_normal).float()
        hr_noisy_normal = torch.from_numpy(hr_noisy_normal).float()
        hr_clean_normal = torch.from_numpy(hr_clean_normal).float()

        return (lr_noisy_normal, lr_clean_normal, hr_noisy_normal, hr_clean_normal), filename, params

    def __len__(self):
        if self.train:
            return len(self.images_hr_clean) * self.repeat
        else:
            return len(self.images_hr_clean)

    def _load_file(self, idx):
        idx = idx % len(self.images_hr_clean) if self.train else idx
        f_hr_clean = self.images_hr_clean[idx]
        f_lr_noisy = self.images_lr_noisy[idx]
        f_hr_noisy = self.images_hr_noisy[idx]
        f_lr_clean = self.images_lr_clean[idx]

        filename, _ = os.path.splitext(os.path.basename(f_lr_noisy))  # without suffix

        lr_noisy = np.fromfile(f_lr_noisy, dtype=np.float32)
        hr_clean = np.fromfile(f_hr_clean, dtype=np.float32)
        hr_noisy = np.fromfile(f_hr_noisy, dtype=np.float32)
        lr_clean = np.fromfile(f_lr_clean, dtype=np.float32)

        lr_noisy = lr_noisy.reshape((128, 128))
        lr_clean = lr_clean.reshape((128, 128))
        hr_clean = hr_clean.reshape((256, 256))
        hr_noisy = hr_noisy.reshape((256, 256))

        lr_noisy = np.rot90(lr_noisy, 3)
        lr_clean = np.rot90(lr_clean, 3)
        hr_clean = np.rot90(hr_clean, 3)
        hr_noisy = np.rot90(hr_noisy, 3)

        return lr_noisy, lr_clean, hr_noisy, hr_clean, filename

    def get_patch(self, lr_noisy, lr_clean, hr_noisy, hr_clean):
        scale = self.scale
        if self.train:
            lr_noisy, lr_clean, hr_noisy, hr_clean = common.get_patch(
                lr_noisy, lr_clean, hr_noisy, hr_clean,
                patch_size=self.opt['datasets']['patch_size'],
                scale=scale
            )
            lr_noisy, lr_clean, hr_noisy, hr_clean = common.augment(lr_noisy, lr_clean, hr_noisy, hr_clean)

        return lr_noisy, lr_clean, hr_noisy, hr_clean

    def get_arr(self):
        names_hr = sorted(glob.glob(os.path.join(self.dir_hr_clean, '*.dat')))
        l = len(names_hr)

        arr = np.arange(l)
        np.random.shuffle(arr)
        return arr

