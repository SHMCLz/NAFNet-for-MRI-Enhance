# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
from torch.utils import data as data
from torchvision.transforms.functional import normalize

# from basicsr.data.data_util import (paired_paths_from_folder,
#                                     paired_paths_from_lmdb,
#                                     paired_paths_from_meta_info_file)
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding
import numpy as np
import os
import cv2

import imageio

import nibabel as nib

def read_niifile(niifilepath):
    try:
        img = nib.load(niifilepath)
        img_fdata = img.get_fdata()
        return img_fdata
    except:
        print('load error ... ', niifilepath)
        img_fdata = np.ones((5, 5, 5))
    return img_fdata

# 测试集患者id列表
test_ids = [
]

# 患者敏感信息，样例:
# 11111111,./MRI3/b1400/11111111/d_slice_0005.nii.gz,./MRI3/b700/11111111/c_slice_0005.nii.gz
# 11111111,./MRI3/b1400/11111111/d_slice_0006.nii.gz,./MRI3/b700/11111111/c_slice_0006.nii.gz
# 22222222,./MRI3/b1400/22222222/d_slice_00010.nii.gz,./MRI3/b700/22222222/c_slice_00010.nii.gz
#
datas = '''
'''

class PairedMRI3UpqualityTestDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedMRI3UpqualityDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        # self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.is_test = opt['is_test']
        self.center_crop = opt.get('center_crop', False)
        self.random_crop = opt.get('random_crop', False)


        # self.dataroot = opt['img_list']

        id_and_labels = []

        # import refile
        # with refile.smart_open(opt['id_and_labels'], 'r') as f:
        lines = datas

        self.root = opt['dataroot']

        self.hq_paths = []
        self.lq_paths = []
        self.ids = []

        for each_line in lines.split('\n'):
            if len(each_line) < 10:
                continue
            each_id, hq_path, lq_path = each_line.split(',')
            if self.is_test == (int(each_id) in test_ids):
                self.hq_paths.append(os.path.join(self.root, hq_path.strip('\'')))
                self.lq_paths.append(os.path.join(self.root, lq_path.strip('\'')))
                self.ids.append(int(each_id))

        self.resize_shape = opt['resize'] if 'resize' in opt else None
        # self.gt_size = opt['gt_size']

    def resize(self, imgs, s):
        results = []
        for img in imgs:
            h, w = img.shape
            img_after = cv2.resize(img, (s, s))
            # if c == 1:
            img_after = img_after[:, :, np.newaxis]
            results.append(img_after)
        return results

    def __getitem__(self, index):
        hq_path = self.hq_paths[index]
        lq_path = self.lq_paths[index]

        hq = read_niifile(hq_path).astype(np.float32)
        lq = read_niifile(lq_path).astype(np.float32)


        hq = hq / hq.max()
        lq = lq / lq.max()

        hq, lq = self.resize([hq, lq], self.resize_shape)



        if self.opt['phase'] == 'train':
            hq, lq = augment([hq, lq], self.opt['use_flip'], self.opt['use_rot'])

            if self.random_crop:
                gt_size = self.opt['gt_size']
                hq, lq= paired_random_crop(hq, lq, gt_size, 1, 'gt_path')
                # print('hq .. lq .. ', hq.shape, lq.shape)

        if self.center_crop:
            t = self.resize_shape
            hq, lq = hq[t//4:-t//4, t//4:-t//4, :], lq[t//4:-t//4, t//4:-t//4, :]

        hq, lq = img2tensor(
            [hq, lq], bgr2rgb=False, float32=True)


        return {
            'lq': lq,
            'gt': hq,
            'lq_path': f'{self.ids[index]}-{index}',
            'gt_path': hq_path
        }


    def __len__(self):
        return len(self.ids)
        # return len(self.paths)

