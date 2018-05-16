from __future__ import print_function, division, absolute_import, unicode_literals

import cv2
import glob
import numpy as np
import os
from data.train_data_provider import BaseDataProvider


class ImageTestBaseProvider(BaseDataProvider):
    def __init__(self, test_path, data_suffix=".jpg", mask_suffix='_mask.jpg', is_dice=False,
                 a_min=None, a_max=None):
        super().__init__(a_min, a_max)
        self.is_dice = is_dice
        self.data_file = glob.glob(os.path.join(test_path, "*" + data_suffix))

        if not is_dice:
            self.mask_file = glob.glob(os.path.join(test_path, "*" + mask_suffix))
            self.data_file = [name for name in self.data_file if name not in self.mask_file]
            self.mask_file.sort()
        self.data_file.sort()

    @staticmethod
    def load_mask(path, d_type=np.bool):
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(d_type)

    @staticmethod
    def load_file(path, d_type=np.float32):
        return cv2.imread(path).astype(d_type)

    def _process_labels(self, label):
        if self.n_class == 2:
            nx = label.shape[1]
            ny = label.shape[0]
            labels = np.zeros((ny, nx, self.n_class), dtype=np.float32)
            labels[..., 1] = label
            labels[..., 0] = ~label
            return labels

        return label

    def _process_data(self, data):
        data = np.clip(np.fabs(data), self.a_min, self.a_max)
        data -= np.amin(data)
        data /= np.amax(data)
        return data

    def __getitem__(self, item):
        data = self.load_file(self.data_file[item], d_type=np.float32)
        print(self.data_file[item])
        data = self._process_data(data)
        data = data[np.newaxis, ...]
        if not self.is_dice:
            mask = self.load_mask(self.mask_file[item], d_type=np.bool)
            print(self.mask_file[item])
            mask = self._process_labels(mask)
            mask = mask[np.newaxis, ...]
            return data, mask
        else:
            return data

    def __len__(self):
        return len(self.data_file)


class ImageTestProvider(object):
    def __init__(self, test_path, batch_size=1, data_suffix=".jpg", mask_suffix='_mask.jpg', is_dice=False
                 , is_shuffle= False):
        super().__init__()
        self.ImageTestBaseProvider = ImageTestBaseProvider(test_path, data_suffix, mask_suffix, is_dice)
        self.len = int(len(self.ImageTestBaseProvider)/batch_size)
        self.batch_size = batch_size
        self.shuffle = is_shuffle
        self.sample = self._sample()
        self.is_dice = is_dice

    def _sample(self):
        list_i = list(range(len(self.ImageTestBaseProvider)))
        if self.shuffle:
            np.random.shuffle(list_i)
        list_sample = []
        index = 0
        for i in range(self.len):
            list_subsample = list_i[index:index+self.batch_size]
            index = index+self.batch_size
            list_sample.append(list_subsample)
        return list_sample

    def __getitem__(self, item):
        list_ = []
        if not self.is_dice:
            list_label = []
        for index in self.sample[item]:
            image = self.ImageTestBaseProvider[index]
            if not self.is_dice:
                list_.append(image[0])
                list_label.append(image[1])
            else:
                list_.append(image)
        if not self.is_dice:
            return np.concatenate(list_, axis=0), np.concatenate(list_label, axis=0)
        else:
            return np.concatenate(list_, axis=0)

    def __len__(self):
        return self.len
