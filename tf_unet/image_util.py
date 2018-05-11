# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.

'''
author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import cv2
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class BaseDataProvider(object):
    """
    Abstract base class for DataProvider implementation. Subclasses have to
    overwrite the `_next_data` method that load the next data and label array.
    This implementation automatically clips the data with the given min/max and
    normalizes the values to (0,1]. To change this behavoir the `_process_data`
    method can be overwritten. To enable some post processing such as data
    augmentation the `_post_process` method can be overwritten.

    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping

    """
    
    channels = 1
    n_class = 2

    def __init__(self, a_min=None, a_max=None):
        self.a_min = a_min if a_min is not None else -np.inf
        self.a_max = a_max if a_min is not None else np.inf

    def _load_data_and_label(self):
        if not self.is_testing:
            data, label = self._next_data()
            labels = self._process_labels(label)
        else:
            data = self._next_data()
     
        train_data = self._process_data(data)
        nx = train_data.shape[1]
        ny = train_data.shape[0]
        if not self.is_testing:
            return train_data.reshape(1, ny, nx, self.channels), labels.reshape(1, ny, nx, self.n_class)
        else:
            return  train_data.reshape(1, ny, nx, self.channels)
    
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
        # normalization
        data = np.clip(np.fabs(data), self.a_min, self.a_max)
        data -= np.amin(data)
        data /= np.amax(data)
        return data
    
    def _post_process(self, data, labels):
        """
        Post processing hook that can be used for data augmentation
        
        :param data: the data array
        :param labels: the label array
        """
        return data, labels
    
    def __call__(self, n):
        if not self.is_testing:
            train_data, labels = self._load_data_and_label()
        else:
            train_data = self._load_data_and_label()
        nx = train_data.shape[1]
        ny = train_data.shape[2]

        X = np.zeros((n, nx, ny, self.channels))
        if not self.is_testing:
            Y = np.zeros((n, nx, ny, self.n_class))
            Y[0] = labels
        X[0] = train_data

        for i in range(1, n):
            if not self.is_testing:
                train_data, labels = self._load_data_and_label()
                Y[i] = labels
            else:
                train_data = self._load_data_and_label()
            X[i] = train_data
        if not self.is_testing:
            return X, Y
        else:
            return X


class SimpleDataProvider(BaseDataProvider):
    """
    A simple data provider for numpy arrays. 
    Assumes that the data and label are numpy array with the dimensions
    data `[n, X, Y, channels]`, label `[n, X, Y, classes]`. Where
    `n` is the number of images, `X`, `Y` the size of the image.

    :param data: data numpy array. Shape=[n, X, Y, channels]
    :param label: label numpy array. Shape=[n, X, Y, classes]
    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping
    :param channels: (optional) number of channels, default=1
    :param n_class: (optional) number of classes, default=2
    
    """
    
    def __init__(self, data, label, a_min=None, a_max=None, channels=1, n_class = 2):
        super(SimpleDataProvider, self).__init__(a_min, a_max)
        self.data = data
        self.label = label
        self.file_count = data.shape[0]
        self.n_class = n_class
        self.channels = channels

    def _next_data(self):
        idx = np.random.choice(self.file_count)
        return self.data[idx], self.label[idx]


class ImageDataProvider(BaseDataProvider):
    """
    Generic data provider for images, supports gray scale and colored images.
    Assumes that the data images and label images are stored in the same folder
    and that the labels have a different file suffix 
    e.g. 'train/fish_1.tif' and 'train/fish_1_mask.tif'

    Usage:
    data_provider = ImageDataProvider("..fishes/train/*.jpg")
        
    :param search_path: a glob search pattern to find all data and label images
    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping
    :param data_suffix: suffix pattern for the data images. Default '.tif'
    :param mask_suffix: suffix pattern for the label images. Default '_mask.tif'
    :param shuffle_data: if the order of the loaded file path should be randomized. Default 'True'
    :param channels: (optional) number of channels, default=1
    :param n_class: (optional) number of classes, default=2
    
    """
    
    def __init__(self, search_path, a_min=None, a_max=None, data_suffix=".jpg", mask_suffix='_mask.jpg',
                 shuffle_data=True, n_class=2, is_testing=False,data_augment=True):
   
        super().__init__(a_min, a_max)
        self.data_suffix = data_suffix
        self.mask_suffix = mask_suffix
        self.shuffle_data = shuffle_data
        self.n_class = n_class
        self.data_augment = data_augment
        self.data_files = self._find_data_files(search_path)
        self.is_testing = is_testing
        self.data_array = []
        self.mask_array = []
        self.data_index = -1

        if self.shuffle_data:
            np.random.shuffle(self.data_files)
        
        assert len(self.data_files) > 0, "No training files"
        print("Number of files used: %s" % len(self.data_files))

        self._load_data_from_file()

        img = self._load_file(self.data_files[0])
        self.channels = 1 if len(img.shape) == 2 else img.shape[-1]
        
    def _find_data_files(self, search_path):
        all_files = glob.glob(search_path)
        return [name for name in all_files if self.data_suffix in name and self.mask_suffix not in name]

    def _load_data_from_file(self):
        for file_name in self.data_files:
            mask_name = file_name.replace(self.data_suffix, self.mask_suffix)
            data_array = self._load_file(file_name)
            mask_array = self._load_mask(mask_name)
            self.data_array.append(data_array)
            self.mask_array.append(mask_array)
            if self.data_augment:
                self.data_array.append(np.rot90(data_array, 1, (0, 1)))
                self.mask_array.append(np.rot90(mask_array, 1, (0, 1)))
                self.data_array.append(np.rot90(data_array, 2, (0, 1)))
                self.mask_array.append(np.rot90(mask_array, 2, (0, 1)))
                self.data_array.append(np.rot90(data_array, 3, (0, 1)))
                self.mask_array.append(np.rot90(mask_array, 3, (0, 1)))

    def _load_file(self, path, dtype=np.float32):
        data = cv2.imread(path).astype(dtype)
        # average = np.average(data, axis=(0,1))
        # std = np.std(data, axis=(0,1), dtype=dtype)
        # data = (data - average) / std
        return data
        # return np.squeeze(cv2.imread(image_name, cv2.IMREAD_GRAYSCALE))

    def _load_mask(self, path, dtype=np.bool):
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(dtype)
        
    def _next_data(self):
        self.data_index += 1
        if self.data_index >= len(self.data_array):
            self.data_index = 0
        if not self.is_testing:
            return self.data_array[self.data_index], self.mask_array[self.data_index]
        else:
            return self.data_array[self.data_index]
        
