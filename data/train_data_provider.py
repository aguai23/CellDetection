from __future__ import print_function, division, absolute_import, unicode_literals
import cv2
import glob
import numpy as np


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

    def _load_data_and_label(self, test=False):

        data, label = self._next_data(test=test)
        labels = self._process_labels(label)
        train_data = self._process_data(data)
        nx = train_data.shape[1]
        ny = train_data.shape[0]
        return train_data.reshape(1, ny, nx, self.channels), labels.reshape(1, ny, nx, self.n_class)

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

    def __call__(self, n, test=False):

        train_data, labels = self._load_data_and_label()

        nx = train_data.shape[1]
        ny = train_data.shape[2]

        X = np.zeros((n, nx, ny, self.channels))
        Y = np.zeros((n, nx, ny, self.n_class))
        Y[0] = labels
        X[0] = train_data

        for i in range(1, n):
            train_data, labels = self._load_data_and_label(test=test)
            Y[i] = labels
            X[i] = train_data

        return X, Y


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

    def __init__(self, data, label, a_min=None, a_max=None, channels=1, n_class=2):
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
                 shuffle_data=True, n_class=2, data_augment=True):

        super().__init__(a_min, a_max)
        self.data_suffix = data_suffix
        self.mask_suffix = mask_suffix
        self.shuffle_data = shuffle_data
        self.n_class = n_class
        self.data_augment = data_augment
        self.data_files = self._find_data_files(search_path)

        # training data array
        self.data_array = []
        self.mask_array = []
        # mask data array
        self.test_data_array = []
        self.test_mask_array = []
        self.test_percent = 0.2
        self.data_index = -1
        self.test_index = -1

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

    def get_test_size(self):
        return len(self.test_data_array)

    def _load_data_from_file(self):
        test_index = int(len(self.data_files) * self.test_percent)

        for file_name in self.data_files[:test_index]:
            mask_name = file_name.replace(self.data_suffix, self.mask_suffix)
            data_array = self._load_file(file_name)
            mask_array = self._load_mask(mask_name)
            self.test_data_array.append(data_array)
            self.test_mask_array.append(mask_array)

        for file_name in self.data_files[test_index:]:
            mask_name = file_name.replace(self.data_suffix, self.mask_suffix)
            data_array = self._load_file(file_name)
            mask_array = self._load_mask(mask_name)
            self.data_array.append(data_array)
            self.mask_array.append(mask_array)
            if self.data_augment:
                self.augment_data(data_array, mask_array)

    def augment_data(self, data_array, mask_array):
        """
        data augmentation function
        :param data_array: the image data array
        :param mask_array: corresponding mask array
        :return:
        """
        self.data_array.append(np.rot90(data_array, 1, (0, 1)))
        self.mask_array.append(np.rot90(mask_array, 1, (0, 1)))
        self.data_array.append(np.rot90(data_array, 2, (0, 1)))
        self.mask_array.append(np.rot90(mask_array, 2, (0, 1)))
        self.data_array.append(np.rot90(data_array, 3, (0, 1)))
        self.mask_array.append(np.rot90(mask_array, 3, (0, 1)))

    def _load_file(self, path, dtype=np.float32):
        data = cv2.imread(path).astype(dtype)
        return data

    def _load_mask(self, path, dtype=np.bool):
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(dtype)

    def _next_data(self, test=False):

        if test:
            self.test_index += 1
            return self.test_data_array[self.test_index], self.test_mask_array[self.test_index]

        self.data_index += 1
        if self.data_index >= len(self.data_array):
            self.data_index = 0

        return self.data_array[self.data_index], self.mask_array[self.data_index]