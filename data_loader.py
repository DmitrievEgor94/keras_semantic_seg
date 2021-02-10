import random

import keras
import numpy as np
from skimage import io, transform, exposure
import cv2

PALLETE = {0: (255, 255, 255),  # Buildings
           1: (0, 0, 0)
           }  # Background


def convert_to_color(arr_2d, palette):
    """ Numeric labels to RGB-color encoding """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d


def convert_from_color(arr_3d, palette):
    """ RGB-color encoding to grayscale labels """
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for i, c in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d


class DataLoader(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, images_paths, labels_paths, batch_size=10, n_classes=2,
                 shuffle=True, size_of_crop=None, augmentation=True):
        'Initialization'
        # window_size format (n_h, n_w, n_channels)
        self.batch_size = batch_size
        self.labels = labels_paths
        self.list_images_paths = images_paths
        self.list_labels_paths = labels_paths
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.size_of_crop = size_of_crop
        self.augmentation = augmentation

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_images_paths) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_images_paths_temp = [self.list_images_paths[k] for k in indexes]
        list_labels_paths_temp = [self.list_labels_paths[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_images_paths_temp, list_labels_paths_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_images_paths))

        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def get_crop(self, image, label, crop_window):
        assert image.shape[0] == label.shape[0]
        assert image.shape[1] == label.shape[1]

        height, width = crop_window

        assert image.shape[0] >= height
        assert image.shape[1] >= width

        x = random.randint(0, image.shape[1] - width)
        y = random.randint(0, image.shape[0] - height)

        return image[y:y + height, x:x + width, :], label[y:y + height, x:x + width, :]

    def augment(self, image, label):
        will_flip, will_mirror, will_rotate, will_shear = False, False, False, False
        if random.random() < 0.5:
            will_flip = True
        if random.random() < 0.5:
            will_mirror = True
        # if random.random() < 0.5:
        #     will_rotate = True
        # if random.random() < 0.5:
        #     will_shear = True

        if will_flip:
            image = image[::-1, :, :]
            label = label[::-1, :, :]
        if will_mirror:
            image = image[:, ::-1, :]
            label = label[:, ::-1, :]
        # if will_rotate:
        #     if random.random() > 0.5:
        #         angle = 45
        #     else:
        #         angle = -45
        #     image = transform.rotate(image, angle)
        #     label = transform.rotate(label, angle)
        # if will_shear:
        #     shearing = transform.AffineTransform(shear=0.2)
        #     image = transform.warp(image, shearing)
        #     label = transform.warp(label, shearing)

        return image, label

    def __data_generation(self, labels_list_paths, images_list_paths):
        'Generates data containing batch_size samples'  # X : (n_samples, window_size)
        # Initialization
        X = np.empty((self.batch_size, *self.size_of_crop))
        Y = np.empty((self.batch_size, *(self.size_of_crop[0:2])), dtype=int)

        # Generate data
        for i, (image_path, label_path) in enumerate(zip(labels_list_paths, images_list_paths)):
            # Store sample
            image = 1 / 255 * cv2.imread(image_path)
            label = cv2.imread(label_path)

            image = image[..., ::-1]

            if self.size_of_crop is not None:
                image, label = self.get_crop(image, label, self.size_of_crop[0:2])

            if self.augmentation:
                image, label = self.augment(image, label)

            X[i] = image

            # Store class
            Y[i] = convert_from_color(label, PALLETE)

        return X, np.expand_dims(Y, axis=-1)

        # return X, keras.utils.to_categorical(Y, num_classes=self.n_classes)
