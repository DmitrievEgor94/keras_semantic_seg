import os
import sys

import cv2
import numpy as np
from segmentation_models import Unet
from skimage import io
from skimage.morphology import remove_small_holes, remove_small_objects

from models.Unet import make_unet

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from models.link_net.LinkNet import LinkNet

SIZE_OF_CROP = [256, 256, 3]

palette = {0: (255, 255, 255),  # Buildings
           1: (0, 0, 0)
           }


def convert_to_color(arr_2d, palette):
    """ Numeric labels to RGB-color encoding """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d


def sliding_window(top, step=10, window_size=(20, 20)):
    """ Slide a window_shape window across the image with a stride of step """
    for y in range(0, top.shape[0], step):
        if y + window_size[0] > top.shape[0]:
            y = top.shape[0] - window_size[0]
        for x in range(0, top.shape[1], step):
            if x + window_size[1] > top.shape[1]:
                x = top.shape[1] - window_size[1]
            yield y, x, window_size[1], window_size[0]


if __name__ == "__main__":
    net = LinkNet(num_classes=1, input_shape=SIZE_OF_CROP).get_model()
    # net = make_unet(SIZE_OF_CROP, 1)
    # net = Unet(backbone_name='vgg16')
    path_to_image = sys.argv[1]
    path_to_save = sys.argv[2]

    if not os.path.isfile(path_to_image):
        # print('Файла '+path_to_image+' не существует!')
        print('Завершение программы!')


    image = 1 / 255 * cv2.imread(path_to_image)
    image = image[:, :, ::-1]

    net.load_weights("checkpoints/LinkNet/Google/Linknet22-0.962.h5")

    prediction = np.zeros((image.shape[0:2]))

    image = image.reshape(1, *image.shape)

    for y, x, width, height in sliding_window(image[0], SIZE_OF_CROP[0], (256, 256)):
        predic_after_net = net.predict(image[:, y:y + height, x: x + width])[0,:,:,0]
        prediction[y:y + height, x: x + width] = np.round(predic_after_net)

    # prediction = np.argmax(prediction, axis=-1)
    prediction = convert_to_color(prediction, palette)
    prediction = remove_small_holes(remove_small_objects(prediction))

    cv2.imwrite(path_to_save, prediction * 255)
