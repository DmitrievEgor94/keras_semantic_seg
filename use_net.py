from hyperparameters_searchers.LinkNet.create_model import LinkNet
import os
import cv2
import numpy as np

folder_to_get_images = '/home/x/Dmitriev/dataset/People/test/data_2/'

folder_to_save_images ='/home/x/Dmitriev/Keras_seg/results/people/'

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

WINDOW_SIZE = [320,240,3]

net = LinkNet(1, input_shape=WINDOW_SIZE).get_model()
net.load_weights('/home/x/Dmitriev/Keras_seg/checkpoints/LinkNet/Google_2/LinkNet49-0.956.h5')

for image_name in os.listdir(folder_to_get_images):
    image = cv2.imread(folder_to_get_images+image_name)/255

    image = np.expand_dims(image, axis=0)

    predic_after_net = net.predict(image)[0,:,:,0]

    cv2.imwrite(folder_to_save_images+image_name, convert_to_color(predic_after_net>=0.5, palette))



