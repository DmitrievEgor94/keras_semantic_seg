import json
import os
from collections import OrderedDict
from operator import itemgetter

import cv2
import keras
import numpy as np
from keras.optimizers import SGD
from skimage import io

from data_loader import DataLoader, convert_to_color, convert_from_color
from models.link_net.LinkNet import LinkNet

BATCH_SIZE = 32
SIZE_OF_CROP = [256, 256, 3]  # size of crop of image [height, width, channels]
CLASSES = ['Buildings', 'Background']
N_CLASSES = len(CLASSES) - 1

palette = {0: (255, 255, 255),  # Buildings
           1: (0, 0, 0)
           }  # Background

DATA_FOLDER = '/home/x/Dmitriev/dataset/'
DATASET = 'Google_Samara/'

DATA_FOLDER = DATA_FOLDER + DATASET

IMAGES_FOLDER = DATA_FOLDER + 'data/'
LABELS_FOLDER = DATA_FOLDER + 'labels/'

'''Это не для гугловских снимков'''
# list_with_train_files, list_with_test_files = files_splitter.get_lists_with_train_and_test_files(IMAGES_FOLDER,
#                                                                                                  LABELS_FOLDER,
#                                                                                                  'lists_with_train_and_test_files/' + DATASET)
# list_with_train_imgs_paths = [IMAGES_FOLDER + x for x in list_with_train_files]
# list_with_train_labels_paths = [LABELS_FOLDER + x for x in list_with_train_files]

'''Это для случаев папок train and test'''
list_with_train_imgs_paths = [DATA_FOLDER + 'train/' + 'data/' + x for x in
                              sorted(os.listdir(DATA_FOLDER + 'train/' + 'data/'))]
list_with_train_labels_paths = [DATA_FOLDER + 'train/' + 'labels/' + x for x in
                                sorted(os.listdir(DATA_FOLDER + 'train/' + 'labels/'))]
list_with_val_imgs_paths = [DATA_FOLDER + 'test/' + 'data/' + x for x in
                            sorted(os.listdir(DATA_FOLDER + 'test/' + 'data/'))]
list_with_labels_paths = [DATA_FOLDER + 'test/' + 'labels/' + x for x in
                          sorted(os.listdir(DATA_FOLDER + 'test/' + 'labels/'))]

train_data = DataLoader(list_with_train_imgs_paths, list_with_train_labels_paths, BATCH_SIZE, N_CLASSES,
                        size_of_crop=SIZE_OF_CROP)

# list_with_val_imgs_paths = [IMAGES_FOLDER + x for x in list_with_test_files]
# list_with_labels_paths = [LABELS_FOLDER + x for x in list_with_test_files]

print('Number of train files:', len(list_with_train_imgs_paths))
print('Test files:', len(list_with_val_imgs_paths))

val_data = DataLoader(list_with_val_imgs_paths, list_with_labels_paths, BATCH_SIZE,
                      N_CLASSES, shuffle=False, size_of_crop=SIZE_OF_CROP, augmentation=False)

# net = make_unet(SIZE_OF_CROP, N_CLASSES)
# net = make_fcn_resnet(SIZE_OF_CROP, N_CLASSES, freeze_base=False)

# net.summary()
# net = Unet(backbone_name='vgg16', encoder_weights='imagenet')
# net = Unet(backbone_name='resnet34', encoder_weights='imagenet',classes=2)
# net = Unet(backbone_name='resnet34')


# net = get_model(SIZE_OF_CROP, N_CLASSES)
net = LinkNet(N_CLASSES, input_shape=SIZE_OF_CROP).get_model(True)

sgd = SGD(momentum=0.9, decay=1e-6)

net.compile(optimizer=sgd, loss=keras.losses.binary_crossentropy, metrics=['acc'])

net.load_weights("checkpoints/LinkNet/Google_2/LinkNet49-0.955.h5")

# cyclic_callback = CyclicLR(step_size=254, max_lr=1e-3, base_lr=1e-6)
# tensoboard = keras.callbacks.TensorBoard(log_dir='./logs/linknet', batch_size=BATCH_SIZE, write_graph=False)
learning_rate_reducer = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2)

model_checkpoint = keras.callbacks.ModelCheckpoint('checkpoints/LinkNet/Google_2/LinkNet{epoch:02d}-{val_acc:.3f}.h5',
                                                   monitor='val_acc',
                                                   verbose=1, save_best_only=True)
# #
history = net.fit_generator(train_data, epochs=50, validation_data=val_data,
                            callbacks=[model_checkpoint])

# with open('LinkNet_history_pretrained_from_Tatarstan', 'wb') as file_pi:
#     pickle.dump(history.history, file_pi)
# print(history)

net.save_weights('checkpoints/LinkNet/Google_2/LinkNet_final.h5')


def sliding_window(top, step=10, window_size=(20, 20)):
    """ Slide a window_shape window across the image with a stride of step """
    for y in range(0, top.shape[0], step):
        if y + window_size[0] > top.shape[0]:
            y = top.shape[0] - window_size[0]
        for x in range(0, top.shape[1], step):
            if x + window_size[1] > top.shape[1]:
                x = top.shape[1] - window_size[1]
            yield y, x, window_size[1], window_size[0]


def test_images(test_images_paths, test_labels_paths, n_net, n_classes, window_size):
    accuracy = 0
    i = 0

    # clean_folder('results/')
    # clean_folder('test_images/')
    # clean_folder('test_images/data/')
    # clean_folder('test_images/labels/')

    bad_files_with_accuracy = {}

    for test_image_path, label_path in zip(test_images_paths, test_labels_paths):
        label_image = cv2.imread(label_path)
        label_image = label_image[:, :, ::-1]

        test_image = cv2.imread(test_image_path)
        test_image = test_image[..., ::-1]

        if label_image.shape == (240, 320):
            label_image = cv2.resize(label_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            test_image = cv2.resize(test_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            label_image = np.dstack([label_image] * 3)

        test_image = 1 / 255 * test_image

        # для сердечек
        label_image = convert_from_color(label_image, palette)

        prediction = np.zeros((*test_image.shape[0:2], n_classes + 1))

        test_image = test_image.reshape(1, *test_image.shape)

        for y, x, width, height in sliding_window(test_image[0], window_size[0] // 4, window_size):
            predic_after_net = n_net.predict(test_image[:, y:y + height, x: x + width])[0]
            prediction[y:y + height, x: x + width] += np.concatenate((1 - predic_after_net, predic_after_net), axis=2)

        # prediction = get_prediction(prediction, test_image[0])

        prediction = np.argmax(prediction, axis=-1)

        accuracy_for_one_image = np.mean(np.equal(prediction, label_image))
        accuracy += accuracy_for_one_image

        name = test_image_path.split('/')[-1]
        print('accuracy for ' + name + '=', accuracy_for_one_image)

        # if accuracy_for_one_image < 0.92:
        #     io.imsave('test_images/data/' + name, test_image[0])
        #     io.imsave('test_images/labels/' + name, convert_to_color(label_image, palette))
        #     io.imsave('results/' + name, convert_to_color(prediction, palette))
        #     bad_files_with_accuracy[name] = accuracy_for_one_image
        io.imsave('test_images/data/' + name, test_image[0])
        io.imsave('test_images/labels/' + name, convert_to_color(label_image, palette))
        io.imsave('results/' + name, convert_to_color(prediction, palette))

        i += 1

        if i == 20:
            break

    accuracy /= i
    print("Accuracy =", accuracy)
    bad_files_with_accuracy = OrderedDict(sorted(bad_files_with_accuracy.items(), key=itemgetter(1)))
    json.dump(bad_files_with_accuracy, open('bad_files', 'w'))


list_with_files = list(zip(list_with_val_imgs_paths, list_with_labels_paths))
# random.shuffle(list_with_files)
list_with_val_imgs_path, list_with_labels_paths = zip(*list_with_files)

list_with_val_imgs_paths = ['test_images/data_test_village.png', 'test_images/data_test.png', 'test_images/500.bmp']
list_with_labels_paths = ['test_images/label_test_village.png', 'test_images/label_test.png',
                          'test_images/label_test.png']
# test_images(list_with_train_imgs_paths, list_with_train_labels_paths, net, N_CLASSES, SIZE_OF_CROP[0:2])
test_images(list_with_val_imgs_paths, list_with_labels_paths, net, N_CLASSES, SIZE_OF_CROP[0:2])
