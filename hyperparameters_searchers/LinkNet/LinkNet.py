from hyperas.distributions import choice, uniform
from hyperas import optim
from hyperopt import Trials, STATUS_OK, tpe

import loss_functions
import utils.files_splitter as files_splitter

from data_loader import DataLoader
from keras.optimizers import SGD

from hyperparameters_searchers.LinkNet.create_model import LinkNet



def data():
    BATCH_SIZE = 15

    SIZE_OF_CROP = [512, 512, 3]  # size of crop of image [height, width, channels]
    CLASSES = ['Buildings', 'Background']
    N_CLASSES = len(CLASSES)

    DATA_FOLDER = '/home/x/Dmitriev-semseg/dataset/'
    DATASET = 'Moscow/'

    DATA_FOLDER = DATA_FOLDER + DATASET

    IMAGES_FOLDER = DATA_FOLDER + 'data/'
    LABELS_FOLDER = DATA_FOLDER + 'labels/'

    list_with_train_files, list_with_test_files = files_splitter.get_lists_with_train_and_test_files(IMAGES_FOLDER,
                                                                                                     LABELS_FOLDER,
                                                                                                     '/home/x/Dmitriev-semseg/Keras_seg/lists_with_train_and_test_files/' + DATASET)
    list_with_train_imgs_paths = [IMAGES_FOLDER + x for x in list_with_train_files]
    list_with_train_labels_paths = [LABELS_FOLDER + x for x in list_with_train_files]

    train_data = DataLoader(list_with_train_imgs_paths, list_with_train_labels_paths, BATCH_SIZE, N_CLASSES,
                            size_of_crop=SIZE_OF_CROP)

    list_with_imgs_paths = [IMAGES_FOLDER + x for x in list_with_test_files]
    list_with_labels_paths = [LABELS_FOLDER + x for x in list_with_test_files]

    val_data = DataLoader(list_with_imgs_paths, list_with_labels_paths, BATCH_SIZE,
                          N_CLASSES, shuffle=False, size_of_crop=SIZE_OF_CROP, augmentation=False)

    return train_data, val_data

def model(train_data, val_data):
    dropout_rate = {{uniform(0,1)}}

    model = LinkNet(2, input_shape = [512,512,3], dropout_rate=dropout_rate).get_model()


    model.compile(optimizer='Nadam',
                  loss=loss_functions.cross_entropy_2d,
                  metrics=['accuracy'])

    model.fit_generator(train_data, epochs=100, validation_data=val_data)

    score, acc = model.evaluate_generator(generator=val_data)

    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


train_data, val_data = data()

best_run, best_model = optim.minimize(model=model,
                                      data=data, algo=tpe.suggest,
                                      max_evals=5,
                                      trials=Trials())
print(best_run)
