from keras.models import Model
from keras.layers import (
    Input, Conv2D, Lambda, Add, Activation)
import tensorflow as tf
from .crfrnn_layer import CrfRnnLayer

from keras.applications.resnet50 import ResNet50

FCN_RESNET = 'fcn_resnet'


def make_fcn_resnet(input_shape, nb_labels, freeze_base, use_crf = True):
    nb_rows, nb_cols, _ = input_shape
    input_tensor = Input(shape=input_shape)

    model = ResNet50(
        include_top=False, weights='imagenet', input_tensor=input_tensor)

    if freeze_base:
        for layer in model.layers:
            layer.trainable = False

    x32 = model.get_layer('activation_22').output
    x16 = model.get_layer('activation_40').output
    x8 = model.get_layer('activation_49').output

    c32 = Conv2D(nb_labels, (1, 1), name='conv_labels_32')(x32)
    c16 = Conv2D(nb_labels, (1, 1), name='conv_labels_16')(x16)
    c8 = Conv2D(nb_labels, (1, 1), name='conv_labels_8')(x8)

    def resize_bilinear(images):
        return tf.image.resize_bilinear(images, [nb_rows, nb_cols])

    r32 = Lambda(resize_bilinear, name='resize_labels_32')(c32)
    r16 = Lambda(resize_bilinear, name='resize_labels_16')(c16)
    r8 = Lambda(resize_bilinear, name='resize_labels_8')(c8)

    m = Add(name='merge_labels')([r32, r16, r8])
    outputs= Activation('sigmoid')(m)
    if use_crf:
        outputs = CrfRnnLayer(image_dims=(nb_rows, nb_cols),
                              num_classes=nb_labels,
                              theta_alpha=160.,
                              theta_beta=3.,
                              theta_gamma=3.,
                              num_iterations=10,
                              name='crfrnn')([m, input_tensor])

    model = Model(inputs=input_tensor, outputs=outputs)

    return model
