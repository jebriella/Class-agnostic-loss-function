# U-Net

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Activation, Flatten, Input, Reshape, Activation
from tensorflow.keras.layers import Dropout, BatchNormalization, Conv2DTranspose, concatenate, SpatialDropout2D, Permute
import tensorflow as tf
import numpy as np

def model_unet(img_size, base, nr_classes, dept, batch_norm = False, s_dropout = 0):

    inputs = Input((img_size[0], img_size[1], 1))

    length = dept*2 + 1
    base_list = np.zeros(length)
    multiply = 1

    for i in range(dept):
        base_list[i] = base*multiply
        base_list[length - i - 1] = base*multiply
        multiply =  multiply*2

    base_list[dept] = int(base*multiply)

    conv = []
    pool_t = inputs

    for i in range(dept):
        conv_t, pool_t = contraction_block(pool_t, base_list[i], batch_norm, s_dropout)
        conv.append(conv_t)

    conv_t = bottleneck(pool_t, base_list[dept], batch_norm)

    for i in range(dept):
        conv_t = expansion_block(conv_t, conv[dept-i-1], base_list[dept+i+1], batch_norm, s_dropout)


    conv_last = Conv2D(nr_classes, (1,1))(conv_t)
    act = Activation('softmax')(conv_last)

    #W, H, _ = conv_last.shape.as_list()[1:]
    #reshape = Reshape((W*H, nr_classes))(conv_last)
    #act = Activation('softmax')(reshape)
    #reshape = Reshape((W, H, nr_classes))(act)


    model = Model(inputs=[inputs],outputs=[act])

    return model

def contraction_block(layer, current_base, batch_n = False, s_dropout = 0):

    conv = Conv2D(int(current_base), kernel_size=(3,3), padding='same')(layer)
    if batch_n == True:
        conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(int(current_base), kernel_size=(3,3), padding='same')(conv)
    if batch_n == True:
        conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    pool = MaxPooling2D(pool_size=(2,2))(conv)
    if s_dropout != 0:
        #pool = Dropout(s_dropout)(pool)
        pool = SpatialDropout2D(s_dropout)(pool)

    return conv, pool

def bottleneck(layer, current_base, batch_n = False):
    conv = Conv2D(int(current_base), kernel_size=(3,3), padding='same')(layer)
    if batch_n == True:
        conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(int(current_base), kernel_size=(3,3), padding='same')(conv)
    if batch_n == True:
        conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)

    return conv

def expansion_block(layer, concat_layer, current_base, batch_n = False, s_dropout = 0):

    ups = Conv2DTranspose(int(current_base),(2,2),strides=(2,2),padding='same')(layer)
    up = concatenate([ups, concat_layer])
    if s_dropout != 0:
        #up = Dropout(s_dropout)(up)
        pool = SpatialDropout2D(s_dropout)(up)
    conv = Conv2D(int(current_base), kernel_size=(3,3), padding='same')(up)
    if batch_n == True:
        conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(int(current_base), kernel_size=(3,3), padding='same')(up)
    if batch_n == True:
        conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)

    return conv
