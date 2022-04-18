import numpy as np
#Xtrain = np.load('Xpre.npy')[:100]#, dtype='float32', mode='r', shape=(6400, 128, 128, 3))[:100]#np.load('cityXi.npz')['x_train']
#Ytrain = np.load('Ypre.npy')[:100]#, dtype='uint8', mode='r', shape=(6400, 128, 128, 20))[:100]#np.load('cityYi.npz')['y_train']
#Xvalid = np.load('Xvalid.npy')[:100]#, dtype='float32', mode='r', shape=(19200, 128, 128, 3))[:100] #np.load('cityXiv.npz')['x_train']
#Yvalid = np.load('Yvalid.npy')[:100]#, dtype='uint8', mode='r', shape=(19200, 128, 128, 20))[:100]#np.load('cityYiv.npz')['y_train']

import tensorflow as tf
from keras_drop_block import DropBlock2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ZeroPadding2D, Dropout,Convolution2D, MaxPooling2D, Activation, BatchNormalization, Reshape
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD, Adadelta
import tensorflow_addons as tfa
from tensorflow.keras.utils import to_categorical
import tensorflow.python.keras.backend as K
sess = K.get_session()
from tensorflow.compat.v1.keras.backend import set_session
import imp, h5py
import pickle
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1
set_session(tf.compat.v1.Session(config=config))
import os
import random
import numpy as np
#import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, BatchNormalization, Activation, Dropout, Reshape, UpSampling2D
from tensorflow.keras.optimizers import Adadelta, Nadam
from tensorflow.keras.models import Model, load_model
#from tensorflow.keras.utils import multi_gpu_model, plot_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy

# [ 148343025 1162958786  215752525]
# [0.687561014639342, 5.390244151256168, 1.0]
def conv_block(tensor, nfilters, size=3, padding='same', initializer="he_normal"):
    x = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def deconv_block(tensor, residual, nfilters, size=2, padding='same', strides=(2, 2)):
    y = UpSampling2D(size=(size, size))(tensor)
    y = Conv2D(nfilters, kernel_size=(size, size), padding=padding)(y)
    y = concatenate([y, residual], axis=3)
    y = conv_block(y, nfilters)
    return y


def Unet(img_height, img_width, nclasses=3, filters=64):
# down
    input_layer = Input(shape=(img_height, img_width, 3), name='image_input')
    conv1 = conv_block(input_layer, nfilters=filters)
    conv1_out = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = conv_block(conv1_out, nfilters=filters*2)
    conv2_out = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = conv_block(conv2_out, nfilters=filters*4)
    conv3_out = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = conv_block(conv3_out, nfilters=filters*8)
    conv4_out = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv4_out = Dropout(0.5)(conv4_out)
    conv5 = conv_block(conv4_out, nfilters=filters*16)
    conv5 = Dropout(0.5)(conv5)
# up
    deconv6 = deconv_block(conv5, residual=conv4, nfilters=filters*8)
    deconv6 = Dropout(0.5)(deconv6)
    deconv7 = deconv_block(deconv6, residual=conv3, nfilters=filters*4)
    deconv7 = Dropout(0.5)(deconv7) 
    deconv8 = deconv_block(deconv7, residual=conv2, nfilters=filters*2)
    deconv9 = deconv_block(deconv8, residual=conv1, nfilters=filters)
# output
    output_layer = Conv2D(filters=nclasses, kernel_size=(1, 1))(deconv9)
    output_layer = BatchNormalization()(output_layer)
    #output_layer = Reshape((img_height*img_width, nclasses), input_shape=(img_height, img_width, nclasses))(output_layer)
    output_layer = Activation('softmax')(output_layer)
    
    model = Model(inputs=input_layer, outputs=output_layer, name='Unet')
    print('Created and loaded model')
    return model

class Dataset:
    def __init__(self, label='train'):
        self.label = label
        if self.label=='train':
            self.xt = np.load('Xtrain.npy', mmap_mode='r')#dtype='float32', mode='r', shape=(6400, 128, 128, 3))
            self.yt = np.load('Ytrain.npy', mmap_mode='r')#dtype='float32', mode='r', shape=(6400, 128, 128, 20))
        else:
            self.xv = np.load('Xvalid.npy', mmap_mode='r')#dtype='float32', mode='r', shape=(19200, 128, 128, 3))
            self.yv = np.load('Yvalid.npy', mmap_mode='r')#dtype='float32', mode='r', shape=(19200, 128, 128, 20))

    def __call__(self):
        if self.label == 'train':
            idx = np.arange(len(self.xt))
            np.random.shuffle(idx)
            for i in idx:
                yield self.xt[i], self.yt[i]
        else:
            for i in np.random.choice(range(len(self.xv)), size=1000):
                yield self.xv[i], self.yv[i]

train = tf.data.Dataset.from_generator(Dataset('train'), output_types=(tf.float32, tf.float32), output_shapes=(tf.TensorShape([128, 128, 3]), tf.TensorShape([128, 128, 20])))
valid = tf.data.Dataset.from_generator(Dataset('valid'), output_types=(tf.float32, tf.float32), output_shapes=(tf.TensorShape([128, 128, 3]), tf.TensorShape([128, 128, 20])))

train = train.batch(32)
valid = valid.batch(32)

#train_dataset = tf.data.Dataset.from_tensor_slices((Xtrain, Ytrain))
#del Xtrain, Ytrain
#test_dataset = tf.data.Dataset.from_tensor_slices((Xvalid, Yvalid))
#del Xvalid, Yvalidi

#train_dataset = train_dataset.batch(64)
#test_dataset = test_dataset.batch(64)

from tensorflow import keras
def build_FCN(optimizer, nrows, ncols, nbands):
    """Function to create Keras model of sample network."""
    model = tf.keras.models.Sequential()
    model.add(ZeroPadding2D((3, 3), input_shape=(nrows, ncols, nbands)))
    model.add(DropBlock2D(block_size=3, keep_prob=0.8))
    model.add(Convolution2D(
              filters=16,
              kernel_size=(3, 3),
              dilation_rate=(1, 1),kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    model.add(BatchNormalization(axis=3))
    model.add(Activation("relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(MaxPooling2D(
              pool_size=(7, 7),
              strides=(1, 1)
    ))
    model.add(ZeroPadding2D((2, 2)))
    model.add(DropBlock2D(block_size=3, keep_prob=0.8))
    model.add(Convolution2D(
              filters=32,
              kernel_size=(5, 5),
              dilation_rate=(1, 1),kernel_regularizer=tf.keras.regularizers.l2(0.001)
    ))
    model.add(BatchNormalization(axis=3))
    model.add(Activation("relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(MaxPooling2D(
            pool_size=(3, 3),
            strides=(1, 1)
    ))
    model.add(ZeroPadding2D((2, 2)))
    model.add(DropBlock2D(block_size=3, keep_prob=0.8))
    model.add(Convolution2D(
              filters=64,
              kernel_size=(7, 7),
           dilation_rate=(1, 1),kernel_regularizer=tf.keras.regularizers.l2(0.001)
    ))
    model.add(BatchNormalization(axis=3))
    model.add(Activation("relu"))
    model.add(ZeroPadding2D((2, 2)))
    model.add(MaxPooling2D(
            pool_size=(3, 3),
            strides=(1, 1)
    ))
    model.add(ZeroPadding2D((2, 2)))
    model.add(DropBlock2D(block_size=3, keep_prob=0.8))
    model.add(Convolution2D(
              filters=128,
              kernel_size=(5, 5),
              dilation_rate=(1, 1),kernel_regularizer=tf.keras.regularizers.l2(0.001)
    ))
    model.add(BatchNormalization(axis=3))
    model.add(Activation("relu"))
    #model.add(Dropout(0.25))
    model.add(ZeroPadding2D((1, 1)))
    model.add(MaxPooling2D(
            pool_size=(3, 3),
            strides=(1, 1)
    ))
    model.add(keras.layers.Conv2D(
              filters=NUMBER_CLASSES,
              kernel_size=(1, 1),kernel_regularizer=tf.keras.regularizers.l2(0.001)
    ))
    model.add(keras.layers.Activation(
              activation="softmax"
    ))
    #model.compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer=optimizer)
    return model

es = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                        patience=2,
                                        verbose = 0,
                                        restore_best_weights=True)

cp = tf.keras.callbacks.ModelCheckpoint(filepath="./unet_final69",
                                        monitor='val_loss',
                                        verbose=0,
                                        save_weights_only=True,
                                        mode='auto',
                                        save_best_only=True)

PATCHSIZE=128
NUMBER_BANDS = 3
NUMBER_CLASSES = 20
NUMBER_EPOCHS = 20


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def focal_loss(gamma=2., alpha=0.25):

    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y_true, y_pred):
        epsilon = 1.e-9
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)

        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.math.log(model_out))
        weight = tf.multiply(y_true, tf.math.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        return tf.reduce_mean(reduced_fl)
    return focal_loss_fixed

def mean_iou(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        intersection = tf.reduce_sum(y_true * y_pred, axis=(1,2))
        sum_ = tf.reduce_sum(y_true + y_pred, axis=(1,2))
        jac = (intersection + 1e-7) / (sum_ - intersection + 1e-7)
        return jac

def jaccard_distance(smooth=100):
    def jd(y_true, y_pred):
        """ Calculates mean of Jaccard distance as a loss function """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        intersection = tf.reduce_sum(y_true * y_pred, axis=(1,2))
        sum_ = tf.reduce_sum(y_true + y_pred, axis=(1,2))
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
        jd =  (1 - jac) * smooth
        return tf.reduce_mean(jd)
    return jd

def dice_coef(y_true, y_pred, smooth=100):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def categorical_focal_loss(alpha, gamma=2.):

    alpha = np.array(alpha, dtype=np.float32)

    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Compute mean loss in mini_batch
        return K.mean(K.sum(loss, axis=-1))

    return categorical_focal_loss_fixed
class MyMeanIOU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)
OPT = tf.keras.optimizers.Adam(learning_rate=0.0001)
m = tf.keras.metrics.MeanIoU(num_classes=20)
#fcn = build_FCN(OPT, PATCHSIZE, PATCHSIZE, NUMBER_BANDS)
fcn = Unet(128, 128, 20, 128)

fcn.compile(optimizer=OPT, loss=dice_coef_loss, metrics=['acc', MyMeanIOU(num_classes=20), f1_m])

#Ytrain = to_categorical(Ytrain, 20)
#Yvalid = to_categorical(Yvalid, 20)
fcn.load_weights("./unet_final")

hist = fcn.fit(train,
                    epochs=NUMBER_EPOCHS,
                    verbose=1,
                    callbacks=[es,cp],
                    validation_data=valid)

#fcn.evaluate(valid)





