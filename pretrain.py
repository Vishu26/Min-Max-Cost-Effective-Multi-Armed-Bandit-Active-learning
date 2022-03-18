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

class Dataset:
    def __init__(self, label='train'):
        self.label = label
        if self.label=='train':
            self.xt = np.load('Xpre.npy', mmap_mode='r')#dtype='float32', mode='r', shape=(6400, 128, 128, 3))
            self.yt = np.load('Ypre.npy', mmap_mode='r')#dtype='float32', mode='r', shape=(6400, 128, 128, 20))
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

train = train.batch(64)
valid = valid.batch(64)

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
                                        patience=30,
                                        verbose = 0,
                                        restore_best_weights=True)

cp = tf.keras.callbacks.ModelCheckpoint(filepath="./pr",
                                        monitor='val_loss',
                                        verbose=0,
                                        save_weights_only=True,
                                        mode='auto',
                                        save_best_only=True)
PATCHSIZE=128
NUMBER_BANDS = 3
NUMBER_CLASSES = 20
NUMBER_EPOCHS = 200


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


OPT = tf.keras.optimizers.Adam(learning_rate=0.00007)

fcn = build_FCN(OPT, PATCHSIZE, PATCHSIZE, NUMBER_BANDS)

fcn.compile(optimizer=OPT, loss=dice_coef_loss, metrics=['acc', mean_iou, f1_m])

#Ytrain = to_categorical(Ytrain, 20)
#Yvalid = to_categorical(Yvalid, 20)
fcn.load_weights("./pr")

hist = fcn.fit(train,
                    epochs=NUMBER_EPOCHS,
                    verbose=1,
                    callbacks=[es,cp],
                    validation_data=valid)
