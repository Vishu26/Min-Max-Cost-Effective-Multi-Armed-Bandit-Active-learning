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
import numpy as np
from tensorflow import keras
def build_FCN(optimizer, nrows, ncols, nbands):
    """Function to create Keras model of sample network."""
    model = tf.keras.models.Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(128, 128, 24)))
    #model.add(DropBlock2D(block_size=3, keep_prob=0.8))
    model.add(Convolution2D(
              filters=64,
              #input_shape=(128, 128, 24),
              kernel_size=(3, 3),
              dilation_rate=(1, 1),kernel_regularizer=tf.keras.regularizers.l2(0.0001)
    ))
    model.add(keras.layers.Activation(
              activation="relu"
    ))
    model.add(Convolution2D(
        filters=20,
        kernel_size=(1, 1),
        activation='softmax'))
    #model.compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer=optimizer)
    return model

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
class MyMeanIOU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)

Xtrain = np.load("useg.npy")
Ytrain = np.load('Ytrain.npy')[:1000]

OPT = tf.keras.optimizers.Adam(learning_rate=0.0007)

fcn = build_FCN(OPT, PATCHSIZE, PATCHSIZE, NUMBER_BANDS)

fcn.compile(optimizer=OPT, loss=dice_coef_loss, metrics=['acc', MyMeanIOU(20), f1_m])

#Ytrain = to_categorical(Ytrain, 20)
#Yvalid = to_categorical(Yvalid, 20)
#fcn.load_weights("./final")

#fcn2 = build_FCN(OPT, PATCHSIZE, PATCHSIZE, NUMBER_BANDS)
#fcn2.compile(optimizer=OPT, loss=dice_coef_loss, metrics=['acc', mean_iou, f1_m])
#fcn2.load_weights("./final")

hist = fcn.fit(Xtrain,Ytrain,
                    epochs=200,
                    batch_size=16,
                    verbose=1)#,
                    #callbacks=[es,cp],
                    #validation_data=validtrain)
fcn.save("mnet")
