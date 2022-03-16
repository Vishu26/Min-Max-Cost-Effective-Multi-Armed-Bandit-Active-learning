import numpy as np
Xtrain = np.load('Xtrain.npz')['arr_0']
Ytrain = np.load('Ytrain.npz')['arr_0']
Xvalid = np.load('Xvalid.npz')['arr_0']
Yvalid = np.load('Yvalid.npz')['arr_0']
Ytone = np.load('ytone.npz')['arr_0'].astype(np.float32)
Yvone = np.load('yvone.npz')['arr_0'].astype(np.float32)
cost = np.load("costs.npy")[:, 0]

import tensorflow as tf
from keras_drop_block import DropBlock2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ZeroPadding2D, Dropout,Convolution2D, MaxPooling2D, Activation, BatchNormalization, Reshape
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD, Adadelta
import tensorflow_addons as tfa

import tensorflow.python.keras.backend as K
sess = K.get_session()
from tensorflow.compat.v1.keras.backend import set_session
import imp, h5py
import pickle
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1
set_session(tf.compat.v1.Session(config=config))
from tensorflow import keras
def build_FCN(optimizer, nrows, ncols, nbands):
    """Function to create Keras model of sample network."""
    model = tf.keras.models.Sequential()
    model.add(ZeroPadding2D((3, 3), input_shape=(nrows, ncols, nbands)))
    model.add(DropBlock2D(block_size=3, keep_prob=0.8)
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
    model.add(DropBlock2D(block_size=3, keep_prob=0.8)
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
    model.add(DropBlock2D(block_size=3, keep_prob=0.8)
    model.add(Convolution2D(
              filters=64,
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
    model.add(DropBlock2D(block_size=3, keep_prob=0.8)
    model.add(Convolution2D(
              filters=64,
              kernel_size=(5, 5),
              dilation_rate=(1, 1),kernel_regularizer=tf.keras.regularizers.l2(0.001)
    ))
    model.add(BatchNormalization(axis=3))
    model.add(Activation("relu"))
    model.add(Dropout(0.25))
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

cp = tf.keras.callbacks.ModelCheckpoint(filepath="./objectrand",
                                        monitor='val_loss',
                                        verbose=0,
                                        save_weights_only=True,
                                        mode='auto',
                                        save_best_only=True)
PATCHSIZE=128
NUMBER_BANDS = 8
NUMBER_CLASSES = 5
NUMBER_EPOCHS = 200 


from tqdm import tqdm


def my_loss(weight):
    def weighted_cross_entropy_with_logits(labels, logits):
        loss = tf.nn.weighted_cross_entropy_with_logits(
            labels, logits, weight
        )
        return loss
    return weighted_cross_entropy_with_logits

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


OPT = tf.keras.optimizers.Adam(learning_rate=0.00007)

fcn = build_FCN(OPT, PATCHSIZE, PATCHSIZE, NUMBER_BANDS)

fcn.compile(optimizer=OPT, loss= jaccard_distance(), metrics=['acc', mean_iou, f1_m])
"""
hist = fcn.fit(x=Xtrain,
                    y=Ytrain,
                    batch_size=64,
                    epochs=NUMBER_EPOCHS,
                    verbose=1,
                    callbacks=[es,cp],
                    validation_data=(Xvalid, Yvalid))
"""


# In[130]:
print(Yvone.shape, Xvalid.shape)
c = 0
labeled_idx = []
idx = np.random.choice(range(len(Xtrain)), size=5)
hist = fcn.fit(x=Xtrain[idx],
                    y=Ytone[idx],
                    batch_size=5,
                    epochs=NUMBER_EPOCHS,
                    verbose=0,
                    callbacks=[es,cp],
                    validation_data=(Xvalid, Yvone))
labeled_idx.extend(idx.tolist())
c += np.sum(cost[idx])
perf = []
print(c)
perf.append(fcn.evaluate(Xvalid, Yvone))
#print(fcn.evaluate(Xtrain, Ytone, batch_size=64))
#print(fcn.evaluate(Xtrain[idx], Ytone[idx]))

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

iterations = 5
labeled_idx = []
for epoch in tqdm(range(iterations)):
    unlabeled_idx = np.arange(Xtrain.shape[0])[np.logical_not(np.in1d(np.arange(Xtrain.shape[0]), labeled_idx))]
    idx = np.random.choice(len(unlabeled_idx), size=5)
    labeled_idx.extend(unlabeled_idx[idx].tolist())
    c += np.sum(cost[unlabeled_idx[idx]])
    #fcn = build_FCN(OPT, PATCHSIZE, PATCHSIZE, NUMBER_BANDS)
    #fcn.compile(optimizer=OPT, loss= jaccard_distance(), metrics=['acc', mean_iou, f1_m])
    hist = fcn.fit(x=Xtrain[labeled_idx],
                    y=Ytone[labeled_idx],
                    batch_size=5,
                    epochs=NUMBER_EPOCHS,
                    verbose=0,
                    callbacks=[es,cp],
                    validation_data=(Xvalid, Yvone))
    #fcn.load_weights("./al")
    perf.append(fcn.evaluate(Xvalid, Yvone))
    print(c)
    #print(fcn.evaluate(Xtrain, Ytone, batch_size=64))
    #print(fcn.evaluate(Xtrain[labeled_idx], Ytone[labeled_idx]))

#print(labeled_idx)

#np.save("randsiteA2", perf)
