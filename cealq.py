import numpy as np
Xtrain = np.load('Xtrain.npz')['arr_0']
Ytrain = np.load('Ytrain.npz')['arr_0']
Xvalid = np.load('Xvalid.npz')['arr_0']
Yvalid = np.load('Yvalid.npz')['arr_0']
Ytone = np.load('ytone.npz')['arr_0'].astype(np.float32)
Yvone = np.load('yvone.npz')['arr_0'].astype(np.float32)
cost = np.load("costs.npy")[:, 0]

import tensorflow as tf

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
    model.add(ZeroPadding2D((2, 2)))
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

cp = tf.keras.callbacks.ModelCheckpoint(filepath="./al",
                                        monitor='val_loss',
                                        verbose=0,
                                        save_weights_only=True,
                                        mode='auto',
                                        save_best_only=True)
PATCHSIZE=128
NUMBER_BANDS = 8
NUMBER_CLASSES = 5
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

def max_entropy(fcn, X, return_index=True):
    scores = []
    for i in range(0, len(X), 64):
        pb = fcn(X[i:i+64]).numpy()
        ent = np.mean((-pb*np.log(pb)).sum(axis=3), axis=(1, 2))
        scores.extend(ent.tolist())
    if return_index:
        return np.argpartition(scores, -5)[-5:]
    else:
        return scores


def bald(fcn, X, n=5, return_index=True):
    scores = []
    for i in range(0, len(X), 64):
        prob = np.stack([fcn(X[i:i+64], training=True) for _ in range(10)])
        pb = np.mean(prob, axis=0)
        entropy1 = (-pb*np.log(pb)).sum(axis=3)
        entropy2 = (-prob*np.log(prob)).sum(axis=4).mean(axis=0)
        un = entropy2 - entropy1
        un = un.reshape(un.shape[0], un.shape[1]*un.shape[2])
        pbald = np.mean(un, axis=1).flatten()
        scores.extend(pbald.tolist())
    if return_index:
        return np.argpartition(scores, n)[:n]
    else:
        return scores


def max_cost_entropy(fcn, X, cost):

    ent = bald(fcn, X, n=5, return_index=False)
    c = ent / (1 + ent*cost)
    return np.argpartition(c, -5)[-5:]

OPT = tf.keras.optimizers.Adam(learning_rate=0.00007)

fcn = build_FCN(OPT, PATCHSIZE, PATCHSIZE, NUMBER_BANDS)

fcn.compile(optimizer=OPT, loss= jaccard_distance(), metrics=['acc', mean_iou, f1_m])

from rl import EpsGreedy

mab = EpsGreedy(3, 0.05)

labeled_idx = []
iou = []

idx = np.random.choice(range(len(Xtrain)), size=5)
hist = fcn.fit(x=Xtrain[idx],
                    y=Ytone[idx],
                    batch_size=5,
                    epochs=NUMBER_EPOCHS,
                    verbose=0,
                    callbacks=[es,cp],
                    validation_data=(Xvalid, Yvone))
labeled_idx.extend(idx.tolist())
iou.append(fcn.evaluate(Xvalid, Yvone)[2])

final_cost = np.sum(cost[labeled_idx])
print(F"Current Labeling Cost: {final_cost}")
steps = 10

for i in range(steps):
    unlabeled_idx = np.arange(Xtrain.shape[0])[np.logical_not(np.in1d(np.arange(Xtrain.shape[0]), labeled_idx))]
    arm = mab.play(i)
    #arm=2
    print(arm)
    if arm==0:
        idx = (np.argpartition(cost[unlabeled_idx], 5)[:5]).tolist()
    elif arm==1:
        idx = bald(fcn, Xtrain[unlabeled_idx]).tolist()
    else:
        idx = max_cost_entropy(fcn, Xtrain[unlabeled_idx], cost[unlabeled_idx]).tolist()

    labeled_idx.extend(unlabeled_idx[idx])

    fcn = build_FCN(OPT, PATCHSIZE, PATCHSIZE, NUMBER_BANDS)

    fcn.compile(optimizer=OPT, loss= jaccard_distance(), metrics=['acc', mean_iou, f1_m])

    hist = fcn.fit(x=Xtrain[labeled_idx],
                    y=Ytone[labeled_idx],
                    batch_size=5,
                    epochs=NUMBER_EPOCHS,
                    verbose=0,
                    callbacks=[es,cp],
                    validation_data=(Xvalid, Yvone))

    iou.append(fcn.evaluate(Xvalid, Yvone)[2])
    t_cost = np.sum(cost[unlabeled_idx[idx]])
    final_cost += t_cost
    print(F"Current Labeling Cost: {final_cost}")

    ioud = iou[-1] - iou[-2]
    reward = ioud / (1 + ioud*t_cost)

    mab.update(arm, reward)


