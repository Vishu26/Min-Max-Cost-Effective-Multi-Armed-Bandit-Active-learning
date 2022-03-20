import numpy as np
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
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from tqdm import tqdm

def compute_costs(xt, unlabeled_idx, fcn):
    costs = []
    for i in tqdm(range(0, len(unlabeled_idx), 64)):
        pred = np.argmax(fcn.predict(xt[unlabeled_idx[i:i+64]]), axis=3)
        for j in range(len(pred)):
            clicks = 0
            for k in np.unique(pred[j]):
                bi = pred[j]==k
                bi = bi.astype(np.float32)
                coords = corner_peaks(corner_harris(bi), min_distance=3, threshold_rel=0.01)
                clicks+=len(coords)
            costs.append(clicks)
    return np.array(costs)+4

def max_entropy(fcn, X, ap, return_index=True):
    scores = []
    for i in range(0, len(ap), 64):
        pb = fcn(X[ap[i:i+64]]).numpy()
        ent = np.mean((-pb*np.log(pb)).sum(axis=3), axis=(1, 2))
        scores.extend(ent.tolist())
    if return_index:
        return np.argpartition(scores, -20)[-20:]
    else:
        return scores


def bald(fcn, X, unlabeled_idx, n=20, return_index=True):
    scores = []
    for i in tqdm(range(0, len(unlabeled_idx), 64)):
        prob = np.stack([fcn(X[unlabeled_idx[i:i+64]], training=True) for _ in range(1)])
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


def max_cost_entropy(fcn, X, unlabeled_idx, cost):

    ent = max_entropy(fcn, X, unlabeled_idx, return_index=False)
    c = ent / (1 + ent*cost)
    return np.argpartition(c, -20)[-20:]

class Dataset:
    def __init__(self, x, y, idx=None, xp=None, yp=None):
        self.x = x
        self.y = y
        self.xp = xp
        self.yp = yp
        self.idx = idx

    def __call__(self):
        if self.idx:
            for i in self.idx:
                yield self.x[i], self.y[i]
        else:
            if type(self.xp)!=type(None):
                idx = np.arange(len(self.xp))
                np.random.shuffle(idx)
                for i in idx:
                    yield self.xp[i], self.yp[i]

            idx = np.arange(len(self.x))
            np.random.shuffle(idx)
            for i in idx:
                yield self.x[i], self.y[i]

Xtrain = np.load('Xtrain.npy', mmap_mode='r')
Ytrain = np.load('Ytrain.npy', mmap_mode='r')
Xvalid = np.load('Xvalid.npy', mmap_mode='r')
Yvalid = np.load('Yvalid.npy', mmap_mode='r')
Xpre = np.load('Xpre.npy', mmap_mode='r')
Ypre = np.load('Ypre.npy', mmap_mode='r')


valid = tf.data.Dataset.from_generator(Dataset(Xvalid, Yvalid), output_types=(tf.float32, tf.float32), output_shapes=(tf.TensorShape([128, 128, 3]), tf.TensorShape([128, 128, 20])))
#train = train.batch(64)
valid = valid.batch(64)

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

cp = tf.keras.callbacks.ModelCheckpoint(filepath="./ce",
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

def dice_coef(y_true, y_pred, smooth=100):

    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return (1-dice_coef(y_true, y_pred))*100


OPT = tf.keras.optimizers.Adam(learning_rate=0.00007)

fcn = build_FCN(OPT, PATCHSIZE, PATCHSIZE, NUMBER_BANDS)

fcn.compile(optimizer=OPT, loss=dice_coef_loss, metrics=['acc', mean_iou, f1_m])

fcn.load_weights("./pr")

final_cost = 0

steps = 10

from rl import EpsGreedy

mab = EpsGreedy(3, 0.05)

labeled_idx = []
iou = []
iou.append(fcn.evaluate(valid)[2])

for i in range(steps):
    unlabeled_idx = np.arange(Xtrain.shape[0])[np.logical_not(np.in1d(np.arange(Xtrain.shape[0]), labeled_idx))]
    ap = np.random.choice(unlabeled_idx, size=1000)
    arm = mab.play(i)
    #arm=2
    print(arm)
    cost = compute_costs(Xtrain, ap, fcn)
    if arm==0:
        idx = (np.argpartition(cost, 20)[:20]).tolist()
    elif arm==1:
        idx = max_entropy(fcn, Xtrain, ap).tolist()
    else:
        idx = max_cost_entropy(fcn, Xtrain, ap, cost).tolist()

    labeled_idx.extend(ap[idx])

    #fcn = build_FCN(OPT, PATCHSIZE, PATCHSIZE, NUMBER_BANDS)

    #fcn.compile(optimizer=OPT, loss= jaccard_distance(), metrics=['acc', mean_iou, f1_m])

    valididx = np.random.choice(range(len(Xvalid)), size=1000).tolist()
    validtrain = tf.data.Dataset.from_generator(Dataset(Xvalid, Yvalid, valididx), output_types=(tf.float32, tf.float32), output_shapes=(tf.TensorShape([128, 128, 3]), tf.TensorShape([128, 128, 20])))
    validtrain = validtrain.batch(64)
    train = tf.data.Dataset.from_generator(Dataset(Xtrain[labeled_idx], Ytrain[labeled_idx], None, Xpre, Ypre),  output_types=(tf.float32, tf.float32), output_shapes=(tf.TensorShape([128, 128, 3]), tf.TensorShape([128, 128, 20])))
    train = train.batch(64)
    hist = fcn.fit(train,
                    epochs=10,
                    verbose=1,
                    callbacks=[es,cp],
                    validation_data=validtrain)

    iou.append(fcn.evaluate(valid)[2])
    t_cost = np.sum(cost[idx])
    final_cost += t_cost
    print(F"Current Labeling Cost: {final_cost}")

    ioud = iou[-1] - iou[-2]
    reward = ioud / (1 + ioud*t_cost)

    mab.update(arm, reward)
    
