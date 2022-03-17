import numpy as np
from PIL import Image
import os
import glob
from tqdm import tqdm

mapping_20 = { 
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0,
        7: 1,
        8: 2,
        9: 0,
        10: 0,
        11: 3,
        12: 4,
        13: 5,
        14: 0,
        15: 0,
        16: 0,
        17: 6,
        18: 0,
        19: 7,
        20: 8,
        21: 9,
        22: 10,
        23: 11,
        24: 12,
        25: 13,
        26: 14,
        27: 15,
        28: 16,
        29: 0,
        30: 0,
        31: 17,
        32: 18,
        33: 19,
        -1: 0
    }

def read_images(paths, patchsize=320):

    imgs = []
    for path in tqdm(paths):
        imgarray = np.array(Image.open(path)) / 255
        nrows, ncols, nbands = imgarray.shape
        for i in range(int(nrows/patchsize)):
            for j in range(int(ncols/patchsize)):
                tocat = imgarray[i*patchsize:(i+1)*patchsize,
                                j*patchsize:(j+1)*patchsize, :]
                imgs.append(tocat)
    return np.array(imgs).reshape(-1, patchsize, patchsize, nbands)

def read_images_gt(paths, patchsize=320):

    global mapping_20

    imgs = []
    for path in tqdm(paths):
        imgarray = np.array(Image.open(path), dtype=np.uint8)
        label_mask = np.zeros_like(imgarray, dtype=np.uint8)
        for k in mapping_20:
            label_mask[imgarray == k] = mapping_20[k]
        nrows, ncols = imgarray.shape
        for i in range(int(nrows/patchsize)):
            for j in range(int(ncols/patchsize)):

                tocat = label_mask[i*patchsize:(i+1)*patchsize,
                                j*patchsize:(j+1)*patchsize]
                #tocat = to_categorical(tocat, 20)
                imgs.append(tocat)
    return np.array(imgs).reshape(-1, patchsize, patchsize)

 
if __name__ == '__main__':

    pathsx, pathsy = [], []

    pathx = 'leftImg8bit_trainvaltest/leftImg8bit/train/'
    pathy = 'gtFine_trainvaltest/gtFine/train/'
    dirs = os.listdir(pathx)
    for i in dirs:
        pathsx.extend(glob.glob(pathx+i+'/*.png'))
    dirs = os.listdir(pathy)
    for i in dirs:
        pathsy.extend(glob.glob(pathy+i+'/*instanceIds.png'))    

    idx = np.random.choice(range(len(pathsx)), size=500).tolist()
    train_paths = [pathsx[i] for i in idx[:250]]
    valid_paths = [pathsx[i] for i in idx[250:]]
    train_pathsy = [pathsy[i] for i in idx[:250]]
    valid_pathsy = [pathsy[i] for i in idx[250:]]

    train_imgs = read_images(train_paths)
    np.savez_compressed("cityXi", x_train=train_imgs)
    del train_imgs, train_paths
    valid_imgs = read_images(valid_paths)
    np.savez_compressed("cityXv", x_test=valid_imgs)
    del valid_imgs, valid_paths
    train_imgsy = read_images_gt(train_pathsy)
    np.savez_compressed("cityYi", y_train=train_imgsy)
    del train_imgsy, train_pathsy
    valid_imgsy = read_images_gt(valid_pathsy)
    np.savez_compressed("cityYv", y_test=valid_imgsy)
    
