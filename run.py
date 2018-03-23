from __future__ import print_function
import os
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
K.set_image_data_format('channels_last')
from keras.utils import multi_gpu_utils

from models.VNet import get_model
from losses import dice_coef_loss
from metrics import dice_coef

def main():
    crop_shape = (128, 128, 64)
    batch_size = 2
    # num_classes = 2
    epochs = 20

    imgs_train = np.load('/home/zhygallo/Documents/GuidedResearch/lesion_segmentation/numpy_data/train_imgs.npy')
    masks_train = np.load('/home/zhygallo/Documents/GuidedResearch/lesion_segmentation/numpy_data/train_masks.npy')

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    masks_train = masks_train.astype('float32')
    # masks_train /= 255.  # scale masks to [0, 1]


    model = get_model(crop_shape)
    model = multi_gpu_utils(model, gpus=2)
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)
    model.fit(imgs_train, masks_train, batch_size=batch_size, nb_epoch=epochs, verbose=1, shuffle=True,
              validation_split=0.2,
              callbacks=[model_checkpoint])

    return 0

if __name__ == "__main__":
    main()