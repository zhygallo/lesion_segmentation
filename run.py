from __future__ import print_function
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
K.set_image_data_format('channels_last')

from models.VNet import get_model
from losses import dice_coef_loss
from metrics import dice_coef

def main():
    crop_shape = (128, 128, 64)
    batch_size = 32
    epochs = 2

    imgs_train = np.load('numpy_data/train_imgs.npy')
    masks_train = np.load('numpy_data/train_masks.npy')

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    masks_train = masks_train.astype('float32')

    imgs_test = np.load('numpy_data/test_imgs.npy')
    masks_test = np.load('numpy_data/test_masks.npy')

    imgs_test = imgs_test.astype('float32')
    mean = np.mean(imgs_test)
    std = np.std(imgs_test)

    imgs_test -= mean
    imgs_test /= std

    masks_test = masks_test.astype('float32')

    model = get_model(crop_shape)
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)
    model.fit(imgs_train, masks_train, batch_size=batch_size, nb_epoch=epochs, verbose=1, shuffle=True,
              callbacks=[model_checkpoint], validation_data=(imgs_test, masks_test))

    print("Testing")
    score = model.evaluate(imgs_test, masks_test)
    print('Test score: ' + str(score))
    print('Test score:', score)


    return 0

if __name__ == "__main__":
    main()