from __future__ import print_function
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
K.set_image_data_format('channels_last')

from models.VNet import get_model
from losses import dice_coef_loss
from metrics import dice_coef, recall, f1_score

def main():
    crop_shape = (128, 128, 64)
    batch_size = 4
    epochs = 40
    learning_rate = 1e-5

    data = np.load('numpy_data/images.npy')
    data = data.astype('float32')
    mean = np.mean(data)  # mean for data centering
    std = np.std(data)  # std for data normalization
    data -= mean
    data /= std

    masks = np.load('numpy_data/masks.npy')
    masks = masks.astype('float32')

    model = get_model(crop_shape)
    model.compile(optimizer=Adam(lr=learning_rate), loss=dice_coef_loss,
                  metrics=[dice_coef, recall, f1_score])

    # model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)
    # model.fit(imgs_train, masks_train, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True,
    #           callbacks=[model_checkpoint], validation_data=(imgs_test, masks_test))

    model.fit(data, masks, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True,
              validation_split=0.2)

    return 0

if __name__ == "__main__":
    main()