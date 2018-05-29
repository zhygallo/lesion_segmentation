from __future__ import print_function
import click
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras import backend as K
K.set_image_data_format('channels_last')
import os

from models.HighRes3DNet import get_model
from losses import dice_coef_loss
from metrics import dice_coef, recall, f1_score

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

@click.command()
@click.argument('train_images')
@click.argument('train_masks')
@click.argument('test_images')
@click.argument('test_masks')
@click.argument('outdir')
def main(train_images, train_masks, test_images, test_masks, outdir):
    # crop_shape = (128, 128, 64)
    crop_shape = (96, 96, 96)
    batch_size = 4
    epochs = 400
    learning_rate = 1e-5

    train_data = np.load(train_images)
    train_data = train_data.astype('float32')
    train_masks = np.load(train_masks)
    train_masks = train_masks.astype('float32')

    test_data = np.load(test_images)
    test_data = test_data.astype('float32')
    test_masks = np.load(test_masks)
    test_masks = test_masks.astype('float32')

    model = get_model(crop_shape)

    # model.load_weights(outdir + '/weights.h5')

    model.compile(optimizer=Adam(lr=learning_rate), loss=dice_coef_loss,
                  metrics=[dice_coef, recall, f1_score])

    model_checkpoint = ModelCheckpoint(outdir+'/weights.h5', monitor='val_loss', save_best_only=True)

    model.fit(train_data, train_masks, batch_size=batch_size, epochs=epochs,
              verbose=1, shuffle=True,
              callbacks=[model_checkpoint],
              validation_data=(test_data, test_masks))

    # new_model = load_model(outdir+'/weights.h5')

    return 0

if __name__ == "__main__":
    main()
