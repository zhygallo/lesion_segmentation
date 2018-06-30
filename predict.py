from __future__ import print_function

import numpy as np
import nibabel as nib
import os
import json
import click
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
K.set_image_data_format('channels_last')

from models.HighRes3DNet import get_model
from losses import dice_coef_loss
from metrics import dice_coef, recall, f1_score
from data_gen import get_stride_patches


def predict_on_strides(input_fold, model, crop_shape, strides=(1, 1, 1), normalize=True, output_fold=''):
    img_files = os.listdir(os.path.join(input_fold, 'images'))
    mask_files = os.listdir(os.path.join(input_fold, 'masks'))

    img_count = len(img_files)
    assert (img_count == len(mask_files))

    file_names = []
    true_masks = []
    pred_masks = []

    for ind, img_file in enumerate(img_files):
        img = nib.load(os.path.join(input_fold, 'images/' + img_file)).get_data()
        mask = nib.load(os.path.join(input_fold, 'masks/' + img_file.split('.')[0] + '_mask.nii')).get_data()
        assert img.shape == mask.shape
        img = np.nan_to_num(img)
        mask = np.nan_to_num(mask)
        pred_mask = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
        for row in range(0, (img.shape[0] - crop_shape[0]) // strides[0], strides[0]):
            for col in range(0, (img.shape[1] - crop_shape[1]) // strides[1], strides[1]):
                for depth in range(0, (img.shape[2] - crop_shape[2]) // strides[2], strides[2]):
                    crop_img = img[row:row + crop_shape[0], col:col + crop_shape[1], depth:depth + crop_shape[2]]
                    crop_img = np.reshape(crop_img, (1, crop_img.shape[0], crop_img.shape[1], crop_img.shape[2], 1))
                    if normalize == True:
                        crop_img -= np.mean(crop_img)
                        std = np.std(crop_img)
                        if std != 0:
                            crop_img /= np.std(crop_img)
                    pred_mask[row:row+crop_shape[0], col:col+crop_shape[1], depth:depth+crop_shape[2]] += model.predict(crop_img)[0][:, :, :, 1].round()
        pred_mask = np.clip(pred_mask, 0, 1)
        file_names.append(img_file)
        true_masks.append(mask)
        pred_masks.append(pred_mask)
    return file_names, true_masks, pred_masks

def get_dice_coef(files, true_masks, pred_masks):
    result = {}
    for file, true_mask, pred_mask in zip(files, true_masks, pred_masks):
        result[file] = 2 * np.sum(true_mask * pred_mask) /  (np.sum(true_mask) + np.sum(pred_mask))
    return result

@click.command()
@click.argument('test_folder')
def main(test_folder):

    crop_shape = (32, 32, 32)
    learning_rate = 1e-4
    model = get_model(crop_shape)
    model.load_weights('weights.h5')
    model.compile(optimizer=Adam(lr=learning_rate), loss=dice_coef_loss,
                  metrics=[dice_coef, recall, f1_score])

    keep_label = 1
    strides = (10, 10, 10)
    test_data, test_masks = get_stride_patches(test_folder, crop_shape, keep_label, strides)

    loss, dice, rec, f1 = model.evaluate(x=test_data, y=test_masks)

    print(loss)
    print(dice)
    print(rec)
    print(f1)

    # input_fold = '/home/zhygallo/zhygallo/tum/GuidedResearch/lesion_segmentation/raw_data_test/'
    # strides = (5, 5, 5)
    # file_names, true_masks, pred_masks = predict_on_strides(input_fold, model, crop_shape, strides)
    # dice_coef_result = get_dice_coef(file_names, true_masks, pred_masks)
    # with open('output.json', 'w+') as of:
    #     json.dump(dice_coef_result, of)

if __name__ == "__main__":
    main()