from __future__ import print_function
from __future__ import division

import os
import numpy as np
import random
import nibabel as nib


def get_random_patches(input_fold, crop_shape, patch_per_img=5, normalize=True, output_fold=''):
    img_files = os.listdir(os.path.join(input_fold, 'images'))
    mask_files = os.listdir(os.path.join(input_fold, 'masks'))

    img_count = len(img_files)
    assert (img_count == len(mask_files))

    imgs = np.zeros((patch_per_img * img_count, crop_shape[0], crop_shape[1], crop_shape[2], 1))
    masks = np.zeros((patch_per_img * img_count, crop_shape[0], crop_shape[1], crop_shape[2], 1))

    for file_ind, img_file in enumerate(img_files):
        img = nib.load(os.path.join(input_fold, 'images/' + img_file)).get_data()
        mask = nib.load(os.path.join(input_fold, 'masks/' + img_file.split('.')[0] + '_mask.nii')).get_data()
        assert img.shape == mask.shape
        img = np.nan_to_num(img)
        mask = np.nan_to_num(mask)
        valid_patch_num = patch_per_img
        while valid_patch_num != 0:
            row = random.randint(0, img.shape[0] - crop_shape[0])
            col = random.randint(0, img.shape[1] - crop_shape[1])
            dep = random.randint(0, img.shape[2] - crop_shape[2])
            crop_img = img[row:row + crop_shape[0], col:col + crop_shape[1], dep:dep + crop_shape[2]]
            crop_mask = mask[row:row + crop_shape[0], col:col + crop_shape[1], dep:dep + crop_shape[2]]
            if crop_mask.sum() == 0.0:
                continue
            crop_img = np.reshape(crop_img, (crop_img.shape[0], crop_img.shape[1], crop_img.shape[2], 1))
            crop_mask = np.reshape(crop_mask, (crop_mask.shape[0], crop_mask.shape[1], crop_mask.shape[2], 1))
            if normalize == True:
                crop_img -= np.mean(crop_img)
                crop_img /= np.std(crop_img)
            imgs[file_ind * (patch_per_img - valid_patch_num)] = crop_img
            masks[file_ind * (patch_per_img - valid_patch_num)] = crop_mask
            valid_patch_num -= 1

    if output_fold != '':
        np.save(os.path.join(output_fold, 'images'), imgs)
        np.save(os.path.join(output_fold, 'masks'), masks)

    return imgs, masks


def get_stride_patches(input_fold, crop_shape, strides=(1, 1, 1), normalize=True, output_fold=''):
    img_files = os.listdir(os.path.join(input_fold, 'images'))
    mask_files = os.listdir(os.path.join(input_fold, 'masks'))

    img_count = len(img_files)

    assert (img_count == len(mask_files))

    imgs = []
    masks = []

    for ind, img_file in enumerate(img_files):
        img = nib.load(os.path.join(input_fold, 'images/' + img_file)).get_data()
        mask = nib.load(os.path.join(input_fold, 'masks/' + img_file.split('.')[0] + '_mask.nii')).get_data()
        assert img.shape == mask.shape
        img = np.nan_to_num(img)
        mask = np.nan_to_num(mask)
        for row in range(0, (img.shape[0] - crop_shape[0]) // strides[0], strides[0]):
            for col in range(0, (img.shape[1] - crop_shape[1]) // strides[1], strides[1]):
                for depth in range(0, (img.shape[2] - crop_shape[2]) // strides[2], strides[2]):
                    crop_img = img[row:row + crop_shape[0], col:col + crop_shape[1], depth:depth + crop_shape[2]]
                    crop_mask = mask[row:row + crop_shape[0], col:col + crop_shape[1], depth:depth + crop_shape[2]]
                    crop_img = np.reshape(crop_img, (crop_img.shape[0], crop_img.shape[1], crop_img.shape[2], 1))
                    crop_mask = np.reshape(crop_mask, (crop_mask.shape[0], crop_mask.shape[1], crop_mask.shape[2], 1))
                    if normalize == True:
                        crop_img -= np.mean(crop_img)
                        crop_img /= np.std(crop_img)
                    imgs.append(crop_img)
                    masks.append(crop_mask)

    # TODO Fix Memory Error For Small Crop Sizes
    imgs = np.array(imgs)
    masks = np.array(masks)

    if output_fold != '':
        np.save(os.path.join(output_fold, 'images'), imgs)
        np.save(os.path.join(output_fold, 'masks'), masks)

    return imgs, masks


def main():
    input_fold_train = 'raw_data_no_control/train/'
    input_fold_test = 'raw_data_no_control/test/'
    output_fold_train = 'np_rand_crop/train/'
    output_fold_test = 'np_rand_crop/test/'
    crop_shape = (96, 96, 96)
    # crop_shape = (128, 128, 64)
    # strides = (10, 10, 10)
    # output_folder = 'numpy_data/'
    # get_stride_patches(input_fold, crop_shape, strides, output_folder)
    num_patches = 30
    normilize = True
    get_random_patches(input_fold_train, crop_shape, num_patches, normilize, output_fold_train)
    get_random_patches(input_fold_test, crop_shape, num_patches, normilize, output_fold_test)


if __name__ == '__main__':
    main()
