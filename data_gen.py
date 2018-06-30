from __future__ import print_function
from __future__ import division

import os
import numpy as np
import random
import nibabel as nib


def get_random_patches(input_fold, crop_shape, keep_label=1, patch_per_img=5, normalize=True, output_fold=''):
    img_files = os.listdir(os.path.join(input_fold, 'images'))
    mask_files = os.listdir(os.path.join(input_fold, 'masks'))

    img_count = len(img_files)
    assert (img_count == len(mask_files))

    imgs = np.zeros((patch_per_img * img_count, crop_shape[0], crop_shape[1], crop_shape[2], 1))
    masks = np.zeros((patch_per_img * img_count, crop_shape[0], crop_shape[1], crop_shape[2], 1))

    count = 0
    for file_ind, img_file in enumerate(img_files):
        img = nib.load(os.path.join(input_fold, 'images/' + img_file)).get_data()
        mask = nib.load(os.path.join(input_fold, 'masks/' + img_file.split('.')[0] + '_mask.nii')).get_data()
        assert img.shape == mask.shape
        img = np.nan_to_num(img)
        mask = np.nan_to_num(mask)
        mask[mask==keep_label] = 1
        mask[mask!=keep_label] = 0
        for patch_ind in range(patch_per_img):
            row = random.randint(0, img.shape[0] - crop_shape[0])
            col = random.randint(0, img.shape[1] - crop_shape[1])
            dep = random.randint(0, img.shape[2] - crop_shape[2])

            crop_img = img[row:row + crop_shape[0], col:col + crop_shape[1], dep:dep + crop_shape[2]]
            crop_mask = mask[row:row + crop_shape[0], col:col + crop_shape[1], dep:dep + crop_shape[2]]
            crop_img = np.reshape(crop_img, (crop_img.shape[0], crop_img.shape[1], crop_img.shape[2], 1))
            crop_mask = np.reshape(crop_mask, (crop_mask.shape[0], crop_mask.shape[1], crop_mask.shape[2], 1))
            if normalize == True:
                std = np.std(crop_img)
                crop_img -= np.mean(crop_img)
                if std != 0:
                    crop_img /= np.std(crop_img)
            imgs[count] = crop_img
            masks[count] = crop_mask
            count += 1

    if output_fold != '':
        np.save(os.path.join(output_fold, 'images'), imgs)
        np.save(os.path.join(output_fold, 'masks'), masks)

    return imgs, masks


def get_stride_patches(input_fold, crop_shape, keep_label=1, strides=(1, 1, 1), normalize=True, output_fold=''):
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
        mask[mask==keep_label] = 1
        mask[mask!=keep_label] = 0
        num_rows = (img.shape[0] - crop_shape[0]) // strides[0]
        num_cols = (img.shape[1] - crop_shape[1]) // strides[1]
        num_dep = (img.shape[2] - crop_shape[2]) // strides[2]
        num_crops = num_rows * num_cols * num_dep
        crop_img_np = np.zeros((num_crops, crop_shape[0], crop_shape[1], crop_shape[2], 1))
        crop_mask_np = np.zeros((num_crops, crop_shape[0], crop_shape[1], crop_shape[2], 1))
        num_crop = 0
        for row in range(0, num_rows):
            for col in range(0, num_cols):
                for depth in range(0, num_dep):
                    r = row * strides[0]
                    c = col * strides[1]
                    d = depth * strides[2]
                    crop_img = img[r:r + crop_shape[0], c:c + crop_shape[1], d:d + crop_shape[2]]
                    crop_mask = mask[r:r + crop_shape[0], c:c + crop_shape[1], d:d + crop_shape[2]]
                    crop_img = np.reshape(crop_img, (crop_img.shape[0], crop_img.shape[1], crop_img.shape[2], 1))
                    crop_mask = np.reshape(crop_mask, (crop_mask.shape[0], crop_mask.shape[1], crop_mask.shape[2], 1))
                    if normalize == True:
                        std = np.std(crop_img)
                        crop_img -= np.mean(crop_img)
                        if std != 0:
                            crop_img /= np.std(crop_img)
                    crop_img_np[num_crop] = crop_img
                    crop_mask_np[num_crop] = crop_mask
                    num_crop += 1
                    imgs.append(crop_img)
                    masks.append(crop_mask)

        if output_fold != '':
            np.save(os.path.join(output_fold + 'images', img_file.split('.')[0]), crop_img_np)
            np.save(os.path.join(output_fold + 'masks', img_file.split('.')[0] + '_mask'), crop_mask_np)



    # TODO Fix Memory Error For Small Crop Sizes
    imgs = np.array(imgs)
    masks = np.array(masks)
    #
    # if output_fold != '':
    #     np.save(os.path.join(output_fold, 'images'), imgs)
    #     np.save(os.path.join(output_fold, 'masks'), masks)

    return imgs, masks


def main():
    input_fold_train = 'raw_data/train/'
    input_fold_test = 'raw_data/test/'
    output_fold_train = 'np_rand_crop/train/'
    output_fold_test = 'np_rand_crop/test/'
    crop_shape = (32, 32, 32)
    num_patches = 10
    normilize = True
    keep_label = 1
    get_random_patches(input_fold_train, crop_shape, keep_label, num_patches, normilize, output_fold_train)
    get_random_patches(input_fold_test, crop_shape, keep_label, num_patches, normilize, output_fold_test)

    # input_srtide_fold = '/home/zhygallo/zhygallo/tum/GuidedResearch/lesion_segmentation/raw_data_test/'
    # strides = (10, 10, 10)
    # output_stride_fold = '/home/zhygallo/zhygallo/tum/GuidedResearch/lesion_segmentation/strides_per_image/'
    # get_stride_patches(input_srtide_fold, crop_shape, keep_label, strides, normalize=True, output_fold=output_stride_fold)


if __name__ == '__main__':
    main()
