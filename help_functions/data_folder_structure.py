from __future__ import print_function
from __future__ import division


import os
import shutil
import random


def gen_img_mask_folders(input_fold, output_img_fold, output_mask_fold):
    for root, subfolders, files in os.walk(input_fold):
        for folder in subfolders:
            for file in os.listdir(os.path.join(root,folder)):
                if file == 'rmf2.nii':
                    old_full_name = os.path.join(root, os.path.join(folder, file))
                    new_full_name = os.path.join(output_img_fold, folder + '.' + file.split('.')[1])
                    shutil.copy(old_full_name, new_full_name)
                elif file == 'sicher_vergleich.nii':
                    old_full_name = os.path.join(root, os.path.join(folder, file))
                    new_full_name = os.path.join(output_mask_fold, folder + '_mask' + '.' + file.split('.')[1])
                    shutil.copy(old_full_name, new_full_name)

def gen_train_test_folds(input_fold, output_fold, train_data_part = 0.8):
    assert 0.0 <= train_data_part <= 1.0

    img_mask_list = []
    for file in os.listdir(os.path.join(input_fold,'images')):
        img_path = os.path.join(os.path.join(input_fold,'images'), file)
        mask_path = os.path.join(os.path.join(input_fold,'masks'), file.split('.')[0] + '_mask.nii')
        img_mask_list.append((img_path, mask_path))

    random.shuffle(img_mask_list)
    num_train_sampls = int(len(img_mask_list)*train_data_part)
    train_data_files = img_mask_list[:num_train_sampls]
    test_data_files = img_mask_list[num_train_sampls:]

    for tuple in train_data_files:
        new_img_path = os.path.join(output_fold, 'train/images/' + tuple[0].split('/')[-1])
        new_mask_path = os.path.join(output_fold, 'train/masks/' + tuple[1].split('/')[-1])
        shutil.copy(tuple[0], new_img_path)
        shutil.copy(tuple[1], new_mask_path)

    for tuple in test_data_files:
        new_img_path = os.path.join(output_fold, 'test/images/' + tuple[0].split('/')[-1])
        new_mask_path = os.path.join(output_fold, 'test/masks/' + tuple[1].split('/')[-1])
        shutil.copy(tuple[0], new_img_path)
        shutil.copy(tuple[1], new_mask_path)

def main():
    input_fold = '/home/zhygallo/zhygallo/tum/GuidedResearch/MS_database/controls/'
    output_img_fold = '/home/zhygallo/zhygallo/tum/GuidedResearch/lesion_segmentation/raw_data/test/images/'
    output_mask_fold = '/home/zhygallo/zhygallo/tum/GuidedResearch/lesion_segmentation/raw_data/test/masks/'

    gen_img_mask_folders(input_fold, output_img_fold, output_mask_fold)

    # input_fold = '/home/zhygallo/Documents/GuidedResearch/MS_database_img_mask_structure'
    # output_fold =  '/home/zhygallo/Documents/GuidedResearch/MS_database_train_test'
    # gen_train_test_folds(input_fold, output_fold, train_data_part=0.8)

    return 0

if __name__ == "__main__":
    main()