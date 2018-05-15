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

def gen_train_test_folds(input_fold, output_fold, test_list = None):
    for root, subfolders, files in os.walk(input_fold):
        for folder in subfolders:
            subfold = 'train/'
            if folder in test_list:
                subfold = 'test/'
            for file in os.listdir(os.path.join(root,folder)):
                if file == 'rmf2.nii':
                    old_full_name = os.path.join(root, os.path.join(folder, file))
                    new_full_name = os.path.join(output_fold + subfold + 'images/', folder + '.' + file.split('.')[1])
                    shutil.copy(old_full_name, new_full_name)
                elif file == 'sicher_vergleich.nii':
                    old_full_name = os.path.join(root, os.path.join(folder, file))
                    new_full_name = os.path.join(output_fold + subfold + 'masks/', folder + '_mask' + '.' + file.split('.')[1])
                    shutil.copy(old_full_name, new_full_name)

def main():
    # input_fold = '/home/zhygallo/zhygallo/tum/GuidedResearch/MS_database/controls/'
    # output_img_fold = '/home/zhygallo/zhygallo/tum/GuidedResearch/lesion_segmentation/raw_data_no_control/train/images/'
    # output_mask_fold = '/home/zhygallo/zhygallo/tum/GuidedResearch/lesion_segmentation/raw_data_no_control/train/masks/'
    #
    # gen_img_mask_folders(input_fold, output_img_fold, output_mask_fold)

    input_fold = '/home/zhygallo/zhygallo/tum/GuidedResearch/MS_database/ms_patients/'
    output_fold = '/home/zhygallo/zhygallo/tum/GuidedResearch/lesion_segmentation/raw_data_no_control/'
    test_list = ['m_773608_20090804',
                    'm_797185_20100217',
                    'm_813275_20090206',
                    'm_860816_20090604',
                    'm_884443_20091110',
                    'm_902675_20100302',
                    'm_919034_20100114',
                    'm_971122_20100112',
                    'm_986873_20091209',
                    'm_576379_20090428',
                    'm_659205_20091220',
                    'm_691905_20090421',
                    'm_727543_20091104',
                    'm_765559_20090309',

                    'm_761101_20080828',
                    'm_770822_20080715',
                    'm_860939_20081010',
                    'm_874572_20081216',
                    'm_891684_20081120'
                 ]
    gen_train_test_folds(input_fold, output_fold, test_list=test_list)

    return 0

if __name__ == "__main__":
    main()