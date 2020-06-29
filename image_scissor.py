import cv2
import os
from os import listdir
from os.path import join, split
import numpy as np


def image_sciss(image_file, left_savepath, right_savepath):
    print('-- current image :' + image_file + " --")
    stacked = cv2.imread(image_file)
    print(stacked.shape)
    left_img = stacked[:1024, :, :]
    right_img = stacked[1024:, :, :]
    path, file = split(image_file)

    cv2.imwrite(join(left_savepath, file), left_img)
    cv2.imwrite(join(right_savepath, file), right_img)




def image_scissor(path):
    rootpath = path
    keyframe_list = [join(rootpath, kf) for kf in listdir(rootpath) if ('keyframe' in kf and 'ignore' not in kf)]
    for kf in keyframe_list:
        print(kf)
        stacked_filepath = join(rootpath, kf) + '/data/rgb_data'
        stacked_filelist = [sf for sf in listdir(stacked_filepath) if '.png' in sf]
        left_savepath = join(rootpath, kf) + '/data/left'
        right_savepath = join(rootpath, kf) + '/data/right'
        os.makedirs(left_savepath, exist_ok=True)
        os.makedirs(right_savepath, exist_ok=True)

        for sf in stacked_filelist:
            image_file = join(stacked_filepath, sf)
            image_sciss(image_file, left_savepath, right_savepath)


if __name__ == '__main__':
    #path = '/media/10TB/dataset3'
    rootpath = '/media/10TB/EndoVis_depth/dataset_7'
    image_scissor(rootpath)

