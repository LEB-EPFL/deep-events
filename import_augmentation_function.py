from PIL import Image
import pandas as pd
import numpy as np
from glob import glob
from os import path
from sklearn.model_selection import train_test_split
from albumentations import Compose, Rotate, RandomRotate90, HorizontalFlip, Flip, ElasticTransform, GaussNoise, RandomCrop, Resize
from myfunctions import augStack, augImg
import matplotlib.pyplot as plt 


def import_fun(joinpath, fdir, imdir):
    all_image_array=np.zeros((256,256))[None, :, :]
    all_image_array_gauss=np.zeros((256,256))[None, :, :]

    for image_file in os.listdir(joinpath):
        image, date, cell_type, bf_fl, index, number_gauss  = image_file.split('_')
        joined_image_path = os.path.join(fdir, imdir, image_file)

        if 'gauss' in number_gauss:
            img_gauss = Image.open(joined_image_path)
            image_array_gauss = np.zeros((img_gauss.n_frames,256,256))
            for i in range(img_gauss.n_frames-1):
                img_gauss.seek(i)
                image_array_gauss[i,:,:] = np.array(img_gauss)
            all_image_array_gauss = np.concatenate([all_image_array_gauss, image_array_gauss])
        else:
            img = Image.open(joined_image_path)
            image_array = np.zeros((img.n_frames,256,256))
            for i in range(img.n_frames-1):
                img.seek(i)
                image_array[i,:,:] = np.array(img)
            all_image_array = np.concatenate([all_image_array, image_array])
    return all_image_array, all_image_array_gauss

def aug_fun(data_set_test_trainvalid_ratio, imarray, imarray_gauss, img_size):
    data_split_state = None   # Random split on each call
    augmentation_data, validation_data, augmentation_data_gauss, validation_data_gauss =  train_test_split(imarray, imarray_gauss, 
                                                                                                       test_size=data_set_test_trainvalid_ratio, random_state=data_split_state)
    transform = Compose([RandomRotate90(p=0.5), HorizontalFlip(p=0.5), Flip(p=0.5)])
    aug_data, aug_data_gauss= augStack(augmentation_data, augmentation_data_gauss,transform, sigma=8)
    return aug_data, aug_data_gauss, validation_data, validation_data_gauss
