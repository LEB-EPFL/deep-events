from PIL import Image
import pandas as pd
import numpy as np
import os, sys
from os import path
from glob import glob
from sklearn.model_selection import train_test_split
from albumentations import Compose, Rotate, RandomRotate90, HorizontalFlip, Flip, ElasticTransform, GaussNoise, RandomCrop, Resize
from myfunctions import augStack, augImg
import matplotlib.pyplot as plt 
import random

def import_fun(joinpath, fdir, imdir):
        ## this creates one aray of 3 columns and fills it will all the images ##
    all_image_array=np.zeros((256,256))[None, :, :]
    all_image_array_gauss=np.zeros((256,256))[None, :, :]

    for image_file in os.listdir(joinpath):
        image, date, cell_type, bf_fl, index, number_gauss  = image_file.split('_')
        joined_image_path = os.path.join(fdir, imdir, image_file)

        if 'gauss' in number_gauss:
            img_gauss = Image.open(joined_image_path)
            image_array_gauss = np.zeros((img_gauss.n_frames,256,256))
            for i in range(0, img_gauss.n_frames-1):
                img_gauss.seek(i)
                image_array_gauss[i,:,:] = np.array(img_gauss)
            all_image_array_gauss = np.concatenate([all_image_array_gauss, image_array_gauss])
        else:
            img = Image.open(joined_image_path)
            image_array = np.zeros((img.n_frames,256,256))
            for i in range(0,img.n_frames-1):
                img.seek(i)
                image_array[i,:,:] = np.array(img)
            all_image_array = np.concatenate([all_image_array, image_array])
    all_image_array= np.delete(all_image_array, 1, axis=0) #removes the elements in the first axis which were just zeros
    all_image_array_gauss= np.delete(all_image_array_gauss, 1, axis=0)
    return all_image_array, all_image_array_gauss


def aug_fun(data_set_test_trainvalid_ratio, imarray, imarray_gauss, img_size):
    data_split_state = None   # Random split on each call
    augmentation_data, validation_data, augmentation_data_gauss, validation_data_gauss =  train_test_split(imarray, imarray_gauss, 
                                                                                                       test_size=data_set_test_trainvalid_ratio, random_state=data_split_state)
    transform = Compose([RandomRotate90(p=0.5), HorizontalFlip(p=0.5), Flip(p=0.5)])
    aug_data, aug_data_gauss= augStack(augmentation_data, augmentation_data_gauss,transform, sigma=8)
    return aug_data, aug_data_gauss, validation_data, validation_data_gauss

def import_aug_fun(joinpath, fdir, imdir, data_set_test_trainvalid_ratio,img_size):
        ## this should conserve memory##
    all_image_array_val=np.zeros((256,256))[None, :, :]
    all_image_array_gauss_aug=np.zeros((256,256))[None, :, :]
    all_image_array_val=np.zeros((256,256))[None, :, :]
    all_image_array_gauss_aug=np.zeros((256,256))[None, :, :]

    data_split_state = None   # Random split on each call
    

    for image_file in os.listdir(joinpath):
        image, date, cell_type, bf_fl, index, number_gauss  = image_file.split('_')
        joined_image_path = os.path.join(fdir, imdir, image_file)

        p_rot = random.randint(0, 1) #generate random transform probabilities per file
        p_hor = random.randint(0, 1)
        p_flip = random.randint(0, 1)
        transform = Compose([RandomRotate90(p=p_rot), HorizontalFlip(p=p_hor), Flip(p=p_flip)])

        if 'gauss' in number_gauss:
            img_gauss = Image.open(joined_image_path)
            image_array_gauss = np.zeros((img_gauss.n_frames,256,256))
            for i in range(0, img_gauss.n_frames-1):
                img_gauss.seek(i)
                image_array_gauss[i,:,:] = np.array(img_gauss)
            augmentation_data_gauss, validation_data_gauss = train_test_split(image_array_gauss, test_size=data_set_test_trainvalid_ratio, random_state=data_split_state)
            aug_data_gauss= augStack(augmentation_data_gauss,transform, sigma=8)
            all_image_array_gauss_aug = np.concatenate([all_image_array_gauss_aug, aug_data_gauss])
            all_image_array_gauss_val = np.concatenate([all_image_array_gauss_val, validation_data_gauss])
        else:
            img = Image.open(joined_image_path)
            image_array = np.zeros((img.n_frames,256,256))
            for i in range(0,img.n_frames-1):
                img.seek(i)
                image_array[i,:,:] = np.array(img)
            augmentation_data, validation_data = train_test_split(image_array, test_size=data_set_test_trainvalid_ratio, random_state=data_split_state)
            aug_data= augStack(augmentation_data,transform, sigma=8)
            all_image_array_aug = np.concatenate([all_image_array_aug, aug_data])
            all_image_array_val = np.concatenate([all_image_array_val, validation_data])

        
    all_image_array= np.delete(all_image_array, 1, axis=0) #removes the elements in the first axis which were just zeros
    all_image_array_gauss= np.delete(all_image_array_gauss, 1, axis=0)
    return all_image_array_gauss_aug, all_image_array_gauss_val, all_image_array_aug, all_image_array_val
