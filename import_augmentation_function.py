from PIL import Image
import pandas as pd
import numpy as np
import os, sys
from os import path
from glob import glob
from sklearn.model_selection import train_test_split
from albumentations import Compose, Rotate, RandomRotate90, HorizontalFlip, Flip, ElasticTransform, GaussNoise, RandomCrop, Resize
from myfunctions import augStack, augImg, augStack_one, augImg_one
import matplotlib.pyplot as plt 
import random

def import_fun(joinpath, fdir, imdir,sigma_chosen):
        ## this creates one aray of 3 columns and fills it will all the images ##
    all_image_array=np.zeros((256,256))[None, :, :]
    all_image_array_gauss=np.zeros((256,256))[None, :, :]

    for image_file in os.listdir(joinpath):
        # image, date, cell_type, bf_fl, index, number_gauss  = image_file.split('_')
        joined_image_path = os.path.join(fdir, imdir, image_file)
        
        if 'gauss' in image_file:
            i,d,ct,dy,n,sigma,ng = image_file.split('_')
            s= [int(k) for k in sigma if k.isdigit()]
            sigma=s[0]
            if sigma==sigma_chosen:
                img_gauss = Image.open(joined_image_path)
                image_array_gauss = np.zeros((img_gauss.n_frames,256,256))
                for i in range(0, img_gauss.n_frames):
                    img_gauss.seek(i)
                    image_array_gauss[i,:,:] = np.array(img_gauss)
                all_image_array_gauss = np.concatenate([all_image_array_gauss, image_array_gauss])
        else:
            img = Image.open(joined_image_path)
            image_array = np.zeros((img.n_frames,256,256))
            for i in range(0,img.n_frames):
                img.seek(i)
                image_array[i,:,:] = np.array(img)
            all_image_array = np.concatenate([all_image_array, image_array])

    all_image_array= np.delete(all_image_array, 0, axis=0) #removes the elements in the first axis which were just zeros
    all_image_array_gauss= np.delete(all_image_array_gauss, 0, axis=0)
    return all_image_array, all_image_array_gauss


def aug_fun(augmentation_data, augmentation_data_gauss, transform):
    
    aug_data, aug_data_gauss= augStack(augmentation_data, augmentation_data_gauss, transform, sigma=8)
    return aug_data, aug_data_gauss





def import_aug_fun(joinpath, fdir, imdir):
        ## this should conserve memory##
    all_image_array_gauss=np.zeros((256,256))[None, :, :]
    all_image_array=np.zeros((256,256))[None, :, :]
    all_data_gauss_aug=np.zeros((256,256))[None, :, :]
    all_data_aug=np.zeros((256,256))[None, :, :]
    

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
            #non augmented data#
            all_image_array_gauss = np.concatenate([all_image_array_gauss, image_array_gauss])
            #augmented data#
            data_gauss_aug = augStack_one(image_array_gauss, transform, sigma=8)
            all_data_gauss_aug = np.concatenate([all_data_gauss_aug, data_gauss_aug])
            
        else:
            img = Image.open(joined_image_path)
            image_array = np.zeros((img.n_frames,256,256))
            for i in range(0,img.n_frames-1):
                img.seek(i)
                image_array[i,:,:] = np.array(img)
            #non augmented data#
            all_image_array = np.concatenate([all_image_array, image_array])
            #augmented data#
            data_aug= augStack_one(image_array,transform, sigma=8)
            all_data_aug = np.concatenate([all_data_aug, data_aug])

        #augmentation_data, validation_data, augmentation_data_gauss, validation_data_gauss =  train_test_split(, imarray_gauss, 
                                                                                                       #test_size=data_set_test_trainvalid_ratio, random_state=data_split_state)  
        #all_image_array_gauss_val = np.concatenate([all_image_array_gauss_val, validation_data_gauss])
        #all_image_array_val = np.concatenate([all_image_array_val, validation_data])

        
    all_image_array= np.delete(all_image_array, 1, axis=0) #removes the elements in the first axis which were just zeros
    all_image_array_gauss= np.delete(all_image_array_gauss, 1, axis=0)
    return all_data_aug, all_data_gauss_aug, all_image_array, all_image_array_gauss


    
# def normalization_fun(data_val, data_aug, data_gauss_val, data_gauss_aug,k):
#     data_aug_norm = data_aug 
#     data_gauss_aug_norm= data_gauss_aug
#     data_val_norm = data_val 
#     data_gauss_val_norm= data_gauss_val
#     # k is the bit of data that we want to set to background and this should be reconsidered since maybe just quoting a number isn't very productive #
#     kk=1/(1-k)

#     for framenumber in range(np.size(data_val, 0)):

#         # validation data #
#         data_vall = (data_val[framenumber])/(np.max(data_val[framenumber]))                                           
#         data_vall = data_vall-k
#         data_vall[data_vall < 0] = 0   
#         data_val_norm[framenumber] = data_vall*kk

#         data_gauss_vall = (data_gauss_val[framenumber])/(np.max(data_gauss_val)) 
#         data_gauss_vall = data_gauss_vall-k
#         data_gauss_vall[data_gauss_vall < 0] = 0     
#         data_gauss_val_norm[framenumber] = data_gauss_vall*kk                                             
    

#     for framenumber in range(np.size(data_aug,0)):

#         # augmentation data #
#         data_augg = (data_aug[framenumber])/(np.max(data_aug))                                               
#         data_augg = data_augg-k
#         data_augg[data_augg < 0] = 0   
#         data_aug_norm[framenumber] = data_augg*kk

#         data_gauss_augg = (data_gauss_aug[framenumber])/(np.max(data_gauss_aug))
#         data_gauss_augg = data_gauss_augg-k
#         data_gauss_augg[data_gauss_augg < 0] = 0     
#         data_gauss_aug_norm[framenumber] = data_gauss_augg*kk  
    
#     return data_val_norm, data_aug_norm, data_gauss_val_norm, data_gauss_aug_norm

def normalization_fun(data_first, k):
    data_norm = data_first
    kk=1/(1-k)

    for framenumber in range(np.size(data_first, 0)):

        # validation data #
        data = (data_first[framenumber])/(np.max(data_first[framenumber]))                                             
        data = data-k
        data[data < 0] = 0   
        data_norm[framenumber] = data*kk

    return data_norm

    
def normalization_fun_g(data_second, k):
    data_g_norm= data_second
    kk=1/(1-k)

    for framenumber in range(np.size(data_second, 0)):
        data_g = (data_second[framenumber])/(np.max(data_second)) 
        data_g = data_g-k
        data_g[data_g < 0] = 0     
        data_g_norm[framenumber] = data_g*kk   
    return data_g_norm                                


def import_fun_neg(joinpath, fdir, imdir):
        ## this creates one aray of 3 columns and fills it will all the images ##
    all_image_array_neg=np.zeros((256,256))[None, :, :]

    for image_file in os.listdir(joinpath):
        # image, date, cell_type, bf_fl, index, number_gauss  = image_file.split('_')
        joined_image_path = os.path.join(fdir, imdir, image_file)
        
        img = Image.open(joined_image_path)
        image_array = np.zeros((img.n_frames,256,256))
        for i in range(0,img.n_frames):
            img.seek(i)
            image_array[i,:,:] = np.array(img)
        all_image_array_neg= np.concatenate([all_image_array_neg, image_array])

    all_image_array_neg= np.delete(all_image_array_neg, 0, axis=0) #removes the elements in the first axis which were just zeros
    all_image_array_gauss_neg=np.zeros((np.shape(all_image_array_neg)))
    return all_image_array_neg, all_image_array_gauss_neg