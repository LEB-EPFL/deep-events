from PIL import Image
import pandas as pd
from tqdm import tqdm
import numpy as np
import os, sys
from albumentations import Compose, Rotate, RandomRotate90, HorizontalFlip, Flip, VerticalFlip
import random
from skimage.filters import threshold_otsu, threshold_local,rank
from myfunctions import augStack,augImg



def import_fun(joinpath, fdir, imdir,sigma_chosen):
        ## this creates one aray of 3 columns and fills it will all the images ##
    all_image_array=np.zeros((256,256))[None, :, :]
    all_image_array_gauss=np.zeros((256,256))[None, :, :]

    for image_file in os.listdir(joinpath):
        # image, date, cell_type, bf_fl, index, number_gauss  = image_file.split('_')
        joined_image_path = os.path.join(fdir, imdir, image_file)

        if 'gauss' in image_file:
            if 'neg' in image_file:
                i,d,ct,dy,n,sigma,neg,ng = image_file.split('_')
            else:
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

    aug_data, aug_data_gauss= augStack(augmentation_data, augmentation_data_gauss, transform)
    return aug_data, aug_data_gauss

# def aug_fun_sep(augmentation_data, augmentation_data_gauss, transform):
#     aug_input_data = np.zeros(augmentation_data.shape, dtype=np.float64)
#     aug_output_data = np.zeros(augmentation_data_gauss.shape, dtype=np.float32)

#     for i in tqdm(range(augmentation_data.shape[0]), total=augmentation_data.shape[0]):
#         aug_input_data[i], aug_output_data[i]= augImg(augmentation_data[i], augmentation_data_gauss[i], transform[i])
#     return aug_input_data, aug_output_data

def normalization_fun_loc(data_first, k,ofs, perc, bf_fl):
    final_loc_bin=np.zeros((np.size(data_first,0),np.size(data_first,1),np.size(data_first,2)))
    kk=1/(1-perc)
    if bf_fl=='fl':
        for framenumber in range(np.size(data_first, 0)):
            data_g = (data_first[framenumber])/(np.max(data_first[framenumber])) 
            data_g = data_g-perc
            data_g[data_g < 0] = 0     
            image_loc_bin = data_g*kk 
            local_thresh = threshold_local(image_loc_bin, k, method='gaussian', offset=ofs)
            image_loc_bin[image_loc_bin < local_thresh] = 0
            final_loc_bin[framenumber,:,:] = image_loc_bin
    elif bf_fl=='bf':
        for framenumber in range(np.size(data_first, 0)):
            data_g = (data_first[framenumber])/(np.max(data_first[framenumber])) 
            data_g = data_g - perc
            data_g[data_g < 0] = 0
            final_loc_bin[framenumber,:,:]= data_g*kk 

    return final_loc_bin

def normalization_fun_glob(data_first):
    final_glob=np.zeros((np.size(data_first,0),np.size(data_first,1),np.size(data_first,2)))

    for framenumber in range(np.size(data_first, 0)):
        image_glob= data_first[framenumber]
        threshold= threshold_otsu(image_glob)
        image_glob[image_glob < threshold] = 0
        final_glob[framenumber,:,:]= image_glob
        image_glob= data_first[framenumber]
        threshold= threshold_otsu(image_glob)
        image_glob[image_glob < threshold] = 0
        final_glob[framenumber,:,:]= image_glob

    return final_glob


def normalization_fun_g(data_second):
    data_g_norm= data_second
    k=0.1
    kk=1/(1-k)

    for framenumber in range(np.size(data_second, 0)):
        data_g = (data_second[framenumber])/(np.max(data_second))
        data_g = data_g-k
        data_g[data_g < 0] = 0
        data_g_norm[framenumber] = data_g*kk

    return data_g_norm



def import_aug_fun(joinpath, fdir, imdir, sigma_chosen):
        ## this should conserve memory##
    all_image_array=np.zeros((3,256,256))[None,:, :, :]
    all_image_array_gauss=np.zeros((3,256,256))[None,:, :, :]

    for image_file in os.listdir(joinpath):
        joined_image_path = os.path.join(fdir, imdir, image_file)
        image = Image.open(joined_image_path)
        framenum=image.n_frames

        if framenum>2:
            if 'gauss' in image_file:
                if 'neg' in image_file:
                    i,d,ct,dy,n,sigma,neg,ng = image_file.split('_')
                else:
                    i,d,ct,dy,n,sigma,ng = image_file.split('_')
                s= [int(k) for k in sigma if k.isdigit()]
                sigma=s[0]
                if sigma==sigma_chosen:
                    image_array_gauss = np.zeros((1,3,256,256))
                    for i in range(0, framenum-2):
                        k=0
                        for i in range(i, i+3):
                            image.seek(i)
                            image_array_gauss[0,k,:,:] = np.array(image)
                            k=k+1
                        all_image_array_gauss= np.concatenate([all_image_array_gauss, image_array_gauss])
            else:
                image_array = np.zeros((1,3,256,256))
                for i in range(0, framenum-2):
                    k=0
                    for i in range(i, i+3):
                        image.seek(i)
                        all_image_array[0,k,:,:] = np.array(image)
                        k=k+1
                    all_image_array= np.concatenate([all_image_array, image_array])


    all_image_array= np.delete(all_image_array, 0, axis=0) #removes the elements in the first axis which were just zeros
    all_image_array_gauss= np.delete(all_image_array_gauss, 0, axis=0)

    return all_image_array, all_image_array_gauss
