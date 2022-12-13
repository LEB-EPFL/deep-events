from PIL import Image
import pandas as pd
import numpy as np
import os, sys
from albumentations import Compose, Rotate, RandomRotate90, HorizontalFlip, Flip, VerticalFlip
from myfunctions import augStack
from skimage.filters import threshold_otsu, threshold_local
import matplotlib.pyplot as plt
import random
from skimage.filters import threshold_otsu, threshold_local,rank



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





def import_aug_fun(joinpath, fdir, imdir,sigma_chosen, numaug):
        ## this should conserve memory##
    data_gauss=np.zeros((256,256))[None, :, :]
    data =np.zeros((256,256))[None, :, :]
    aug_data = np.zeros((256,256))[None, :, :]
    g_data = np.zeros((256,256))[None, :, :]

    for image_file in os.listdir(joinpath):
        image, date, cell_type, bf_fl, index, number_gauss  = image_file.split('_')
        joined_image_path = os.path.join(fdir, imdir, image_file)


        if 'gauss' in number_gauss:
            i,d,ct,dy,n,sigma,ng = image_file.split('_')
            s= [int(k) for k in sigma if k.isdigit()]
            sigma=s[0]
            if sigma==sigma_chosen:
                img_gauss = Image.open(joined_image_path)
                image_array_gauss = np.zeros((img_gauss.n_frames,256,256))
                for i in range(0, img_gauss.n_frames):
                    img_gauss.seek(i)
                    image_array_gauss[i,:,:] = np.array(img_gauss)

                norm_gauss = normalization_fun_g(image_array_gauss, 0.1)

                for n in range(1,numaug+1):
                    g_data = np.concatenate(norm_gauss)
                g_data= np.delete(g_data, 0, axis=0)
                data_gauss = np.concatenate([data_gauss, norm_gauss])

        else:
            img = Image.open(joined_image_path)
            image_array = np.zeros((img.n_frames,256,256))
            for i in range(0,img.n_frames):
                img.seek(i)
                image_array[i,:,:] = np.array(img)

            norm_data= normalization_fun_g(image_array, 0.1)

            for n in range(1,numaug+1):
                p_rot = random.randint(0, 1) #generate random transform probabilities per file
                p_rot9 = random.randint(0, 1)
                p_hor = random.randint(0, 1)
                p_flip = random.randint(0, 1)
                p_vert = random.randint(0, 1)
                transform = Compose([Rotate(limit=45, p=p_rot), RandomRotate90(p=p_rot9), HorizontalFlip(p=p_hor), Flip(p=p_flip), VerticalFlip(p=p_vert)])
                aug_data = np.concatenate(augStack(norm_data, transform, sigma=8))

            aug_data= np.delete(aug_data, 0, axis=0)
            data = np.concatenate([data, aug_data])



    data= np.delete(data, 0, axis=0) #removes the elements in the first axis which were just zeros
    data_gauss= np.delete(data_gauss, 0, axis=0)

    return data, data_gauss

def normalization_fun_loc(data_first, k,ofs,perc):
    final_loc_bin=np.zeros((np.size(data_first,0),np.size(data_first,1),np.size(data_first,2)))
    kk=1/(1-perc)

    for framenumber in range(np.size(data_first, 0)):
        data_g = (data_first[framenumber])/(np.max(data_first[framenumber]))
        data_g = data_g-perc
        data_g[data_g < 0] = 0
        #data_g_norm[framenumber] = data_g*kk
        image_loc_bin = data_g*kk
        #image_loc_bin= data_first[framenumber]
        local_thresh = threshold_local(image_loc_bin, k, method='gaussian', offset=ofs)
        image_loc_bin[image_loc_bin < local_thresh] = 0
        final_loc_bin[framenumber,:,:]=image_loc_bin

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