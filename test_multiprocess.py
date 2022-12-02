import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from mitosplit_net import util
from training_functions import create_model,train_model
from albumentations import Compose, Rotate, RandomRotate90, HorizontalFlip, Flip, VerticalFlip
import os
from os import path
import random as r
import imageio
from import_augmentation_function import import_fun, aug_fun, import_fun_neg, normalization_fun, normalization_fun_g, import_aug_fun
import tensorflow as tf 

data_ratio= 0.1
data_split_state = None


def load_aug_train(files_dir, images_dir,images_neg_dir, sigma, number_of_augmentations):
    date, dye, cell_type, microscope, bf_fl, pos_neg = images_dir.split('_')
    joined_path = os.path.join(files_dir, images_dir)
    joined_path_neg = os.path.join(files_dir, images_neg_dir)

    all_image_array, all_image_array_gauss= import_fun(joined_path, files_dir, images_dir,sigma)
    all_image_array_gauss, all_image_array= zero_frames(all_image_array.shape[1], all_image_array_gauss, all_image_array)
    all_image_array_neg, all_image_array_gauss_neg = import_fun_neg(joined_path_neg,files_dir, images_neg_dir)

    all_image_array = np.concatenate((all_image_array,all_image_array_neg))
    all_image_array_gauss = np.concatenate((all_image_array_gauss,all_image_array_gauss_neg))

    norm_image_array= normalization_fun(all_image_array, 0.1)
    norm_image_array_gauss = normalization_fun_g(all_image_array_gauss, 0.1)

                                    ## CROP ##

    data_crop = np.zeros((np.size(norm_image_array , 0),128,128))
    data_crop_gauss = np.zeros((np.size(norm_image_array_gauss , 0),128,128))

    for frame_index in range(np.size(norm_image_array , 0)):
        data_crop[frame_index,:,:] = norm_image_array[frame_index, 64:192 , 64:192]
        data_crop_gauss[frame_index,:,:] = norm_image_array_gauss[frame_index, 64:192 , 64:192]

                            ## AUGMENTATION ##

    augmentation_data, data_val, augmentation_data_gauss, data_gauss_val =  train_test_split(data_crop, data_crop_gauss,
                                                                                                       test_size=data_ratio, random_state=data_split_state)
    data_aug = augmentation_data
    data_gauss_aug = augmentation_data_gauss
    
    for j in range(number_of_augmentations):
        transform = Compose([Rotate(limit=45, p=0.5), RandomRotate90(p=0.5), HorizontalFlip(p=0.5), Flip(p=0.5), VerticalFlip(p=0.5)])
        print('augmentation', j)
        augment_data, augment_data_gauss = aug_fun(augmentation_data, augmentation_data_gauss,transform)
        data_aug = np.concatenate((data_aug, augment_data))
        data_gauss_aug = np.concatenate((data_gauss_aug,augment_data_gauss))


                                ## SHUFFLE ##

    data_shuffa=np.zeros((data_aug.shape))
    data_g_shuffa=np.zeros((data_gauss_aug.shape))
    data_shuffv=np.zeros((data_val.shape))
    data_g_shuffv=np.zeros((data_gauss_val.shape))
    # produce empty arrays to fill in with the shuffled elements
    shuffle_array_val= np.arange(0, data_val.shape[0], 1)
    shuffle_array_aug= np.arange(0, data_aug.shape[0], 1)
    r.shuffle(shuffle_array_val)
    r.shuffle(shuffle_array_aug)

    for frame_index in range(np.size(data_val , 0)):
        x=shuffle_array_val[frame_index]
        data_shuffv[frame_index]= data_val[x]
        data_g_shuffv[frame_index]= data_gauss_val[x]

    for frame_index in range(np.size(data_aug , 0)):
        x=shuffle_array_aug[frame_index]
        data_shuffa[frame_index]= data_aug[x]
        data_g_shuffa[frame_index]= data_gauss_aug[x]

    data_val=data_shuffv
    data_gauss_val=data_g_shuffv
    data_aug=data_shuffa
    data_gauss_aug=data_g_shuffa

    # name1=f'{cell_type}_{microscope}_{bf_fl}_{sigma}_data_val.tiff'
    # name2=f'{cell_type}_{microscope}_{bf_fl}_{sigma}_data_gauss_val.tiff'
    # name3=f'{cell_type}_{microscope}_{bf_fl}_{sigma}_data_aug.tiff'
    # name4=f'{cell_type}_{microscope}_{bf_fl}_{sigma}_data_gauss_aug.tiff'

    # imageio.mimwrite(name1, (data_val).astype(np.float64))
    # imageio.mimwrite(name2, (data_gauss_val).astype(np.float64))
    # imageio.mimwrite(name3, (data_aug).astype(np.float64))
    # imageio.mimwrite(name4, (data_gauss_aug).astype(np.float64))
    base_dir = r'C:\Users\roumba\Documents\Software\deep-events'
    model_path = base_dir + '\Models'
    
    gpu = tf.config.list_physical_devices('GPU')[0]
    tf.config.experimental.set_memory_growth(gpu, True)
    gpu = tf.device('GPU:0/') 

    with gpu:
        print(gpu)
        nb_filters = 8
        firstConvSize = 9
        batch_size = [8, 16, 32, 256]
        model, history= {}, {}
        
        b=batch_size[1]
        model_name = 'ref_f%i_c%i_b%i'%(nb_filters, firstConvSize, b)
        print('Model:', model_name)
        model[model_name] = create_model(nb_filters, firstConvSize)
        history[model_name] = train_model(model[model_name], data_aug, data_gauss_aug, b, data_ratio)
    
    folder_name = list(model.keys())

    util.save_model(model, model_path, [f'model_{cell_type}_{microscope}_{bf_fl}_{sigma}']*len(model), folder_name)
    util.save_pkl(history, model_path, [f'history_{cell_type}_{microscope}_{bf_fl}_{sigma}']*len(model), folder_name)




def zero_frames(size,datag,dataim):
    frame_check=np.zeros((size,size))
    z_list=[]

    for x in range(np.size(datag,0)):
        frame_check = datag[x]
        if not np.any(frame_check)==True:
            z_list.append(x)

    datag= np.delete(datag, z_list, axis=0)
    dataim= np.delete(dataim, z_list, axis=0)
    return datag,dataim



def load_aug_train_time(files_dir, images_dir,images_neg_dir, sigma, number_of_augmentations):
    date, dye, cell_type, microscope, bf_fl, pos_neg = images_dir.split('_')
    joined_path = os.path.join(files_dir, images_dir)
    #joined_path_neg = os.path.join(files_dir, images_neg_dir)

    data, data_gauss = import_aug_fun(joined_path, files_dir, images_dir,sigma, number_of_augmentations)

    num= int(data.shape[0]*0.1)
    frames=[]
    for i in range(num):
        frames[i]= r.randint(0,data.shape[0]+1)
    for j in range(len(frames)):
        data_aug = data[frames[j]]
        data_gauss_aug = data_gauss[frames[j]]


    data_crop = np.zeros((np.size(data_aug , 0),128,128))
    data_crop_gauss = np.zeros((np.size(data_aug , 0),128,128))
    data_cropp = np.zeros((np.size(data_val , 0),128,128))
    data_cropp_gauss = np.zeros((np.size(data_val , 0),128,128))

    for frame_index in range(np.size(data_aug , 0)):
        data_crop[frame_index,:,:] = data_aug[frame_index, 64:192 , 64:192]
        data_crop_gauss[frame_index,:,:] = data_gauss_aug[frame_index, 64:192 , 64:192]

    for frame_index in range(np.size(data_val , 0)):
        data_cropp[frame_index,:,:] = data_val[frame_index, 64:192 , 64:192]
        data_cropp_gauss[frame_index,:,:] = data_gauss_val[frame_index, 64:192 , 64:192]


    data_aug=data_crop
    data_gauss_aug=data_crop_gauss
    data_val=data_cropp
    data_gauss_val=data_cropp_gauss
        
    name1=f'{cell_type}_{microscope}_{bf_fl}_data_val.tiff'
    name2=f'{cell_type}_{microscope}_{bf_fl}_data_gauss_val.tiff'
    name3=f'{cell_type}_{microscope}_{bf_fl}_data_aug.tiff'
    name4=f'{cell_type}_{microscope}_{bf_fl}_data_gauss_aug.tiff'

    imageio.mimwrite(name1, (data_val).astype(np.float64))
    imageio.mimwrite(name2, (data_gauss_val).astype(np.float64))
    imageio.mimwrite(name3, (data_aug).astype(np.float64))
    imageio.mimwrite(name4, (data_gauss_aug).astype(np.float64))

    base_dir = r'C:\Users\roumba\Documents\Software\deep-events'
    model_path = base_dir + '\Models'

    gpu = tf.config.list_physical_devices('GPU')[0]
    tf.config.experimental.set_memory_growth(gpu, True)
    gpu = tf.device('GPU:0/') 

    with gpu:
        print(gpu)
        nb_filters = 8
        firstConvSize = 9
        batch_size = [8, 16, 32, 256]
        model, history= {}, {}
        
        b=batch_size[1]
        model_name = 'ref_f%i_c%i_b%i'%(nb_filters, firstConvSize, b)
        print('Model:', model_name)
        model[model_name] = create_model(nb_filters, firstConvSize)
        history[model_name] = train_model(model[model_name], data_aug, data_gauss_aug, b, data_ratio)


    

    folder_name = list(model.keys())

    util.save_model(model, model_path, [f'model_{cell_type}_{microscope}_{bf_fl}_{sigma}']*len(model), folder_name)
    util.save_pkl(history, model_path, [f'history_{cell_type}_{microscope}_{bf_fl}_{sigma}']*len(model), folder_name)