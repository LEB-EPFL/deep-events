import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from mitosplit_net import util
from training_functions import create_model,train_model
from albumentations import Compose, Rotate, RandomRotate90, HorizontalFlip, Flip, VerticalFlip
import os
from os import path
import random as r
import math
from import_augmentation_function import import_fun, aug_fun, normalization_fun_loc, normalization_fun_g,import_aug_fun
import tensorflow as tf

data_ratio= 0.1
data_split_state = None


def load_aug_train(files_dir, images_dir,images_neg_dir, sigma, number_of_augmentations,k,ofs,perc):
    date, dye, cell_type, microscope, bf_fl, pos_neg = images_dir.split('_')
    joined_path = os.path.join(files_dir, images_dir)
    joined_path_neg = os.path.join(files_dir, images_neg_dir)

    all_image_array, all_image_array_gauss= import_fun(joined_path, files_dir, images_dir,sigma)
    #all_image_array_gauss, all_image_array= zero_frames(all_image_array.shape[1], all_image_array_gauss, all_image_array)
    all_image_array_neg, all_image_array_gauss_neg = import_fun(joined_path_neg,files_dir, images_neg_dir,sigma)

    all_image_array = np.concatenate((all_image_array,all_image_array_neg))
    all_image_array_gauss = np.concatenate((all_image_array_gauss,all_image_array_gauss_neg))

                            ## NORMALIZATION ##

    norm_image_array= normalization_fun_loc(all_image_array, k, ofs, perc, bf_fl)
    norm_image_array_gauss = normalization_fun_g(all_image_array_gauss)

                            ## AUGMENTATION ##

    augmentation_data, validation_data, augmentation_data_gauss, validation_data_gauss =  train_test_split(norm_image_array, norm_image_array_gauss,
                                                                                                       test_size=data_ratio, random_state=data_split_state)
    data_aug = np.zeros((np.size(augmentation_data , 0),128,128))
    data_gauss_aug = np.zeros((np.size(augmentation_data , 0),128,128))
    data_val = np.zeros((np.size(validation_data , 0),128,128))
    data_gauss_val = np.zeros((np.size(validation_data_gauss , 0),128,128))

    for frame_index in range(np.size(data_aug , 0)):
        data_aug[frame_index,:,:] = augmentation_data[frame_index, 64:192 , 64:192]
        data_gauss_aug[frame_index,:,:] = augmentation_data_gauss[frame_index, 64:192 , 64:192]
    for frame_index in range(np.size(data_val , 0)):
        data_val[frame_index,:,:] = validation_data[frame_index, 64:192 , 64:192]
        data_gauss_val[frame_index,:,:] = validation_data_gauss[frame_index, 64:192 , 64:192]

    for j in range(number_of_augmentations):
        transform = Compose([Rotate(limit=45, p=0.5), RandomRotate90(p=0.5), HorizontalFlip(p=0.5), Flip(p=0.5), VerticalFlip(p=0.5)])
        print('augmentation', j)
        augment_data, augment_data_gauss = aug_fun(augmentation_data, augmentation_data_gauss,transform)
        data_aug = np.concatenate((data_aug, augment_data[:, 64:192 , 64:192]))
        data_gauss_aug = np.concatenate((data_gauss_aug,augment_data_gauss[:, 64:192 , 64:192]))

                                ## SHUFFLE ##

    shuffle_array_val= np.arange(0, data_val.shape[0], 1)
    shuffle_array_aug= np.arange(0, data_aug.shape[0], 1)
    r.shuffle(shuffle_array_val)
    r.shuffle(shuffle_array_aug)

    data_val=data_val[shuffle_array_val]
    data_gauss_val=data_gauss_val[shuffle_array_val]
    data_aug=data_aug[shuffle_array_aug]
    data_gauss_aug=data_gauss_aug[shuffle_array_aug]

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
        nb_filters, firstConvSize, nb_input_channels = 8, 9, 1
        model[model_name] = create_model(nb_filters, firstConvSize, nb_input_channels)
        history[model_name] = train_model(model[model_name], data_aug, data_gauss_aug, b, data_ratio)

    folder_name = list(model.keys())

    util.save_model(model, model_path, [f'model_{cell_type}_{microscope}_{bf_fl}_{sigma}']*len(model), folder_name)
    util.save_pkl(history, model_path, [f'history_{cell_type}_{microscope}_{bf_fl}_{sigma}']*len(model), folder_name)



def load_aug_train_time(files_dir, images_dir,images_neg_dir, sigma, number_of_augmentations,k,ofs,perc):
    date, dye, cell_type, microscope, bf_fl, pos_neg = images_dir.split('_')
    joined_path = os.path.join(files_dir, images_dir)
    joined_path_neg = os.path.join(files_dir, images_neg_dir)

    all_image_array, all_image_array_gauss= import_aug_fun(joined_path, files_dir, images_dir,sigma)
    #all_image_array_gauss, all_image_array= zero_frames(all_image_array.shape[1], all_image_array_gauss, all_image_array)
    all_image_array_neg, all_image_array_gauss_neg = import_aug_fun(joined_path_neg,files_dir, images_neg_dir,sigma)

    all_image_array = np.concatenate((all_image_array,all_image_array_neg))
    all_image_array_gauss = np.concatenate((all_image_array_gauss,all_image_array_gauss_neg))

                            ## NORMALIZATION ##
    norm_image_array= np.zeros(np.shape(all_image_array))
    norm_image_array_gauss= np.zeros(np.shape(all_image_array_gauss))
    for n_vid in range(0,np.size(all_image_array , 0)):
        norm_image_array[n_vid]= normalization_fun_loc(all_image_array[n_vid], k, ofs, perc, bf_fl)
        norm_image_array_gauss[n_vid] = normalization_fun_g(all_image_array_gauss[n_vid])

                            ## SPLIT TRAINING SET AND TEST SET ##
    
    augmentation_data, validation_data, augmentation_data_gauss, validation_data_gauss =  train_test_split(norm_image_array, norm_image_array_gauss,
                                                                                                       test_size=data_ratio, random_state=data_split_state)
                            ## AUGMENTATION ##

    data_aug = np.zeros((np.size(augmentation_data , 0),3,128,128))
    data_gauss_aug = np.zeros((np.size(augmentation_data , 0),3,128,128))
    data_val = np.zeros((np.size(validation_data , 0),3,128,128))
    data_gauss_val = np.zeros((np.size(validation_data , 0),3,128,128))

    for frame_index in range(np.size(data_aug , 0)):
        data_aug[frame_index,:,:,:] = augmentation_data[frame_index,:, 64:192 , 64:192]
        data_gauss_aug[frame_index,:,:,:] = augmentation_data_gauss[frame_index,:, 64:192 , 64:192]
    for frame_index in range(np.size(data_val , 0)):
        data_val[frame_index,:,:,:] = validation_data[frame_index,:, 64:192 , 64:192]
        data_gauss_val[frame_index,:,:,:] = validation_data_gauss[frame_index,:, 64:192 , 64:192]

    for j in range(number_of_augmentations):
        print('augmentation', j)
        augment_data = np.zeros(np.shape(augmentation_data))
        augment_data_gauss = np.zeros(np.shape(augmentation_data))
        for n_vid in range(0,np.size(data_aug , 0)):
            p_rot = r.randint(0, 1) 
            p_rot9 = r.randint(0, 1)
            p_hor = r.randint(0, 1)
            p_flip = r.randint(0, 1)
            p_vert = r.randint(0, 1)
            t = Compose([Rotate(limit=45, p=p_rot), RandomRotate90(p=p_rot9), HorizontalFlip(p=p_hor), Flip(p=p_flip), VerticalFlip(p=p_vert)])
            augment_data[n_vid], augment_data_gauss[n_vid] = aug_fun(augmentation_data[n_vid], augmentation_data_gauss[n_vid],t)
            data_aug = np.concatenate((data_aug, augment_data[:,:, 64:192 , 64:192]))
            data_gauss_aug = np.concatenate((data_gauss_aug,augment_data_gauss[:,:, 64:192 , 64:192]))

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
        nb_filters, firstConvSize, nb_input_channels = 8, 9, 3
        model[model_name] = create_model(nb_filters, firstConvSize, nb_input_channels)
        history[model_name] = train_model(model[model_name], data_aug, data_gauss_aug, b, data_ratio)

    folder_name = list(model.keys())

    util.save_model(model, model_path, [f'model_{cell_type}_{microscope}_{bf_fl}_{sigma}']*len(model), folder_name)
    util.save_pkl(history, model_path, [f'history_{cell_type}_{microscope}_{bf_fl}_{sigma}']*len(model), folder_name)