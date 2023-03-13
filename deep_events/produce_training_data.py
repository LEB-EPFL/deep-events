from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import path
import xmltodict
from myfunctions import event_separation , image_crop, image_crop_negative, poi
from myfunctions import delete_old_extracted_events
from database.extract_yaml import get_dict
import tifffile
from pathlib import Path
import copy
import os
import sys

BASE_DIR = r'//lebnas1.epfl.ch/microsc125/deep_events'
SAVING_SCHEME = "ws_0.1"


def get_data_from_folder(original_folder):
    files_dir = os.path.join(BASE_DIR, 'original_data')
    joined_path = os.path.join(files_dir, original_folder)
    # size=(2048,2048)
    img,input_name,output_name,datacsv,pixel_size={},{},{},{},{}

    path = Path(joined_path)
    tif_files = sorted(list(path.glob("*.ome.tif")))
    csv_files = sorted(list(path.glob("*.csv")))

    for index, input_file in enumerate(tif_files):
        input_file = str(input_file.name)
        joined_file_path = os.path.join(files_dir, original_folder, input_file)
        img[index] = Image.open(joined_file_path)
        print('Loaded image:'+input_file)
    for index, input_file in enumerate(csv_files):
        input_file = str(input_file.name)
        joined_file_path = os.path.join(files_dir, original_folder, input_file)
        datacsv[index] = pd.read_csv(joined_file_path)
        print('Loaded csv:'+ input_file)
    print('Pixel scaling:',pixel_size)
    data = {'images': img, "labels": datacsv, "tif_files": tif_files,
            'csv_files': csv_files}
    return data

def produce_gaussians(data, path = ""):
    print(f'PATH IN PRODUCE_GAUSSIANS {path}')
    for x in range(len(data['images'])):
        csv=data['labels'][x]
        im=data['images'][x]
        folder_dict = get_dict(path)
        if all([max(csv['axis-1']) < 205,
               max(csv['axis-2']) < 205,
               folder_dict['scale_csv']]):
            print("=== Scaling adjusted!!!")
            print(folder_dict['microscope'])
            factor = folder_dict['ome']['physical_size_x']
            csv['axis-1']=csv['axis-1'] * 1/factor
            csv['axis-2']=csv['axis-2'] * 1/factor
        ##
        for i in range(2,7,2):
            sigma=(i,i)
            s=sigma[0]
            if path:
                original_folder = os.path.basename(path)
            in_name=f'{original_folder}_{x+1}_sigma{s}.tiff'
            framenum=im.n_frames
            im_shape = (im.height, im.width)
            poi(csv,in_name,sigma,im.size,framenum, im_shape, path=path)


def crop_data(data, path = ""):
    if path:
        original_folder = os.path.basename(path)
    base_dir = BASE_DIR
    training_dir = os.path.join(base_dir, 'training_data')
    folder_name = original_folder
    filepath= os.path.join(training_dir , folder_name)

    folder_dict = get_dict(original_folder,os.path.join(BASE_DIR, "original_data"))
    delete_old_extracted_events(folder_dict, os.path.join(training_dir, original_folder))
    event_dict = copy.deepcopy(folder_dict)
    event_dict['type'] = "event"
    del event_dict['extracted_events']

    for x in range(len(data['images'])):
        event_dict['original_file'] = os.path.basename(data['tif_files'][x])
        event_dict['label_file'] = os.path.basename(data['csv_files'][x])
        event_dict['event_content'] = 'division'

        csv=data['labels'][x]
        im=data['images'][x]
        print('file number:',x+1)
        list1, event_dict = event_separation(csv, event_dict)
        l=len(list1)
        for event in list1:
            event = [event]
            out_name=f"{data['tif_files'][x]}"
            event_dict = image_crop(1,event, csv, im,0, out_name,filepath, SAVING_SCHEME=SAVING_SCHEME,
                    folder_dict=folder_dict, event_dict=event_dict)

            for i in range(2,7,2):
                sigma=(i,i)
                s=sigma[0]
                print('sigma:',s)
                in_name=f"{original_folder}_{x+1}_sigma{s}.tiff"

                gauss=os.path.join(base_dir, 'original_data',original_folder,in_name)
                gauss_image=Image.open(gauss)
                image_crop(1,event, csv, gauss_image,1, out_name,filepath, SAVING_SCHEME=SAVING_SCHEME,
                    folder_dict=folder_dict, event_dict=event_dict)



try:
    original_folder = sys.argv[1]
except (IndexError, KeyError) as e:
    print("No folder specified")


def main(my_folder = None):
    if my_folder:
        original_folder = my_folder
    files_dir = os.path.join(BASE_DIR, 'original_data')
    joined_path = os.path.join(files_dir, original_folder)
    print("GAUSSIAN FOLDER", joined_path)
    data = get_data_from_folder(original_folder)
    produce_gaussians(data, joined_path)
    crop_data(data, joined_path)

def do_all():
    files_dir = os.path.join(BASE_DIR, 'original_data')
    folders = Path(files_dir).glob("*")
    for folder in folders:
        main(folder)

if __name__ == "__main__":
    main("200101_mitogfp_cos7_janeliasim_fl")