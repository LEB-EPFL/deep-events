from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
import os
import tensorflow_probability as tfp
import tensorflow as tf
from tqdm import tqdm
import tifffile
from bson.objectid import ObjectId
from database.extract_yaml import save_dict
import shutil
from skimage import filters, segmentation, feature, measure, morphology
import scipy.ndimage as ndi

def event_separation(data):
    #this function takes in the data from the excel file and splits them into a nested list: each list within the nested list corresponds to the excel lines of a single division
    #it differentiates the lines based on conditions on both frame number and x,y-distance that could potentially be changed if they don't work
    length_of_file=len(data)-1
    all_event_lines=[]
    single_divison_events=[]
    for excelindex in range(0,length_of_file):
        framedataline1= data.iloc[excelindex,1]
        framedataline2= data.iloc[excelindex+1,1]
        framediff= abs(framedataline2-framedataline1)

        ydistancedataline1= data.iloc[excelindex,2]
        ydistancedataline2= data.iloc[excelindex+1,2]
        ydistancediff= abs(ydistancedataline2-ydistancedataline1)

        xdistancedataline1= data.iloc[excelindex,2]
        xdistancedataline2= data.iloc[excelindex+1,2]
        xdistancediff= abs(xdistancedataline2-xdistancedataline1)

        if framediff < 10 and ydistancediff <  15 and xdistancediff < 15 :
            single_divison_events.append(excelindex)
            if excelindex == length_of_file-1:
                single_divison_events.append(excelindex+1)
                all_event_lines.append(single_divison_events)
        else:
            single_divison_events.append(excelindex)
            all_event_lines.append(single_divison_events)
            single_divison_events=[]
    return all_event_lines


def image_crop(l,list_of_divisions, data, img, g_state, outputname, foldname, SAVING_SCHEME= "None",
                    folder_dict = None, event_dict = None):
    division_list=[]

    for index_list in range(0, l):
        l1=len(list_of_divisions[index_list])
        division_list=[]

        for index_list1 in range(0,l1):
            division_list.append(list_of_divisions[index_list][index_list1])
        minlist=division_list[0]
        maxlist=division_list[index_list1]

        data_croped= data.iloc[minlist:maxlist+1,1:4]
        frame1=int(data_croped['axis-0'].min())
        frame2=int(data_croped['axis-0'].max())
        ymean = data_croped['axis-1'].mean()
        xmean = data_croped['axis-2'].mean()
        ycrop1=ymean+128
        ycrop2=ymean-128
        xcrop1=xmean+128
        xcrop2=xmean-128

        if ycrop1 > 2048:                           #safety conditions in case pics are at the upper edges
            ycrop1=2048
            ycrop2=1792
        if xcrop1 > 2048:
            xcrop1=2048
            xcrop2=1792
        if ycrop2 < 0:                           #safety conditions in case pics are at the lower edges
            ycrop1=256
            ycrop2=0
        if xcrop2 < 0:
            xcrop1=256
            xcrop2=0

        dataar=np.zeros((frame2-frame1+1, 256, 256))

        for frame_index, frame_number in enumerate(range (frame1, frame2+1)):
            img.seek(frame_number) 
            box = (xcrop2, ycrop2, xcrop1, ycrop1) #choose dimensions of box
            imcrop= img.crop(box)

            dataar[frame_index, :, :] = np.array(imcrop)

        if "ws" in SAVING_SCHEME:
            # Adjust the names to the database optimized saving scheme
            if g_state == 0:
                path, folder_dict, event_id = get_save_info(foldname, folder_dict)
                event_dict['event_path'] = path
                event_dict['_id'] = event_id
                save_dict(event_dict)
                currname_crop = f"images.tiff"
                savepath = os.path.join(path, currname_crop)
                tifffile.imwrite(savepath, (dataar).astype(np.uint16), photometric='minisblack')
            elif g_state == 1:
                path = event_dict['event_path']
                currname_crop = f"gaussians.tiff"
                savepath = os.path.join(path, currname_crop)
                tifffile.imwrite(savepath, (dataar).astype(np.uint8), photometric='minisblack')
        else:
            if g_state==0:
                save_im(index_list, outputname, foldname, dataar)
            if g_state==1:
                save_gauss(index_list, outputname, foldname, dataar)

    if folder_dict:
        save_dict(folder_dict)
        return event_dict

def get_save_info(foldname, folder_dict):
    filepath = os.path.dirname(foldname)
    event_id = ObjectId()
    folder = os.path.basename(foldname)
    try:
        folder_dict['extracted_events'].append(event_id.binary.hex())
    except (KeyError, AttributeError) as e:
        folder_dict['extracted_events'] = [event_id.binary.hex()]
    filepath = os.path.join(filepath, folder, event_id.binary.hex())
    Path(filepath).mkdir(parents=True, exist_ok=True)
    return filepath, folder_dict, event_id


def delete_old_extracted_events(folder_dict, training_path):
    date = folder_dict['date']
    if not folder_dict['extracted_events']:
        return folder_dict
    for event in folder_dict['extracted_events']:
        folder = event
        path = os.path.join(training_path, folder)
        print(f"removing {path}.")
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            print("Not found in this folder.")
        except PermissionError:
            print("No permission, skip.")
    folder_dict['extracted_events'] = []
    save_dict(folder_dict)


def save_gauss(index_list, outputname, foldname, dataar_gauss):
    currname_crop_gauss = f'{outputname}_{index_list}gauss.tiff'
    savepath=os.path.join(foldname,currname_crop_gauss)
    tifffile.imwrite(savepath, (dataar_gauss).astype(np.uint8), photometric='minisblack')

def save_im(index_list, outputname, foldname, dataar):
    currname_crop = f'{outputname}_{index_list}.tiff'
    savepath= os.path.join(foldname,currname_crop)
    tifffile.imwrite(savepath, (dataar).astype(np.uint16), photometric='minisblack')

def get_gaussian(mu, sigma, size):
    mu = ((mu[1]+0.5-0.5*size[1])/(size[1]*0.5), (mu[0]+0.5-0.5*size[0])/(size[0]*0.5))
    sigma = (sigma[0]/size[0], sigma[1]/size[1])
    mvn = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
    x,y = tf.cast(tf.linspace(-1,1,size[0]),tf.float32), tf.cast(tf.linspace(-1,1,size[1]), tf.float32)
    # meshgrid as a list of [x,y] coordinates
    coords = tf.reshape(tf.stack(tf.meshgrid(x,y),axis=-1),(-1,2)).numpy()
    gauss = mvn.prob(coords)
    return tf.reshape(gauss, size)


def image_crop_negative(l,list_of_divisions, data, img, g_state, outputname, foldname):
    outnm = f'{outputname}_neg'
    division_list=[]
    for index_list in range(0, l):
        l1=len(list_of_divisions[index_list])
        division_list=[]

        for index_list1 in range(0,l1):
            division_list.append(list_of_divisions[index_list][index_list1])
        minlist=division_list[0]
        maxlist=division_list[index_list1]

        data_croped_after_event= data.iloc[minlist:maxlist+1,1:4]
        max_frame_data=int(data_croped_after_event['axis-0'].max())
        max_frame_img=img.n_frames
        if max_frame_data< max_frame_img-10:
            frame1_after=max_frame_data
            frame2_after=max_frame_data+10
            ymean_after = data_croped_after_event['axis-1'].mean()
            xmean_after = data_croped_after_event['axis-2'].mean()
            y_moved_lower=ymean_after+64
            y_moved_upper=ymean_after+320
            x_moved_lower=xmean_after+64
            x_moved_upper=xmean_after+320
            if y_moved_upper>2048 and x_moved_upper>2048:
                y_moved_lower=ymean_after-320
                y_moved_upper=ymean_after-64
                x_moved_lower=xmean_after-320
                x_moved_upper=xmean_after-64
            elif y_moved_upper<2048 and x_moved_upper>2048:
                y_moved_lower=ymean_after+64
                y_moved_upper=ymean_after+320
                x_moved_lower=xmean_after-320
                x_moved_upper=xmean_after-64
            elif y_moved_upper>2048 and x_moved_upper<2048:
                y_moved_lower=ymean_after-320
                y_moved_upper=ymean_after-64
                x_moved_lower=xmean_after+64
                x_moved_upper=xmean_after+320



            dataar_a=np.zeros((frame2_after-frame1_after+1, 256, 256))

            for frame_index_a, frame_number_a in enumerate(range (frame1_after, frame2_after+1)):
                img.seek(frame_number_a) 
                box_a = (x_moved_lower, y_moved_lower, x_moved_upper, y_moved_upper) #choose dimensions of box
                imcrop= img.crop(box_a)

                dataar_a[frame_index_a, :, :] = np.array(imcrop)
            if g_state==0:
                save_im(index_list, outnm, foldname, dataar_a)
            if g_state==1:
                save_gauss(index_list, outnm, foldname, dataar_a)


def augImg(input_img, output_img, transform):
    #input_mask = (input_img>0).astype(np.uint8)
    transformed = transform(image=input_img, image0=output_img)

    aug_input_img, aug_output_img= transformed['image'], transformed['image0']
    return aug_input_img.astype(np.float64), aug_output_img


def augStack(input_data, output_data, transform):
    aug_input_data = np.zeros(input_data.shape, dtype=np.float64)
    aug_output_data = np.zeros(output_data.shape, dtype=np.float32)

    for i in tqdm(range(input_data.shape[0]), total=input_data.shape[0]):
        aug_input_data[i], aug_output_data[i]= augImg(input_data[i], output_data[i], transform)
    return aug_input_data, aug_output_data


def augImg_one(input_img, transform, **kwargs):
    #input_mask = (input_img>0).astype(np.uint8)
    transformed = transform(image=input_img)
    aug_input_img= transformed['image']

    # aug_fission_coords = preprocessing.fissionCoords(aug_labels, aug_output_img)
    # aug_output_img, aug_fission_props = preprocessing.prepareProc(aug_output_img, coords=aug_fission_coords, **kwargs)
    return aug_input_img.astype(np.uint8)

def augStack_one(input_data, transform, **kwargs):
    aug_input_data = np.zeros(input_data.shape, dtype=np.uint8)

    for i in tqdm(range(input_data.shape[0]), total=input_data.shape[0]):
        aug_input_data[i] = augImg_one(input_data[i],  transform, **kwargs)
    return aug_input_data

def poi(datacsv,input_name, sigma_trial, size_trial,total_frames):
    points_of_interest= np.zeros((total_frames, 2048, 2048))

    for row_number in tqdm(range(0, len(datacsv))):
        framenumber_in_row= int(datacsv.loc[row_number, 'axis-0'])
        fission_ycoord = int(datacsv.loc[row_number, 'axis-1'])
        fission_xcoord = int(datacsv.loc[row_number, 'axis-2'])
        fission_coords = (fission_ycoord,fission_xcoord)
        gaussian_points = get_gaussian(fission_coords,sigma_trial,size_trial)                                     #gets gaussian points at a single frame
        gaussian_points = gaussian_points.numpy()                                                               #convers tensor into numpy array
        gaussian_points = gaussian_points/np.max(gaussian_points)                                               #divides by the max
        gaussian_points[gaussian_points < 0.1] = 0                                                              #sets background to zero
        gaussian_points = gaussian_points/np.max(gaussian_points)                                               #divides by max again
        points_of_interest[framenumber_in_row] = points_of_interest[framenumber_in_row] + gaussian_points       #adds the gaussian intensity in the empty file



    #TODO We could save this as float and save all of the work when we load it later
    tifffile.imwrite(input_name, (points_of_interest*254).astype(np.uint8))
    return

import cv2 as cv
def find_int(int_dir, thresh):
    coords=np.zeros(1,1)
    for image_file in int_dir:
        # change it with your absolute path for the image
        image = cv.imread(image_file)
        blur = cv.GaussianBlur(image, (5, 5),cv.BORDER_DEFAULT)
        ret, thresh = cv.threshold(blur, 0.5, 1, cv.THRESH_BINARY_INV)

        contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        for i in contours:
            M = cv.moments(i)
            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                #cv.drawContours(image, [i], -1, (0, 255, 0), 2)
                #cv.circle(image, (cx, cy), 7, (0, 0, 255), -1)
                #cv.putText(image, "center", (cx - 20, cy - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            coords.append(cy,cx)
    coords= np.delete(coords, 0, axis=0)