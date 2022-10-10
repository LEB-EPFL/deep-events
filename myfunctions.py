from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.io
import plotly.express as px
from scipy import ndimage as ndi
import tensorflow
import imageio
from skimage.segmentation import flood, flood_fill
from scipy import signal
import tensorflow_probability as tfp
import tensorflow as tf
from tqdm import tqdm


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

        if framediff < 10 and ydistancediff <  10 and xdistancediff < 10 :
            single_divison_events.append(excelindex)
            if excelindex == length_of_file-1:
                single_divison_events.append(excelindex+1)
                all_event_lines.append(single_divison_events)
        else:
            single_divison_events.append(excelindex)
            all_event_lines.append(single_divison_events)
            single_divison_events=[]
    return all_event_lines


def image_crop_save(l,list_of_divisions, data, img):   
    division_list=[]
    for index_list in range(0, l):
        l1=len(list_of_divisions[index_list])
        division_list=[]

        for index_list1 in range(0,l1):
            division_list.append(list_of_divisions[index_list][index_list1])
            minlist=division_list[0]
            maxlist=division_list[index_list1]

        data_croped= data.iloc[minlist:maxlist,1:4]
        frame1=int(data_croped['axis-0'].min())
        frame2=int(data_croped['axis-0'].max())
        ymean = data_croped['axis-1'].mean()
        xmean = data_croped['axis-2'].mean()
        ycrop1=ymean+128
        ycrop2=ymean-128
        xcrop1=xmean+128
        xcrop2=xmean-128
        dataar=np.zeros((frame2-frame1, 256, 256))

        for frame_index, frame_number in enumerate(range (frame1, frame2)):
            img.seek(frame_number) #starts from 0 I think?
            box = (xcrop2, ycrop2, xcrop1, ycrop1) #choose dimensions of box
            imcrop= img.crop(box)
        
            dataar[frame_index, :, :] = np.array(imcrop)
        currname_crop = f'image_{index_list}.tiff'
        imageio.mimwrite(currname_crop, dataar)
    return dataar    



def image_crop_save_gauss(l,list_of_divisions, data, img):   
    division_list=[]
    for index_list in range(0, l):
        l1=len(list_of_divisions[index_list])
        division_list=[]

        for index_list1 in range(0,l1):
            division_list.append(list_of_divisions[index_list][index_list1])
            minlist=division_list[0]
            maxlist=division_list[index_list1]

        data_croped= data.iloc[minlist:maxlist,1:4]
        frame1=int(data_croped['axis-0'].min())
        frame2=int(data_croped['axis-0'].max())
        ymean = data_croped['axis-1'].mean()
        xmean = data_croped['axis-2'].mean()
        ycrop1=ymean+128
        ycrop2=ymean-128
        xcrop1=xmean+128
        xcrop2=xmean-128
        dataar_gauss=np.zeros((frame2-frame1, 256, 256))

        for frame_index, frame_number in enumerate(range (frame1, frame2)):
            img.seek(frame_number) #starts from 0 I think?
            box = (xcrop2, ycrop2, xcrop1, ycrop1) #choose dimensions of box
            imcrop= img.crop(box)
        
            dataar_gauss[frame_index, :, :] = np.array(imcrop)
        currname_crop_gauss = f'image_{index_list}gauss.tiff'
        imageio.mimwrite(currname_crop_gauss, dataar_gauss)
    return dataar_gauss 