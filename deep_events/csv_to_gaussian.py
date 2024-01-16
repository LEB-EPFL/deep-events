"""Translate csv files to gaussians"""

from pathlib import Path
import os
from deep_events.database.folder_benedict import get_dict
import tifffile
import pandas as pd
from deep_events.myfunctions import poi
import numpy as np

folder = "//lebnas1.epfl.ch/microsc125/deep_events/data/original_data/20231010_cos7_bf_zeiss/"
SIGMA = 5

def csv_to_gaussian(folder, SIGMA):
    db_files = list(Path(folder).rglob(r'db.yaml'))

    # From all the folders with a db.yaml file get the csv file
    csv_files = []
    for db_file in db_files:
        csv = list(Path(os.path.dirname(db_file)).glob(r'*.csv'))
        if csv:
            csv_files.append(csv[0])

    csv_files = [str(x) for x in csv_files]

    print("\n".join(csv_files))

    for csv in csv_files:
        csv_name = csv
        print(csv_name)
        img_file = sorted(Path(os.path.dirname(csv)).glob(r'*.ome.tif'), key=os.path.getmtime)[-1]
        meta = get_dict(os.path.dirname(img_file))
        if meta['scale_csv'] == "inverse":
            scale_value = 1/meta['ome']['physical_size_x']
        elif meta['scale_csv']:
            scale_value = meta['ome'].get('physical_size_x', 1)
        else:
            scale_value = 1

        csv = pd.read_csv(csv)
        print('scaling by', scale_value)
        csv['axis-1']=csv['axis-1']*(1/scale_value)
        csv['axis-2']=csv['axis-2']*(1/scale_value)
        mean_pos = (np.mean(csv['axis-1']) + np.mean(csv['axis-2']))/2
        if mean_pos < 205:
            print("\033[91m", mean_pos, "\033[00m <- should be >0 and <image_size (2048)")
        elif mean_pos > 2048:
            print("\033[91m", mean_pos, "\033[00m <- should be >0 and <image_size (2048)")
        else:
            print(mean_pos, " <- should be >0 and <image_size (2048)")
        print("If not, check scale_csv value in db.yaml file.")

        sigma = (SIGMA, SIGMA)
        in_name = os.path.join(os.path.dirname(img_file), 'ground_truth.tiff')

        with tifffile.TiffFile(img_file) as tif:
            framenum = len(tif.pages)
            size = tif.pages[0].shape

        poi(csv,in_name,sigma,size,framenum, size)


if __name__ == "__main__":
    csv_to_gaussian(folder, SIGMA)