"""Translate csv files to gaussians"""

from pathlib import Path
import os
from deep_events.database.folder_benedict import get_dict
import tifffile
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
import pandas as pd
from deep_events.myfunctions import poi
import numpy as np
from deep_events.database.convenience import glob_zarr

folder = "//sb-nas1.rcp.epfl.ch/LEB/Scientific_projects/deep_events_WS/data/original_data/20231114_series_COS7_zeiss_brightfield"
# folder = "//sb-nas1.rcp.epfl.ch/LEB/Scientific_projects/deep_events_WS/data/phaseEDA/20240719_drp1emerald"
SIGMA = 5
csv_file_pattern = 'ld_mito'

def csv_to_gaussian(folder, SIGMA, csv_file_pattern=''):
    db_files = glob_zarr(Path(folder), r'db.yaml')

    # From all the folders with a db.yaml file get the csv file
    csv_files = []
    for db_file in db_files:
        print(db_file)
        csv = list(Path(os.path.dirname(db_file)).glob(csv_file_pattern + '*.csv'))
        if csv:
            csv_files.append(csv[0])

    csv_files = [str(x) for x in csv_files]

    print("\n".join(csv_files))

    for csv in csv_files:
        csv_name = csv
        print(csv_name)
        try:
            img_file = sorted(glob_zarr(Path(os.path.dirname(csv)), r'*.ome.tif'), key=os.path.getmtime)[-1]
            file_type = 'tif'
        except IndexError:
            img_file = Path(os.path.dirname(csv))
            file_type = 'zarr'

        meta = get_dict(os.path.dirname(img_file))
        def get_scale_value(meta, value):
            if value == "inverse":
                scale_value = 1/meta['ome']['physical_size_x']
            elif value is False:
                scale_value = 1
            elif value:
                scale_value = meta['ome'].get('physical_size_x', 1)
            else:
                scale_value = 1
            return scale_value
        
        if isinstance(meta.get('scale_csv', False), dict):
            for key, value in meta.get('scale_csv').items():
                if key in csv_file_pattern:
                    scale_value = get_scale_value(meta, value)
        else:
            scale_value = get_scale_value(meta, meta.get('scale_csv', False))


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
        in_name = os.path.join(os.path.dirname(img_file), f'ground_truth_{csv_file_pattern}.tiff')

        if file_type == 'tif':
            with tifffile.TiffFile(img_file) as tif:
                framenum = len(tif.pages)
                size = tif.pages[0].shape
            group = None    
        if file_type == 'zarr':
            reader = Reader(parse_url(img_file))
            node = list(reader())[0]
            framenum = node.data[0].shape[0]
            size = node.data[0].shape[-2:]
            group = parse_url(img_file, mode='a')



        poi(csv,in_name,sigma,size,framenum, size, None, group)


if __name__ == "__main__":
    csv_to_gaussian(folder, SIGMA, csv_file_pattern)