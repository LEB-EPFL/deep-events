#%% imports
from pathlib import Path
from benedict import benedict
import re
import os
from tqdm import tqdm
import tifffile
import xmltodict
import numpy as np

#%% setup
MAIN_PATH = r'C:\Users\roumba\Documents\Software\deep-events\original_data'
KEYS_PATH = "./keys.yaml"

keys = benedict(KEYS_PATH)


def get_dict(folder: str):
    try:
        folder_dict = benedict(os.path.join(MAIN_PATH, folder, "db.yaml"))
    except ValueError:
        folder_dict = benedict(benedict().
                               to_yaml(filepath=os.path.join(MAIN_PATH, folder, "db.yaml")))
    folder_dict['original_folder'] = folder
    return folder_dict


def save_dict(folder_dict: dict):
    folder_dict.to_yaml(filepath=os.path.join(MAIN_PATH,
                                              folder_dict['original_folder'],
                                              "db.yaml"))


def extract_folders(path: Path):
    folders = Path(path).iterdir()
    for folder in folders:
        folder = os.path.basename(str(folder))
        # pbar.set_description(str(folder))
        extract_foldername(folder)
        extract_ome(folder)



def extract_foldername(folder: Path):
    try:
        date = re.search(r'(\d{2}[01]\d{3})', str(folder)).group(1)
    except AttributeError:
        date = None
    folder_dict = get_dict(folder)
    folder_dict['date'] = date
    for key in keys.keys():
        matches = {x for x in keys[key] if x.lower() in str(folder).lower()}
        folder_dict[key] = matches
    save_dict(folder_dict)


#%%

def extract_ome(folder: str):
    folder_dict = get_dict(folder)
    tif_files = list(Path(os.path.join(MAIN_PATH, folder)).glob(r'*.ome.tif*'))
    if tif_files:
        fps = extract_fps(tif_files[0])
        folder_dict['fps'] = fps
    save_dict(folder_dict)

def extract_fps(tif_file: Path):
    with tifffile.TiffFile(tif_file) as tif:
        mdInfoDict = xmltodict.parse(tif.ome_metadata, force_list={'Plane'})
        elapsed = []
        for plane in mdInfoDict['OME']['Image']['Pixels']['Plane']:
            elapsed.append(float(plane['@DeltaT']))
        fpu = np.mean(np.diff(elapsed))

        #TODO: Can this also be 'min'?
        try:
            multiplier = 1 if plane['@DeltaTUnit'] == 's' else 1000
        except KeyError:
            multiplier = 1
        # rounds the fps to 100s of ms
        fps = round(fpu/multiplier*10)/10
    return fps


#%%
# extract_ome('220915_mtstaygold_cos7_ZEISS_bf')
extract_folders(MAIN_PATH)
# %%
