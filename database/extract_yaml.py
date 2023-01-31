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
MAIN_PATH = r'\\lebnas1.epfl.ch\microsc125\deep_events\original_data'
KEYS_PATH = os.path.dirname(os.path.realpath(__file__)) + "/keys.yaml"

keys = benedict(KEYS_PATH)


def get_dict(folder: str, path:str = MAIN_PATH):
    try:
        folder_dict = benedict(os.path.join(path, folder, "db.yaml"))
    except ValueError:
        folder_dict = benedict(benedict().
                               to_yaml(filepath=os.path.join(path, folder, "db.yaml")))
    folder_dict['original_path'] = '/'.join([path, folder])
    folder_dict['original_folder'] = folder
    return folder_dict


def save_dict(folder_dict: dict):
    if folder_dict['type'] == "original":
        folder_dict.to_yaml(filepath=os.path.join(folder_dict['original_path'],
                                              "db.yaml"))
    elif folder_dict['type'] == "event":
        folder_dict.to_yaml(filepath=os.path.join(folder_dict['event_path'],
                                              "db.yaml"))


def extract_foldername(folder: Path):
    try:
        date = re.search(r'(\d{2}[01]\d{3})', str(folder)).group(1)
    except AttributeError:
        date = None
    folder_dict = get_dict(folder)
    folder_dict['date'] = date
    folder_dict['type'] = 'original'
    folder_dict['extracted_events'] = []
    for key in keys.keys():
        matches = {x for x in keys[key] if x.lower() in str(folder).lower()}
        folder_dict[key] = matches
    save_dict(folder_dict)


def extract_ome(folder: str):
    folder_dict = get_dict(folder)
    tif_files = list(Path(os.path.join(MAIN_PATH, folder)).glob(r'*.ome.tif*'))
    if tif_files:
        with tifffile.TiffFile(tif_files[0]) as tif:
            fps = extract_fps(tif)
            folder_dict['fps'] = fps
            params = extract_params(tif)
            folder_dict['ome'] = params
    save_dict(folder_dict)


def extract_fps(tif: tifffile.TiffFile):

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
def extract_params(tif: tifffile.TiffFile):
    mdInfoDict = xmltodict.parse(tif.ome_metadata, force_list={'Plane'})
    pixels = mdInfoDict['OME']['Image']["Pixels"]
    params = translate_ome_dict(pixels)
    params["channel"] = translate_ome_dict(pixels["Channel"])
    params["instrument"] = translate_ome_dict(mdInfoDict['OME']['Instrument'], True)
    return params


def translate_ome_dict(ome_dict: dict, recursive: bool = False):
    params = {}
    for key, value in ome_dict.items():
        if "@" in key:
            try:
                value = float(value)
            except ValueError:
                pass
            params[convert_camel_case(key)] = value
        elif recursive:
            params[convert_camel_case(key)] = translate_ome_dict(value)
    return params

def convert_camel_case(camel_case: str):
    camel_case = camel_case.replace('@','')
    return re.sub(r'(?<!^)(?=[A-Z])', '_', camel_case).lower()


#%%
def extract_folders(path: Path):
    folders = Path(path).iterdir()
    for folder in folders:
        folder = os.path.basename(str(folder))
        # pbar.set_description(str(folder))
        extract_foldername(folder)
        extract_ome(folder)

extract_folders(MAIN_PATH)

# %%
