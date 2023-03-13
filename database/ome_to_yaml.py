from pathlib import Path
import os
import re

import numpy as np
import tifffile
from database.folder_benedict import get_dict, save_dict
import xmltodict


def extract_ome(folder: str):
    "Get all the first tif file in a folder and get OME metadata from it, transfer to yaml"
    folder_dict = get_dict(folder)
    tif_files = list(Path(os.path.join(folder)).glob(r'*.ome.tif*'))
    if tif_files:
        with tifffile.TiffFile(tif_files[0]) as tif:
            fps = extract_fps(tif)
            folder_dict['fps'] = fps
            params = extract_params(tif)
            folder_dict['ome'] = params
    save_dict(folder_dict)


def get_ome(tif: tifffile.TiffFile):
    "Get the OME metadata itself from a Tif file"
    try:
        return xmltodict.parse(tif.ome_metadata, force_list={'Plane'})
    except TypeError:
        print("WARNING: OME metadata not valid, set fps and params manually")
        print(tif.filename)
        return False


def extract_fps(tif: tifffile.TiffFile):
    "From a tif file, calculate the fps by looking at the times that the frames were taken at."
    mdInfoDict = get_ome(tif)
    if not mdInfoDict:
        return 0
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
    """Extract specific parameters from the OME metadata"""
    mdInfoDict = get_ome(tif)
    if not mdInfoDict:
        return {}
    pixels = mdInfoDict['OME']['Image']["Pixels"]
    params = translate_ome_dict(pixels)
    params["channel"] = translate_ome_dict(pixels["Channel"])
    params["instrument"] = translate_ome_dict(mdInfoDict['OME']['Instrument'], True)
    return params

def translate_ome_dict(ome_dict: dict, recursive: bool = False):
    "Take the OME metadata and translate into a dictionary"
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