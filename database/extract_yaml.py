#%% imports
from pathlib import Path
from benedict import benedict
import re
import os
from tqdm import tqdm
import tifffile
import xmltodict
import numpy as np
from database.folder_benedict import get_dict, save_dict, set_dict_entry




#%% setup
MAIN_PATH = benedict(os.path.dirname(os.path.realpath(__file__)) + "/settings.yaml")['MAIN_PATH']
KEYS_PATH = os.path.dirname(os.path.realpath(__file__)) + "/keys.yaml"


keys = benedict(KEYS_PATH)


def set_defaults(folder:Path):
    "Get default settings from file and set them to the local yaml"
    folder_dict = get_dict(folder)
    folder_dict['type'] = 'original'
    default_keys = benedict(KEYS_PATH)['defaults']

    for key, value in default_keys.items():
        folder_dict[key] = value
    save_dict(folder_dict)


def set_manual(folder):
    "Get manual entries from a yaml file and transfer them"
    folder_dict = get_dict(folder)
    try:
        manual_entries = benedict(os.path.join(folder, "db_manual.yaml"))
        entries = manual_entries[folder_dict['original_folder']]
        for key, value in entries.items():
            set_dict_entry(folder_dict, key, value)
    except (KeyError, ValueError) as e:
        # No manual entries for this folder
        pass
    save_dict(folder_dict)


def extract_foldername(folder: Path):
    "Get a folder name and try to set info from it"
    try:
        date = re.search(r'(\d{2,4}[01]\d{3})', str(folder)).group(1)
    except AttributeError:
        date = None
    folder_dict = get_dict(folder)
    folder_dict['date'] = date
    folder_dict['original_folder'] = os.path.basename(folder)
    print(folder_dict['original_folder'])
    for key in keys.keys():
        matches = {x for x in keys[key] if x.lower() in str(folder).lower()}
        folder_dict[key] = matches
    save_dict(folder_dict)





#%%
def extract_folders(path: Path):
    folders = Path(path).iterdir()
    for folder in folders:
        folder = os.path.basename(str(folder))
        # pbar.set_description(str(folder))
        extract_foldername(folder)

        set_defaults(folder)

if __name__ == "__main__":
    # extract_folders(MAIN_PATH)

    folder = "//lebsrv2.epfl.ch/LEB_SHARED/_Lab members/Juan/230222_MitoSplitNet_TrainingSet_U2OS_iSIM"
    extract_folders(folder)
