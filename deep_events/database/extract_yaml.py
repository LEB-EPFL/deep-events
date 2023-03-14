#%% imports
from pathlib import Path
from benedict import benedict
import re
import os
from deep_events.database.folder_benedict import get_dict, save_dict, set_dict_entry, handle_folder_dict

from typing import Union


#%% setup
MAIN_PATH = benedict(os.path.dirname(os.path.realpath(__file__)) + "/settings.yaml")['MAIN_PATH']
KEYS_PATH = os.path.dirname(os.path.realpath(__file__)) + "/keys.yaml"


keys = benedict(KEYS_PATH)


def recursive_folder(folder: Path):
    folder_dict = get_dict(folder)
    subpath = folder
    while subpath != os.path.dirname(subpath):
        set_manual(folder_dict, subpath)
        folder_dict = extract_foldername(folder_dict, subpath)
        subpath = os.path.dirname(subpath)


def check_minimal(folder: Path):
    folder_dict = get_dict(folder)
    present_keys = folder_dict.keys()
    min_keys = list(keys.keys())
    matched_keys = [key in present_keys for key in min_keys]
    return all(matched_keys)

#%%

def delete_db_files(folder: Path):
    file_list = list(Path(folder).rglob(r'db.yaml'))
    for file in file_list:
        os.remove(file)

@handle_folder_dict
def set_defaults(folder_dict: Union[dict, str, Path]):
    "Get default settings from file and set them to the local yaml"

    folder_dict['type'] = 'original'
    default_keys = benedict(KEYS_PATH)['defaults']

    for key, value in default_keys.items():
        folder_dict[key] = value
    return folder_dict


@handle_folder_dict
def set_manual(folder_dict: Union[dict, str, Path], folder):
    "Get manual entries from a yaml file and transfer them"
    try:
        manual_entries = benedict(os.path.join(folder, "db_manual.yaml"))
        for key, value in manual_entries.items():
            set_dict_entry(folder_dict, key, value)
    except (KeyError, ValueError) as e:
        # No manual entries for this folder
        pass
    return folder_dict


def extract_foldername(folder_dict: Union[dict, str, Path], folder):
    "Get a folder name and try to set info from it"
    folder = os.path.basename(folder).lower()

    # DATE
    if not 'date' in folder_dict.keys():
        try:
            date = re.search(r'(\d{2,4}[01]\d{3})', str(folder)).group(1)
            folder_dict['date'] = date
        except AttributeError:
            pass

    if not 'original_folder' in folder_dict.keys():
        folder_dict['original_folder'] = os.path.basename(folder)

    for key in keys.keys():
        if not key in folder_dict.keys():
            matches = {x for x in keys[key] if x.lower() in str(folder).lower()}
            if matches:
                folder_dict[key] = matches
    return folder_dict





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
