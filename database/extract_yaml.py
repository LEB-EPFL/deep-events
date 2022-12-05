#%% imports
from pathlib import Path
from benedict import benedict
import re
import os
from tqdm import tqdm

#%% setup
MAIN_PATH = "//lebnas1.epfl.ch/microsc125/deep_events/original_data"
KEYS_PATH = "./keys.yaml"


keys = benedict(KEYS_PATH)

def extract_yamls(path: Path):
    folders = Path(path).iterdir()
    for folder in (pbar := tqdm(folders)):
        folder = os.path.basename(str(folder))
        pbar.set_description(str(folder))
        extract_yaml(folder)


def extract_yaml(folder):
    try:
        date = re.search(r'(\d{2}[01]\d{3})', str(folder)).group(1)
    except AttributeError:
        date = None
    try:
        folder_dict = benedict(os.path.join(MAIN_PATH, folder, "db.yaml"))
    except ValueError:
        folder_dict = benedict(benedict().
                               to_yaml(filepath=os.path.join(MAIN_PATH, folder, "db.yaml")))
    folder_dict['date']= date
    for key in keys.keys():
        matches = {x for x in keys[key] if x.lower() in str(folder).lower()}
        folder_dict[key] = matches
    folder_dict.to_yaml(filepath=os.path.join(MAIN_PATH, folder, "db.yaml"))

# %%

extract_yamls(MAIN_PATH)
# %%
