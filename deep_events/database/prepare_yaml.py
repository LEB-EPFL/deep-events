#%%
from pathlib import Path
import os
from warnings import warn

import extract_yaml



folder = "Z:/_Lab members/Juan/230222_MitoSplitNet_TrainingSet_U2OS_iSIM/"

#%%
extract_yaml.delete_db_files(folder)


#%% Get all of the tif files from that folder that we might be interested in
tif_files = list(Path(folder).rglob(r'*.ome.tif*'))


for file in tif_files:
    extract_yaml.recursive_folder(os.path.dirname(file))
    extract_yaml.set_defaults(os.path.dirname(file))
    minimal_present = extract_yaml.check_minimal(os.path.dirname(file))
    if not minimal_present:
        warn(f"Folder {os.path.dirname(file)} does not have all necessary entries.")


# %%
