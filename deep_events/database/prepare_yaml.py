#%%
from pathlib import Path
import os

import extract_yaml



folder = "Z:/_Lab members/Juan/230222_MitoSplitNet_TrainingSet_U2OS_iSIM/"

#%%
extract_yaml.delete_db_files(folder)


#%% Get all of the tif files from that folder that we might be interested in
tif_files = list(Path(folder).rglob(r'*.ome.tif*'))


for file in tif_files:
    print(file)
    extract_yaml.set_defaults(os.path.dirname(file))
