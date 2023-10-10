#%%
from pathlib import Path
import os
from warnings import warn

import deep_events.database.extract_yaml as extract_yaml
import deep_events.database.ome_to_yaml as ome_to_yaml


# folder = "//lebsrv2.epfl.ch/LEB_SHARED/SHARED/_Lab members/Juan/230511_PDA_TrainingSet_iSIM"
# folder = "Z:/_Lab members/Emily/"
# folder = Path("//lebnas1.epfl.ch/microsc125/deep_events/data/original_data/20230619_cos7_bf_zeiss")
folder = Path("//lebnas1.epfl.ch/microsc125/deep_events/data/original_data/20231010_cos7_bf_zeiss")


def prepare_all_folder(folder: str):
    #%%
    extract_yaml.delete_db_files(folder)

    #%% Get all of the tif files from that folder that we might be interested in
    tif_files = list(Path(folder).rglob(r'*.ome.tif*'))

    # Do the yaml file from the information in files and folder names
    for file in tif_files:
        extract_yaml.recursive_folder(file)
        extract_yaml.set_defaults(os.path.dirname(file))
        minimal_present = extract_yaml.check_minimal(os.path.dirname(file))
        if not minimal_present:
            warn(f"Folder {os.path.dirname(file)} does not have all necessary entries.")


    # Get the tif files that we want to look at for the OME metadata
    # The oldest one should have the most true OME data
    db_files = list(Path(folder).rglob(r'db.yaml'))
    tif_files = []
    print(f"Found {len(db_files)} db files")
    for db_file in db_files:
        print(db_file)
        tifs = sorted(Path(os.path.dirname(db_file)).glob(r'*.ome.tif'), key=os.path.getmtime)
        tif_files.append(tifs[0])
    tif_files = [str(file) for file in tif_files]
    print("tif_files", tif_files)

    # Get the OME fields from the tif_files

    for tif in tif_files:
        print(str(tif))
        ome_to_yaml.extract_ome(tif)

# %%
if __name__ == "__main__":
    prepare_all_folder(folder)
