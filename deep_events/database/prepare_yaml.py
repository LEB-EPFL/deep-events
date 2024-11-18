#%%
from pathlib import Path
import os
from warnings import warn

import deep_events.database.extract_yaml as extract_yaml
import deep_events.database.ome_to_yaml as ome_to_yaml
from deep_events.database.convenience import glob_zarr

# folder = "//lebsrv2.epfl.ch/LEB_SHARED/SHARED/_Lab members/Juan/230511_PDA_TrainingSet_iSIM"
# folder = "Z:/_Lab members/Emily/"
folder = Path("//sb-nas1.rcp.epfl.ch/LEB/Scientific_projects/deep_events_WS/data/original_data/20230628_cos7_bf_zeiss")
folder = Path("//lebnas1.epfl.ch/microsc125/deep_events/experiments/exploration/20240228_phaseEDA")
folder = Path("//sb-nas1.rcp.epfl.ch/LEB/Scientific_projects/deep_events_WS/data/phaseEDA/20240719_drp1emerald")

    #%%
def prepare_all_folder(folder: str):
    extract_yaml.delete_db_files(folder)

    #%% Get all of the tif files from that folder that we might be interested in
    tif_files = glob_zarr(folder, "*.ome.tif*")
    print("tif globs done")
    zarr_folders = glob_zarr(folder, "*.ome.zarr")
    zarr_folders = [x / "dummy.ome.zarr" for x in zarr_folders]
    all_files = tif_files + zarr_folders

    print("globs done")

    # Do the yaml file from the information in files and folder names
    for file in all_files:
        extract_yaml.recursive_folder(file)
        extract_yaml.set_defaults(os.path.dirname(file))
        minimal_present = extract_yaml.check_minimal(os.path.dirname(file))
        if not minimal_present:
            warn(f"Folder {os.path.dirname(file)} does not have all necessary entries.")

    print("extraction done")

    # Get the tif files that we want to look at for the OME metadata
    # The oldest one should have the most true OME data
    db_files = glob_zarr(folder, 'db.yaml')
    ome_sources = []
    print(f"Found {len(db_files)} db files")
    for db_file in db_files:
        tifs = sorted(Path(os.path.dirname(db_file)).glob(r'*.ome.tif*'), key=os.path.getmtime)
        try:
            ome_sources.append(tifs[0])
        except IndexError:
            # append zarr folder here
            pass
    ome_sources = [str(file) for file in ome_sources]
    print("ome_sources", ome_sources)

    # Get the OME fields from the tif_files

    for tif in ome_sources:
        print(str(tif))
        ome_to_yaml.extract_ome(tif)

# %%
if __name__ == "__main__":
    prepare_all_folder(folder)
