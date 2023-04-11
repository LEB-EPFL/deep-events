from pathlib import Path
import tifffile
from event_extraction import basic_scan, crop_images
import os
import shutil
import copy
from deep_events.database.folder_benedict import get_dict, save_dict
from bson.objectid import ObjectId
import numpy as np
from multiprocessing import Pool

folder = "Z:/_Lab members/Emily/20230329_FtsW_sfGFP_caulobacter_zeiss"
db_files = list(Path(folder).rglob(r'ground_truth*'))
SAVING_SCHEME = "ws_0.2"


def extract_events(db_file):
    tif_file = sorted(Path(os.path.dirname(db_file)).glob(r'*.ome.tif'), key=os.path.getmtime)[-1]
    if os.path.exists(os.path.join(os.path.dirname(db_file), 'ground_truth.tiff')):
        gaussians_file = os.path.join(os.path.dirname(db_file), 'ground_truth.tiff')
    else:
        gaussians_file = os.path.join(os.path.dirname(db_file), 'ground_truth.tif')

    folder_dict = get_dict(Path(os.path.dirname(db_file)))
    event_dict = copy.deepcopy(folder_dict)
    event_dict['type'] = "event"
    event_dict['original_file'] = os.path.basename(tif_file)
    event_dict['label_file'] = os.path.basename(gaussians_file)
    event_dict['event_content'] = 'division'

    with tifffile.TiffFile(tif_file) as images, tifffile.TiffFile(gaussians_file) as gaussians:
        events = basic_scan(gaussians.asarray(), threshold=0.5)
        print(events)
        for event in events:
            gaussians_crop, box = crop_images(event, gaussians)
            imgs_crop, box = crop_images(event, images)
            event_dict = handle_db(event, box, event_dict)

            # tifffile.imwrite(os.path.join(event_dict['event_path'], "ground_truth.tif"),
            #                  (gaussians_crop).astype(np.uint16), photometric='minisblack')
            tifffile.imwrite(os.path.join(event_dict['event_path'], "ground_truth.tif"),
                             (gaussians_crop).astype(np.float16), photometric='minisblack')
            # tifffile.imwrite(os.path.join(event_dict['event_path'], "images.tif"),
            #                  (imgs_crop).astype(np.uint16), photometric='minisblack')
            tifffile.imwrite(os.path.join(event_dict['event_path'], "images.tif"),
                             (imgs_crop).astype(np.float16), photometric='minisblack')


def handle_db(event, box, event_dict):
    event_id = ObjectId()
    event_folder = f"ev_{event_dict['cell_line'][0]}_{event_dict['microscope'][0]}_{event_dict['contrast'][:4]}_{event_id}"
    path = os.path.join(folder, "training_data", event_folder)
    Path(path).mkdir(parents=True, exist_ok=True)
    event_dict['event_path'] = path
    event_dict['_id'] = event_id
    event_dict['frames'] = (event.first_frame, event.last_frame+1)
    event_dict['crop_box'] = box
    event_dict['saving_scheme'] = SAVING_SCHEME
    save_dict(event_dict)
    return event_dict



if __name__ == "__main__":
    shutil.rmtree(os.path.join(folder, "training_data"), ignore_errors=True)
    with Pool(5) as p:
        p.map(extract_events, db_files)
