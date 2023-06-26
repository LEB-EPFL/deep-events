from pathlib import Path
import tifffile
from deep_events.event_extraction import basic_scan, crop_images
import os
import shutil
import copy
from deep_events.database.folder_benedict import get_dict, save_dict
from benedict import benedict
from bson.objectid import ObjectId
import numpy as np
from multiprocessing import Pool


SAVING_SCHEME = "ws_0.2"

def extract_events(db_file, events_folder: str, images_identifier: str = "", channel_contrast: str = "",
                   label: str = ""):
    folder_dict = get_dict(Path(os.path.dirname(db_file)))


    if images_identifier != "":
        tif_identifier = r'*' + images_identifier + r'*.ome.tif'
    if isinstance(channel_contrast, list):
        for contrast in channel_contrast:
            print("Running recursively on the channel constrast list")
            extract_events(db_file, images_identifier, contrast, label, events_folder)
        return
    elif channel_contrast != "":
        tif_identifier =  r'*' + folder_dict['contrast'][channel_contrast] + r'*.ome.tif'
    else:
        tif_identifier = r'*.ome.tif'


    print(f"tif identifier: {tif_identifier}")
    print(channel_contrast)

    tif_files = sorted(Path(os.path.dirname(db_file)).glob(tif_identifier), key=os.path.getmtime)
    if tif_files:
        tif_file = tif_files[-1]
        tif_files.remove(tif_file)
        print(tif_file)
    else:
        print(db_file)
        print("Did not find a corresponding tif file for this db_file")
        return


    if os.path.exists(os.path.join(os.path.dirname(db_file), 'ground_truth.tiff')):
        gaussians_file = os.path.join(os.path.dirname(db_file), 'ground_truth.tiff')
    elif  os.path.exists(os.path.join(os.path.dirname(db_file), 'ground_truth.tif')):
        gaussians_file = os.path.join(os.path.dirname(db_file), 'ground_truth.tif')
    else:
        print(db_file)
        print("Did not find a corresponding ground truth file for this db_file")
        return
    print(gaussians_file)

    event_dict = copy.deepcopy(folder_dict)
    event_dict['type'] = "event"
    event_dict['channel_contrast'] = channel_contrast
    if channel_contrast != "":
        event_dict['contrast'] = channel_contrast
    event_dict['original_file'] = os.path.basename(tif_file)
    event_dict['label_file'] = os.path.basename(gaussians_file)
    event_dict['event_content'] = 'division'
    event_dict['extraction_type'] = 'automatic'

    with tifffile.TiffFile(tif_file) as images, tifffile.TiffFile(gaussians_file) as gaussians:
        # Open additional tifs to be able to read from them.
        tifs = []
        try:
            for tif in tif_files:
                tifs.append(open(tif, 'r'))

            events = basic_scan(gaussians.asarray(), threshold=0.5)
            if label != "":
                channel = folder_dict['labels'][label]
                print(f"Channel: {channel}")
            else:
                channel = 0
            print(events)
            for event in events:
                gaussians_crop, box = crop_images(event, gaussians)
                imgs_crop, box = crop_images(event, images, channel)
                event_dict = handle_db(event, box, event_dict, events_folder)

                # tifffile.imwrite(os.path.join(event_dict['event_path'], "ground_truth.tif"),
                #                  (gaussians_crop).astype(np.uint16), photometric='minisblack')
                tifffile.imwrite(os.path.join(event_dict['event_path'], "ground_truth.tif"),
                                (gaussians_crop).astype(np.float16), photometric='minisblack')
                tifffile.imwrite(os.path.join(event_dict['event_path'], "images.tif"),
                                (imgs_crop).astype(np.float16), photometric='minisblack')
                print(f" EVENT PATH: {event_dict['event_path']}")
        finally:
            for f in tifs:
                f.close()

def handle_db(event, box, event_dict, events_folder):
    event_id = ObjectId()
    try:
        event_folder = f"ev_{event_dict['cell_line'][0]}_{event_dict['microscope'][0]}_{event_dict['contrast'][:4]}_{event_id}"
    except TypeError:
        event_folder = f"ev_{event_dict['cell_line'][0]}_{event_dict['microscope']}_{event_id}"
    path = os.path.join(events_folder, "event_data", event_folder)
    print(path)
    Path(path).mkdir(parents=True, exist_ok=True)
    event_dict['event_path'] = path
    event_dict['_id'] = event_id
    event_dict['frames'] = (event.first_frame, event.last_frame+1)
    event_dict['crop_box'] = box
    event_dict['saving_scheme'] = SAVING_SCHEME
    save_dict(event_dict)
    return event_dict


def delete_automically_extracted_events(folder):
    db_files = list((folder).rglob(r'event_db.yaml'))
    for db_file in db_files:
        event_dict = benedict(db_file)
        if "extraction_type" in event_dict.keys() and event_dict['extraction_type'] == 'manual':
            if event_dict['extraction_type'] == 'manual':
                print(f"Not deleting {db_file}")
                continue
        shutil.rmtree(os.path.join(os.path.dirname(db_file)))


def reextract_events(folder, channel_contrast = "", label="", img_identifier = ""):
    "Delete old automatically extracted events and replace them with newly extracted ones"
    if (folder / "event_data").is_dir():
        delete_automically_extracted_events(folder / "event_data")
    db_files = list((folder).rglob(r'db.yaml'))
    print(db_files)
    # extract_events(db_files[30], folder/"event_data")
    with Pool(30) as p:
        p.starmap(extract_events, zip(db_files,
                                      [folder]*len(db_files),
                                      [img_identifier]*len(db_files),
                                      [channel_contrast]*len(db_files),
                                      [label]*len(db_files),))




def main(): #pragma: no cover

    # folder = Path("//lebsrv2.epfl.ch/LEB_SHARED/SHARED/_Lab members/Emily")
    # channel_contrast = ["brightfield", "fluorescence"]
    # label = ""

    folder = Path("//lebsrv2.epfl.ch/LEB_SHARED/SHARED/_Lab members/Juan/230511_PDA_TrainingSet_iSIM")
    channel_contrast = ""
    label = "mitochondria"

    if (folder / "event_data").is_dir():
        delete_automically_extracted_events(folder / "event_data")
    db_files = list((folder).rglob(r'db.yaml'))
    # for db_file in db_files:
    #     extract_events(db_file, "", ["brightfield", "fluorescence"])
    img_identifier = ""

    # extract_events(db_files[0], img_identifier, channel_contrast, label, folder)

    with Pool(30) as p:
        p.starmap(extract_events, zip(db_files,
                                      [img_identifier]*len(db_files),
                                      [channel_contrast]*len(db_files),
                                      [label]*len(db_files),
                                      [folder]*len(db_files)))

if __name__ == "__main__":
    main()
