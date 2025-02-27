from pathlib import Path
import tifffile
from deep_events.event_extraction import basic_scan, crop_images_tif, crop_images_array
import os
import shutil
import copy
from deep_events.database.folder_benedict import get_dict, save_dict
from benedict import benedict
from bson.objectid import ObjectId
import numpy as np
from multiprocessing import Pool
from deep_events.database.convenience import glob_zarr
from ome_zarr.reader import Reader
from ome_zarr.io import parse_url

SAVING_SCHEME = "ws_0.3"

def extract_events(db_file, settings: dict):
    folder = settings['folder']
    events_folder = settings['event_folder']
    images_identifier = settings.get('img_identifier', "")
    image_type = settings.get('img_type', r'*.ome.tif*')
    gt_name = settings.get('gt_identifier', 'ground_truth')
    channel_contrast = settings.get('channel_contrast', "")
    label = settings.get('label', "") 
    event_content = settings.get('event_content', "")
    add_post_frames = settings.get('add_post_frames' ,1)
    auto_negatives = settings.get('auto_negatives', False)

    folder_dict = get_dict(Path(os.path.dirname(db_file)))
    if auto_negatives:
        folder_dict['auto_negatives'] = auto_negatives

    if images_identifier != "":
        tif_identifier = r'*' + images_identifier
    else:
        tif_identifier = images_identifier
    if isinstance(channel_contrast, list):
        for contrast in channel_contrast:
            extract_events(db_file, folder, events_folder, images_identifier, contrast, label, event_content, add_post_frames)
        return
    elif channel_contrast != "":
        tif_identifier =  r'*' + folder_dict['contrast'][channel_contrast] + image_type
    else:
        tif_identifier = tif_identifier + image_type

    if isinstance(label, list):
        for this_label in label:
            extract_events(db_file, folder, events_folder, images_identifier, channel_contrast, this_label, event_content, add_post_frames)
        return


    print(f"tif identifier: {tif_identifier}")
    print(channel_contrast)

    tif_files = sorted(glob_zarr(Path(os.path.dirname(db_file)), tif_identifier, return_type=Path), key=os.path.getmtime)
    if 'zarr' in image_type and 'zarr' in str(db_file):
        tif_file = Path(db_file).parents[0]
        print(db_file)
    elif tif_files:
        tif_file = tif_files[-1]
        tif_files.remove(tif_file)
        print(tif_file)
    else:
        print(db_file)
        print("Did not find corresponding images for this db_file")
        return


    if os.path.exists(os.path.join(os.path.dirname(db_file), gt_name + '.tiff')):
        gaussians_file = os.path.join(os.path.dirname(db_file), gt_name + '.tiff')
    elif  os.path.exists(os.path.join(os.path.dirname(db_file), gt_name + '.tif')):
        gaussians_file = os.path.join(os.path.dirname(db_file), gt_name + '.tif')
    elif 'zarr' in image_type:
        if not (Path(os.path.dirname(db_file)) / gt_name).is_dir():
            return
        gaussians_file = gt_name
    else:
        print(db_file)
        print("Did not find a corresponding ground truth file for this db_file")
        return

    event_dict = copy.deepcopy(folder_dict)
    event_dict['type'] = "event"
    event_dict['channel_contrast'] = channel_contrast
    if channel_contrast != "":
        event_dict['contrast'] = channel_contrast
    event_dict['label'] = label
    event_dict['original_file'] = os.path.basename(tif_file)
    event_dict['label_file'] = os.path.basename(gaussians_file)
    event_dict['event_content'] = event_content
    event_dict['extraction_type'] = 'automatic'

    if 'zarr' in image_type:
        crop_zarrs(tif_file, gaussians_file, add_post_frames, tif_files, label, folder_dict, folder, events_folder, event_dict)
    else:
        crop_tifs(tif_file, gaussians_file, add_post_frames, tif_files, label, folder_dict, folder, events_folder, event_dict)

def crop_zarrs(zarr_loc, gaussians_file, add_post_frames, tif_files, label, folder_dict, folder, events_folder, event_dict):
    reader = Reader(parse_url(zarr_loc))
    images = list(reader())[0].data[0]
    print('Cropping', zarr_loc, '\n', Path(zarr_loc)/gaussians_file, '\n', images.shape)
    reader = Reader(parse_url(Path(zarr_loc)/gaussians_file))
    gaussians = list(reader())[0].data[0].copy()
    del reader
    events = basic_scan(gaussians, threshold=0.5, additional_post_frames=add_post_frames)
    if label != "":
        channel = folder_dict['labels'][label]
        print(f"Channel: {channel}")
    else:
        channel = 0
    print(events)
    for event in events:
        gaussians_crop, box = crop_images_array(event, gaussians)
        imgs_crop, box = crop_images_array(event, images, channel)
        event_dict = handle_db(event, box, event_dict, folder, events_folder)

        # tifffile.imwrite(os.path.join(event_dict['event_path'], "ground_truth.tif"),
        #                  (gaussians_crop).astype(np.uint16), photometric='minisblack')
        tifffile.imwrite(os.path.join(event_dict['event_path'], "ground_truth.tif"),
                        (gaussians_crop).astype(np.float16), photometric='minisblack')
        tifffile.imwrite(os.path.join(event_dict['event_path'], "images.tif"),
                        (imgs_crop).astype(np.float16), photometric='minisblack')
        print(f" EVENT PATH: {event_dict['event_path']}")
        if event_dict.get('auto_negatives', False):
            print("Doing auto negatives", event_dict.get('auto_negatives', 0))
            pos = [(0,1), (1,0), (0, -1), (-1, 0)]  # up, right, down, left
            offset = event_dict['crop_box'][0] - event_dict['crop_box'][2] 
            event_dict['event_path'] = event_dict['event_path'] + '_neg0'
            for i in range(event_dict['auto_negatives']):
                event.c_p['x'] = event.c_p['x'] + offset*pos[i][0]
                event.c_p['y'] = event.c_p['y'] + offset*pos[i][1]
                event_dict['event_path'] = event_dict['event_path'][:-1] + str(i)
                event_dict['_id'] = ObjectId()
                event_dict['event_content'] = 'none'

                gaussians_crop, box = crop_images_array(event, gaussians)
                imgs_crop, box = crop_images_array(event, images, channel)
                event_dict['crop_box'] = box
                save_dict(event_dict)
                tifffile.imwrite(os.path.join(event_dict['event_path'], "ground_truth.tif"),
                                (gaussians_crop).astype(np.float16), photometric='minisblack')
                tifffile.imwrite(os.path.join(event_dict['event_path'], "images.tif"),
                                (imgs_crop).astype(np.float16), photometric='minisblack')
                print(f" EVENT PATH: {event_dict['event_path']}")
            
 

def crop_tifs(tif_file, gaussians_file, add_post_frames, tif_files, label, folder_dict, folder, events_folder, event_dict):
  
    with tifffile.TiffFile(tif_file) as images, tifffile.TiffFile(gaussians_file) as gaussians:
        # Open additional tifs to be able to read from them.
        tifs = []
        try:
            for tif in tif_files:
                tifs.append(open(tif, 'r'))

            events = basic_scan(gaussians.asarray(), threshold=0.5, additional_post_frames=add_post_frames)
            if label != "":
                channel = folder_dict['labels'][label]
                print(f"Channel: {channel}")
            else:
                channel = 0
            print(events)
            for event in events:
                gaussians_crop, box = crop_images_tif(event, gaussians)
                imgs_crop, box = crop_images_tif(event, images, channel)
                event_dict = handle_db(event, box, event_dict, folder, events_folder)

                tifffile.imwrite(os.path.join(event_dict['event_path'], "ground_truth.tif"),
                                (gaussians_crop).astype(np.float16), photometric='minisblack')
                tifffile.imwrite(os.path.join(event_dict['event_path'], "images.tif"),
                                (imgs_crop).astype(np.float16), photometric='minisblack')
                print(f" EVENT PATH: {event_dict['event_path']}")
        
                if event_dict.get('auto_negatives', False):
                    print("Doing auto negatives")
                    pos = [(0,1), (1,0), (0, -1), (-1, 0)]  # up, right, down, left
                    offset = event_dict['crop_box'][0] - event_dict['crop_box'][2] 
                    event_dict['event_path'] = event_dict['event_path'] + '_neg0' 
                    for i in range(event_dict['auto_negatives']):
                        event.c_p['x'] = event.c_p['x'] + offset*pos[i][0]
                        event.c_p['y'] = event.c_p['y'] + offset*pos[i][1]

                        event_dict['event_path'] = event_dict['event_path'][:-1] + str(i)
                        event_dict['_id'] = ObjectId()
                        event_dict['event_content'] = 'none'

                        gaussians_crop, box = crop_images_tif(event, gaussians)
                        imgs_crop, box = crop_images_tif(event, images, channel)
                        event_dict['crop_box'] = box
                        save_dict(event_dict)
                        tifffile.imwrite(os.path.join(event_dict['event_path'], "ground_truth.tif"),
                                        (gaussians_crop).astype(np.float16), photometric='minisblack')
                        tifffile.imwrite(os.path.join(event_dict['event_path'], "images.tif"),
                                        (imgs_crop).astype(np.float16), photometric='minisblack')
                        print(f" EVENT PATH: {event_dict['event_path']}")
        finally:
            for f in tifs:
                f.close()




def handle_db(event, box, event_dict, folder, event_subfolder="event_data"):
    event_id = ObjectId()
    try:
        event_folder = f"ev_{event_dict['cell_line'][0]}_{event_dict['microscope'][0]}_{event_dict['contrast'][:4]}_{event_id}"
    except (TypeError, KeyError) as e:
        event_folder = f"ev_{event_dict.get('cell_line', ['?cells'])[0]}_{event_dict.get('microscope', '?mic')}_{event_id}"
    path = os.path.join(folder, event_subfolder, event_folder)
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
    db_files = glob_zarr(Path(folder), r'event_db.yaml', return_type=Path)
    for db_file in db_files:
        event_dict = benedict(db_file)
        if "extraction_type" in event_dict.keys() and event_dict['extraction_type'] == 'manual':
            if event_dict['extraction_type'] == 'manual':
                # print(f"Not deleting {db_file}")
                continue
        shutil.rmtree(os.path.join(os.path.dirname(db_file)))

def reextract_events(folder, event_folder = "event_data", channel_contrast = "", label="", img_identifier = ""):
    "Delete old automatically extracted events and replace them with newly extracted ones"
    if (folder / event_folder).is_dir():
        delete_automically_extracted_events(folder / event_folder)
    db_files = glob_zarr(Path(folder), r'db.yaml', return_type=Path)
    print(db_files)
    with Pool(30) as p:
        p.starmap(extract_events, zip(db_files,
                                      [folder]*len(db_files),
                                      [event_folder]*len(db_files),
                                      [img_identifier]*len(db_files),
                                      [channel_contrast]*len(db_files),
                                      [label]*len(db_files),))

def main(FOLDERS, event_folders, img_types, settings): #pragma: no cover
    for FOLDER, event_folder in zip(FOLDERS, event_folders):
        if (FOLDER / event_folder).is_dir():
            delete_automically_extracted_events(FOLDER / event_folder)
    for FOLDER, event_folder, img_type in zip(FOLDERS, event_folders, img_types):
        db_files = glob_zarr(FOLDER, settings['db_name'], return_type=Path)
        print(db_files[0])
        
        settings['folder'] = FOLDER
        settings['event_folder'] = event_folder
        settings['img_type'] = img_type
        
        for db_file in db_files:
            print(db_file)
            extract_events(db_file, settings)
        # with Pool(10) as p:
        #     p.starmap(extract_events, zip(db_files, [settings]*len(db_files)))


if __name__ == "__main__":
    FOLDER = Path('//sb-nas1.rcp.epfl.ch/LEB/Scientific_projects/deep_events_WS/data/original_data/20231114_series_COS7_zeiss_brightfield')

    event_folder = "event_data_fluo" 
    img_type = '*.ome.tif*'
    settings = {
    'img_identifier' : "",
    'gt_identifier' : "ground_truth_ld_mito",  # "ground_truth_ld_mito" 'ground_truth_pearls'
    'db_name' : "db.yaml",
    'channel_contrast' : "",
    'label' : "",
    'add_post_frames' : 0,
    }

    settings['folder'] = FOLDER
    settings['event_folder'] = event_folder
    settings['img_type'] = img_type
    
    db_files = glob_zarr(FOLDER, settings['db_name'], return_type=Path)

    for db_file in db_files:
        print(db_file)
        extract_events(db_file, settings)