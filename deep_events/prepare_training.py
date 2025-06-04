from pathlib import Path
import numpy as np
import datetime
import random
from typing import List
import psutil
import time
from scipy.ndimage import gaussian_filter
import csv
from tqdm import tqdm

import matplotlib.pyplot as plt

# from sklearn.model_selection import train_test_split
import tifffile

from deep_events.database.construct import reconstruct_from_folder
from benedict import benedict
from deep_events.database import get_collection

folder = Path("X:\Scientific_projects\deep_events_WS\data\original_data\event_data")
MAX_N_TIMEPOINTS = 9

seed=42
np.random.seed(seed)

def main():
    prompt_file = 'X:/Scientific_projects/deep_events_WS/data/original_data/training_data/data_20250105_1122_brightfield_cos7_t0.2_f1_sFalse_mito_events_n753_sFalse/db_prompt.yaml'
    prompt = benedict(prompt_file)
    print(prompt['collection'])
    reconstruct_from_folder(Path('X:/Scientific_projects/deep_events_WS/data/original_data/event_data'), prompt["collection"])
    prompt['fps'] = 1
    del prompt['actual_val_split']
    prep_folder = prepare_for_prompt(folder, prompt, 'mito_events')
    print(prep_folder)

def prepare_for_prompt(folder: Path, prompt: dict, collection: str, test_size = 0.2,
                        fps = 1, smooth=False):
    
    # This deletes the keys that are not relevant for the db prompt
    config = {
        'n_event': prompt.pop('n_event', None),
        'fps': prompt.pop('fps', None),
        'smooth': prompt.pop('smooth', False),
        'train_val_split': prompt.pop('train_val_split', 0.2),
        'subset': prompt.pop('subset', False),
        'max_n_timepoints': prompt.pop('max_n_timepoints', MAX_N_TIMEPOINTS)
    }
    for key in ['n_timepoints', 'collection', 'backend']:
        prompt.pop(key, None)
    subset = False

    coll = get_collection(collection)
    print(prompt, collection)
    print('total_events', len(list(coll.find({}))))
    print('first event', list(coll.find({}))[0])
    filtered_list = list(coll.find(prompt))
    db_files = []
    print(len(filtered_list))

    for item in filtered_list:
        db_files.append(Path(item['event_path']) / "event_db.yaml")

    # Here we repopulate the prompt with the config values
    subset_ratio = config['subset'] or 1
    n_events_to_use = int(len(db_files) * subset_ratio)
    prompt.update({
        'train_val_split': config['train_val_split'],
        'fps': config['fps'],
        'smooth': config['smooth'],
        'collection': collection,
        'n_event': n_events_to_use,
        'subset': config['subset'],
        'max_n_timepoints': config['max_n_timepoints']
    })

    training_folder = check_training_folder(folder, prompt, collection)
    if training_folder:
        # We have made this before
        return training_folder
    training_folder = make_training_folder(folder, prompt)

    # Load and split
    all_images, all_gt, dbs = load_folder(folder, db_files, training_folder, fps=fps,
                                          test_size=test_size, subset=subset,
                                          max_n_timepoints=prompt['max_n_timepoints'])

    prompt["actual_val_split"] = len(dbs['eval'])/(len(dbs['eval'] + dbs['train']))
    benedict(prompt).to_yaml(filepath=training_folder / "db_prompt.yaml")
    benedict(dbs).to_yaml(filepath=training_folder / "train_eval_split.yaml")
    #REMOVE
    if smooth:
        all_gt["eval"] = gaussian_filter(all_gt["eval"], (smooth, smooth), axes=(1, 2))
        all_gt["train"] = gaussian_filter(all_gt["train"], (smooth, smooth), axes=(1, 2))
    stacks = {"image":all_images["eval"],"mask": all_gt["eval"]}
    stacks = normalize_stacks(stacks)
    save_data(training_folder, stacks['image'], stacks['mask'], "eval")

    # Normalize
    # all_images["train"] = np.zeros((1, 17, 256, 256, 1))
    # all_gt["train"] = np.zeros((1, 17, 256, 256, 1))
    stacks = {"image":all_images["train"],"mask": all_gt["train"]}
    stacks = normalize_stacks(stacks)
    save_data(training_folder, stacks['image'], stacks["mask"], "train")

    # Save recorded eval event information for later performance calcs
    with open(training_folder / 'eval_events.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(all_gt['eval_events'])
    return training_folder

def check_training_folder(folder: Path, prompt_req: dict, collection: str):
    print('folder', folder)
    prompt_req['collection'] = collection
    data_folders = reversed(sorted(list((folder.parents[0] / "training_data").glob('data_*'))))
    print('data_folders', data_folders)
    for folder in data_folders:
        if not (folder/'db_prompt.yaml').exists():
            continue
        print(folder)
        prompt = benedict(folder/'db_prompt.yaml')
        del prompt['actual_val_split']
        print('folder', prompt)
        print('req', prompt_req)
        ## REACTIVATE
        if prompt == prompt_req:
            print("Folder exists already")
            return folder
    return False

def make_training_folder(folder:Path, prompt: dict, prefix: str = 'data_'):
    i = 0
    while True:
        folder_name = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        for key, value in prompt.items():
            if key == "subset" and value == 1:
                continue
            if isinstance(value, str):
                folder_name = folder_name + "_" + value
            else:
                folder_name = folder_name + "_" + key[0] + str(round(value*100)/100)
        new_folder = folder.parents[0] / "training_data" / (prefix + folder_name)
        try:
            new_folder.mkdir(parents=True)
            break
        except FileExistsError:
            if i == 0:
                print("Would be same folder name, wait")
                i = 1
            time.sleep(3)
    print(new_folder)
    return new_folder

def save_data(folder:Path, images_eval:np.array, gt_eval:np.array, prefix:str = ""):
    i = 0
    images_file = folder / (prefix + "_images_" + str(i).zfill(2) + ".tif")
    gt_file = folder / (prefix + "_gt_" + str(i).zfill(2) + ".tif")
    while images_file.exists():
        images_file = folder / (prefix + "_images_" + str(i).zfill(2) + ".tif")
        gt_file = folder / (prefix + "_gt_" + str(i).zfill(2) + ".tif")
        i += 1
    tifffile.imwrite(images_file, images_eval)
    tifffile.imwrite(gt_file, gt_eval)

def load_folder(parent_folder:Path, db_files: List = None, training_folder: str = None,
                fps: float = 1., test_size: float = 0.2,
                subset: float|bool = False, max_n_timepoints: int = MAX_N_TIMEPOINTS):

    if db_files is None:
        db_files = list(parent_folder.rglob(r'event_db.yaml'))
        
    dbs = {'train': [], 'eval':[]}
    for db_file in db_files.copy():

        event_dict = benedict(db_file)
        dataset_split = event_dict.get('dataset_split', False)
        if dataset_split:
            # print(f'event added to {dataset_split}, due to db_file')
            dbs[dataset_split].append(db_file)
        else:
            rand_val = random.random()
            if rand_val > test_size:
                dbs['train'].append(db_file)
                event_dict['dataset_split'] = 'train'
            else:
                dbs['eval'].append(db_file)
                event_dict['dataset_split'] = 'eval'
            event_dict.to_yaml(filepath=Path(event_dict['event_path'])/'event_db.yaml') 


    if subset:
        random.shuffle(dbs["train"])
        dbs["train"] = dbs["train"][:int(subset*len(dbs["train"]))]

    all_images = {"train": [], "eval": []}
    all_gt = {"train": [], "eval": [], "eval_events": []}
    print(f"Number of train dbs: {len(dbs['train'])}")
    print(f"Number of eval dbs: {len(dbs['eval'])}")
    pos_neg_dist = {"train": {"pos": 0, "neg": 0}, "eval": {"pos": 0, "neg": 0}}
    for train_eval in ["eval", "train"]:
        for db_file in tqdm(dbs[train_eval]):
            try:
                start = all_gt['eval_events'][-1][-1] + 1
            except IndexError:
                start = 0
            folder = db_file.parents[0]
            images, ground_truth = load_tifs(folder)

            try:
                original_fps = benedict(db_file)["fps"]
            except KeyError:
                print("WARNING: no fps in the db files, using 1fps")
                original_fps = 1
            float_increment = original_fps/fps
            # print('time adjust', float_increment)
            if float_increment <= 2 and float_increment >= 0.5:
                float_increment = 1
            time_increment = round(float_increment)
            if (fps/original_fps > 1.5 #image rate is too low
                or (float_increment%1 > 0.25 and float_increment%1 < 0.75) #frame rate mismatch
                or images.shape[0] < max_n_timepoints*time_increment): # not enough data
                continue
            # print(db_file)
            print(db_file)
            print(images.shape)
            if max_n_timepoints > 1:
                images, ground_truth = make_time_series(images, ground_truth, time_increment, max_n_timepoints=max_n_timepoints)
            if images is not False:
                print(images.shape)
                # print('length', train_eval, len(all_images[train_eval]))
                all_images[train_eval].append(images)
                all_gt[train_eval].append(ground_truth)
                if ground_truth.max() == 0:
                    pos_neg_dist[train_eval]["neg"] += images.shape[0]
                else:
                    pos_neg_dist[train_eval]["pos"] += images.shape[0]
                if train_eval == 'eval':
                    all_gt['eval_events'].append(list(range(start, start + ground_truth.shape[0])))
            # These things can get very big. Save inbetween, when memory almost full.
            if psutil.virtual_memory().percent > 90:
                print("Saving multiple tiff files")
                all_images[train_eval] = np.concatenate(all_images[train_eval])
                all_gt[train_eval] = np.concatenate(all_gt[train_eval])
                save_data(training_folder, all_images[train_eval], all_gt[train_eval], "train")
                all_images[train_eval] = []
                all_gt[train_eval] = []
        # for item in all_images[train_eval]:
        #     print(item.shape)
        all_images[train_eval] = np.concatenate(all_images[train_eval])
        all_gt[train_eval] = np.concatenate(all_gt[train_eval])
        if max_n_timepoints == 1:
            all_images[train_eval] = np.expand_dims(all_images[train_eval], 1)
            all_gt[train_eval] = np.expand_dims(all_gt[train_eval], 1)
        print("All images")
        print(all_images[train_eval].shape)
    print('Distribution of events', pos_neg_dist)
    return all_images, all_gt, dbs


def make_time_series(images, ground_truth, time_increment = 1, length=8, max_n_timepoints=MAX_N_TIMEPOINTS):
    max_gt = ground_truth.max()
    if max_gt < 0.5:
        start = 0
    else:
        start = 0
        pos_frames = []
        for idx, frame in enumerate(ground_truth):
            if frame.max() > max_gt/2:
                pos_frames.append(idx)
        start = pos_frames[max(-length, -len(pos_frames))*time_increment]
        start = start - max_n_timepoints*time_increment - 3
        start = max(start, 0)
        print(start)


    got_data = False
    # for idx in range(start, images.shape[0]-(n_timepoints*time_increment)):

    # print(list(range(start,start+max_n_timepoints+5,time_increment)))
    #if positive event take only substacks that will have positive gt

    # if max_gt > max_gt/2 and gt_images[-1].max() == 0:
    #     print('positive event but negative GT, skip')
    #     return False, False
    gt_images = np.expand_dims(ground_truth[start: start+max_n_timepoints+length:time_increment], 0)
    images = np.expand_dims(images[start:start+max_n_timepoints+length:time_increment], 0)
    
    # gt_matrix.append(gt_image)
    # print(gt_images.shape, max_n_timepoints, length, time_increment)
    print(images.shape)
    if gt_images.shape[1] < max_n_timepoints + length or images.shape[1] < max_n_timepoints + length:
        print('event few frames')
        return False, False
    return images, gt_images


def load_tifs(folder:Path):
    image_file = folder / "images.tif"
    gt_file = folder / "ground_truth.tif"
    images = tifffile.imread(image_file)
    ground_truth = tifffile.imread(gt_file)
    return images, ground_truth


def normalize_stacks(stacks:dict):
    " Normalize the stacks and account for overexposed and low mask values"
    for key in stacks.keys():
        for index, frame in enumerate(stacks[key]):
            if np.max(frame) == np.inf: # overexposed
                frame[frame == frame.max()] = np.finfo(frame.dtype).max

            norm_frame = (frame-np.min(frame)).astype(np.float32)

            if np.max(norm_frame) > 1 and key == "mask": # not 0 to 1
                norm_frame = norm_frame/255
            if np.max(norm_frame) > 0.2:
                norm_frame = norm_frame/np.max(norm_frame)
            stacks[key][index] = norm_frame
        stacks[key] = stacks[key].astype(np.float32)
    return stacks


if __name__ == "__main__": #pragma: no cover
    main()