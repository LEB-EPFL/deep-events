from pathlib import Path
import numpy as np
import datetime
import random
from typing import List
import psutil
import time
from scipy.ndimage import gaussian_filter
import csv

from sklearn.model_selection import train_test_split
import tifffile

from benedict import benedict
from deep_events.database import get_collection

folder = Path("Z:/_Lab members/Emily/event_data")


def main():
    training_folder = make_training_folder(folder)

    # Load and split
    all_images, all_gt = load_folder(folder)
    images_train, images_eval, gt_train, gt_eval = train_test_split(all_images, all_gt, test_size=0.2, random_state=42)
    stacks = {"image":images_eval,"mask": gt_eval}
    stacks = normalize_stacks(stacks)
    save_data(training_folder, stacks['image'], stacks['mask'], "eval")

    # Normalize
    stacks = {"image":images_train,"mask": gt_train}
    stacks = normalize_stacks(stacks)

    save_data(training_folder, all_images['image'], all_images["mask"], "train")


def prepare_for_prompt(folder: Path, prompt: dict, collection: str, test_size = 0.2,
                       n_timepoints = 1, fps = 1, smooth=False):

    training_folder = make_training_folder(folder, prompt)
    if "n_timepoints" in prompt.keys():
        n_timepoints = prompt["n_timepoints"]
        del prompt["n_timepoints"]
    if "fps" in prompt.keys():
        fps = prompt["fps"]
        del prompt["fps"]
    if "smooth" in prompt.keys():
        smooth = prompt["smooth"]
        del prompt["smooth"]
    if "train_val_split" in prompt.keys():
        test_size = prompt["train_val_split"]
        del prompt["train_val_split"]
    if "subset" in prompt.keys():
        subset = prompt["subset"]
        del prompt["subset"]
    else:
        subset = False

    coll = get_collection(collection)
    print(prompt)
    filtered_list = list(coll.find(prompt))
    db_files = []
    print(len(filtered_list))

    for item in filtered_list:
        db_files.append(Path(item['event_path']) / "event_db.yaml")

    prompt["train_val_split"] = test_size
    prompt["n_timepoints"]  = n_timepoints
    prompt["fps"] = fps
    prompt["smooth"] = smooth
    prompt["collection"] = collection
    prompt["n_event"] = int(len(db_files)*(subset or 1))
    prompt["subset"] = subset
    print("Number of events:", prompt["n_event"])
    benedict(prompt).to_yaml(filepath=training_folder / "db_prompt.yaml")

    # Load and split
    all_images, all_gt = load_folder(folder, db_files, training_folder, n_timepoints=n_timepoints, fps=fps,
                                     test_size=test_size, subset=subset)
    #Shuffle training data
    seed=420
    print(all_images['train'].shape)
    np.random.seed(seed)
    p = np.random.permutation(all_images['train'].shape[0])
    all_images['train'] = all_images['train'][p]
    all_gt['train'] = all_gt['train'][p]

    if smooth:
        all_gt["eval"] = gaussian_filter(all_gt["eval"], (smooth, smooth), axes=(1, 2))
        all_gt["train"] = gaussian_filter(all_gt["train"], (smooth, smooth), axes=(1, 2))
    stacks = {"image":all_images["eval"],"mask": all_gt["eval"]}
    stacks = normalize_stacks(stacks)
    save_data(training_folder, stacks['image'], stacks['mask'], "eval")

    # Normalize
    stacks = {"image":all_images["train"],"mask": all_gt["train"]}
    stacks = normalize_stacks(stacks)
    save_data(training_folder, stacks['image'], stacks["mask"], "train")

    # Save recorded eval event information for later performance calcs
    with open(training_folder / 'eval_events.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(all_gt['eval_events'])
    return training_folder


def make_training_folder(folder:Path, prompt: dict):
    i = 0
    while True:
        folder_name = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        for key, value in prompt.items():
            if key == "subset" and value == 1:
                continue
            if isinstance(value, str):
                folder_name = folder_name + "_" + value
            else:
                folder_name = folder_name + "_" + key[0] + str(value)
        new_folder = folder.parents[0] / "training_data" / folder_name
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
                n_timepoints: int = 1, fps: float = 1., test_size: float = 0.2,
                subset: float|bool = False):

    if db_files is None:
        db_files = list(parent_folder.rglob(r'event_db.yaml'))
    
    #Split eval/train on the event level
    dbs = {}
    random.shuffle(db_files)
    dbs['eval'] = db_files[:int(test_size*len(db_files))]
    dbs['train'] = db_files[int(test_size*len(db_files)):]

    if subset:
        random.shuffle(dbs["train"])
        dbs["train"] = dbs["train"][:int(subset*len(dbs["train"]))]

    all_images = {"train": [], "eval": []}
    all_gt = {"train": [], "eval": [], "eval_events": []}
    print(f"Number of train dbs: {len(dbs['train'])}")
    print(f"Number of eval dbs: {len(dbs['eval'])}")
    for train_eval in ["train", "eval"]:
        for db_file in dbs[train_eval]:
            try:
                start = all_gt['eval_events'][-1][-1] + 1
            except IndexError:
                start = 0
            folder = db_file.parents[0]
            images, ground_truth = load_tifs(folder)

            if n_timepoints > 1:
                try:
                    original_fps = benedict(db_file)["fps"]
                except KeyError:
                    print("WARNING: no fps in the db files, using 1fps")
                    original_fps = 1
                time_increment = round(original_fps/fps)
                if (fps/original_fps > 1.1 #image rate is too low
                    or (original_fps/fps%1 > 0.25 and original_fps/fps%1 < 0.75) #frame rate mismatch
                    or images.shape[0] < n_timepoints*time_increment): # not enough data
                    continue
                images, ground_truth = make_time_series(images, ground_truth, n_timepoints, time_increment)

            all_images[train_eval].append(images)
            all_gt[train_eval].append(ground_truth)
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
        all_images[train_eval] = np.concatenate(all_images[train_eval])
        all_gt[train_eval] = np.concatenate(all_gt[train_eval])
    return all_images, all_gt


def make_time_series(images, ground_truth, n_timepoints, time_increment = 1):
    image_matrix = []
    gt_matrix = []
    for idx in range(images.shape[0]-(n_timepoints*time_increment)+1):
        image_matrix.append(images[idx:idx+(n_timepoints*time_increment):time_increment])
        gt_matrix.append(ground_truth[idx+(n_timepoints*time_increment)-1])
    return np.stack(image_matrix), np.stack(gt_matrix)


def load_tifs(folder:Path):
    image_file = folder / "images.tif"
    gt_file = folder / "ground_truth.tif"
    images = tifffile.imread(image_file).astype(np.float32)
    ground_truth = tifffile.imread(gt_file).astype(np.float32)
    return images, ground_truth


def normalize_stacks(stacks:dict):
    " Normalize the stacks and account for overexposed and low mask values"
    for key in stacks.keys():
        for index, frame in enumerate(stacks[key]):
            if np.max(frame) == np.inf: # overexposed
                frame[frame == frame.max()] = np.finfo(frame.dtype).max

            norm_frame = (frame-np.min(frame))

            if np.max(norm_frame) > 1 and key == "mask": # not 0 to 1
                norm_frame = norm_frame/255
            if np.max(norm_frame) > 0.2:
                norm_frame = norm_frame/np.max(norm_frame)

            stacks[key][index] = norm_frame
    return stacks


if __name__ == "__main__": #pragma: no cover
    main()