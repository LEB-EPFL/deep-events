from pathlib import Path
import numpy as np
import datetime
from typing import List
import psutil
import time
from scipy.ndimage import gaussian_filter 

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

    coll = get_collection(collection)
    filtered_list = list(coll.find(prompt))
    db_files = []
    for item in filtered_list:
        db_files.append(Path(item['event_path']) / "event_db.yaml")

    prompt["train_val_split"] = test_size
    prompt["n_timepoints"]  = n_timepoints
    prompt["fps"] = fps
    prompt["smooth"] = smooth
    prompt["collection"] = collection
    benedict(prompt).to_yaml(filepath=training_folder / "db_prompt.yaml")

    # Load and split
    all_images, all_gt = load_folder(folder, db_files, training_folder, n_timepoints, fps)
    images_train, images_eval, gt_train, gt_eval = train_test_split(all_images, all_gt,
                                                                    test_size=test_size, random_state=42)

    if smooth:
        gt_eval = gaussian_filter(gt_eval, (smooth, smooth), axes=(1, 2))
        gt_train = gaussian_filter(gt_train, (smooth, smooth), axes=(1, 2))
    stacks = {"image":images_eval,"mask": gt_eval}
    stacks = normalize_stacks(stacks)
    save_data(training_folder, stacks['image'], stacks['mask'], "eval")

    # Normalize
    stacks = {"image":images_train,"mask": gt_train}
    stacks = normalize_stacks(stacks)
    save_data(training_folder, stacks['image'], stacks["mask"], "train")
    return training_folder


def make_training_folder(folder:Path, prompt: dict):
    i = 0
    while True:
        folder_name = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        for key, value in prompt.items():
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
                n_timepoints: int = 1, fps: float = 1.):


    if db_files is None:
        db_files = list(parent_folder.rglob(r'event_db.yaml'))
    all_images = []
    all_gt = []
    print(f"Number of db files: {len(db_files)}")
    for db_file in db_files:
        folder = db_file.parents[0]
        images, ground_truth = load_tifs(folder)

        if n_timepoints > 1:
            original_fps = benedict(db_file)["fps"]
            time_increment = round(original_fps/fps)
            if (fps/original_fps > 1.1 #image rate is too low
                or (original_fps/fps%1 > 0.25 and original_fps/fps%1 < 0.75) #frame rate mismatch
                or images.shape[0] < n_timepoints*time_increment): # not enough data
                continue
            images, ground_truth = make_time_series(images, ground_truth, n_timepoints, time_increment)

        all_images.append(images)
        all_gt.append(ground_truth)
        # These things can get very big. Save inbetween, when memory almost full.
        if psutil.virtual_memory().percent > 90:
            print("Saving multiple tiff files")
            all_images = np.concatenate(all_images)
            all_gt = np.concatenate(all_gt)
            save_data(training_folder, all_images, all_gt, "train")
            all_images = []
            all_gt = []
    all_images = np.concatenate(all_images)
    all_gt = np.concatenate(all_gt)
    return all_images, all_gt


def make_time_series(images, ground_truth, n_timepoints, time_increment = 1):
    image_matrix = []
    gt_matrix = []
    for idx in range(images.shape[0]-(n_timepoints*time_increment)+1):
        image_matrix.append(images[idx:idx+(n_timepoints*time_increment):time_increment])
        gt_matrix.append(ground_truth[idx+(n_timepoints*time_increment) - 1])
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