from pathlib import Path
import numpy as np
import datetime

from sklearn.model_selection import train_test_split
import tifffile
import albumentations as albs

folder = Path("Z:/_Lab members/Emily/event_data")
n_augmentations = 20

def main():
    training_folder = make_training_folder(folder)

    # Load and split
    all_images, all_gt = load_folder(folder)
    images_train, images_eval, gt_train, gt_eval = train_test_split(all_images, all_gt, test_size=0.2, random_state=42)
    save_data(training_folder, images_eval, gt_eval, "eval")

    # Augment
    all_images = augment_stacks({"image":images_train,"mask": gt_train}, n_augmentations)
    save_data(training_folder, all_images['image'], all_images["mask"], "train")


def make_training_folder(folder:Path):
    folder_name = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    folder = folder / folder_name
    folder.mkdir(exist_ok=True)
    return folder


def save_data(folder:Path, images_eval:np.array, gt_eval:np.array, prefix:str = ""):
    images_file = folder / (prefix + "_images.tif")
    gt_file = folder / (prefix + "_gt.tif")
    tifffile.imwrite(images_file, images_eval)
    tifffile.imwrite(gt_file, gt_eval)



def load_folder(parent_folder:Path):
    db_files = list(parent_folder.rglob(r'db.yaml'))
    all_images = []
    all_gt = []
    print(f"Number of db files: {len(db_files)}")
    for db_file in db_files:
        folder = db_file.parents[0]
        images, ground_truth = load_tifs(folder)
        all_images.append(images)
        all_gt.append(ground_truth)
    all_images = np.concatenate(all_images)
    all_gt = np.concatenate(all_gt)
    return all_images, all_gt

def load_tifs(folder:Path):
    image_file = folder / "images.tif"
    gt_file = folder / "ground_truth.tif"

    images = tifffile.imread(image_file).astype(np.float32)
    ground_truth = tifffile.imread(gt_file).astype(np.float32)
    return images, ground_truth

def augment_stacks(stacks:dict, n_augmentations:int):
    transform = albs.Compose([albs.Rotate(limit=45, p=0.5),
                        albs.HorizontalFlip(p=0.5),
                        albs.VerticalFlip(p=0.5)
                        ])

    augmented_stacks = {}
    for key in stacks.keys():
        augmented_stacks[key] = []

    for frame in range(stacks[list(stacks.keys())[0]].shape[0]):
        frames = {}
        for key in stacks.keys():
            frames[key] = stacks[key][frame]

        for i in range(n_augmentations):
            augmented_frames = transform(**frames)
            for key in stacks.keys():
                augmented_stacks[key].append(augmented_frames[key])

    for key in stacks.keys():
        augmented_stacks[key] = np.stack(augmented_stacks[key])

    return augmented_stacks




if __name__ == "__main__":
    main()