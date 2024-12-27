import numpy as np
import tifffile
import os
from pathlib import Path
from keras.utils import Sequence
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

seed=42
np.random.seed(seed)

GENERATOR = ImageDataGenerator(
            horizontal_flip=True,
            vertical_flip=True,
            rotation_range=30,
            width_shift_range=10,
            height_shift_range=10,
        )

def apply_augmentation(self, x, y, x_size=128, y_size=128):
    params = self.generator.get_random_transform(x.shape[:-1], seed=seed)
    bright = np.random.default_rng().uniform(self.brightness_range[0], self.brightness_range[1], size=(1))
    x_old = x.copy()
    crop_pos = (x.shape[-3] - x_size)//2
    if len(x.shape) > 3:
        for c in x.shape[-1]:
            x[..., c] = self.generator.apply_transform(x[..., c], params)
            x[..., c] = x[..., crop_pos:crop_pos+x_size, crop_pos:crop_pos+x_size, c]
        y[..., 0] = self.generator.apply_transform(y[..., 0], params)
        y[..., 0] = y[..., crop_pos:crop_pos+x_size, crop_pos:crop_pos+x_size, 0]
    else:
        x = self.generator.apply_transform(x, params)
        x = x[crop_pos:crop_pos+x_size, crop_pos:crop_pos+x_size, :]
        y = self.generator.apply_transform(y, params)
        y = y[crop_pos:crop_pos+x_size, crop_pos:crop_pos+x_size, :]

    if self.poisson > 0.01:
        intensity_scale = 100 / self.poisson  # Higher noise_level means fewer photons
        gaussian_std = 0.01 * self.poisson   # Higher noise_level means more Gaussian noise
        poisson_noisy = np.random.poisson(x * intensity_scale) / intensity_scale
        gaussian_noisy = poisson_noisy + np.random.normal(0, gaussian_std, x.shape)
        x = np.clip(gaussian_noisy, 0, 1)
    x = x*bright
    return x, y


class FileSequence(Sequence):
    def __init__(self, data_dir, batch_size, augment=True, n_augmentations=10):
        self.data_dir = data_dir
        self.n_augmentations = n_augmentations
        self.batch_size = batch_size
        self.augment = augment
        self.generator = GENERATOR
        self.images_prefix = "train_images"
        self.gt_prefix = "train_gt"
        self.file_list = sorted(os.listdir(data_dir))
        self.num_samples = sum([len(tifffile.TiffFile(os.path.join(data_dir, f)).pages) for f in self.file_list if f.startswith(self.images_prefix)])
        print("Number of frames in generator: ", self.num_samples)

    def __len__(self):
        return int(np.ceil(self.num_samples * self.n_augmentations / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = []
        batch_y = []
        start_index = (idx * self.batch_size) % self.num_samples
        # print("\n start", start_index)
        # end_index = min((idx + 1) * self.batch_size, self.num_samples)

        for f in self.file_list:
            if not f.startswith(self.images_prefix):
                continue
            images_file = os.path.join(self.data_dir, f)
            gt_file = os.path.join(self.data_dir, f.replace(self.images_prefix, self.gt_prefix))
            with tifffile.TiffFile(images_file) as tif_input, tifffile.TiffFile(gt_file) as tif_gt:
                num_pages = len(tif_input.pages)
                i = -1
                while True:
                    i += 1
                    #reset if we went over the total number of frames
                    i = 0 if i >= num_pages else i
                    if start_index <= 0:
                        input_page = tif_input.pages[i]
                        gt_page = tif_gt.pages[i]
                        x = input_page.asarray()
                        y = gt_page.asarray()
                        if x.ndim == 2:
                            x = np.expand_dims(x, axis=-1)
                            y = np.expand_dims(y, axis=-1)
                    else:
                        start_index -= 1
                        continue
                    if self.augment:
                        x, y = self.apply_augmentation(x, y)

                    batch_x.append(x)
                    batch_y.append(y)
                    # plt.imshow(x)
                    # plt.figure()
                    # plt.imshow(y)
                    # plt.show()
                    if len(batch_x) >= self.batch_size:
                        break


                if len(batch_x) >= self.batch_size:
                    break

        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        return batch_x[:, 64:128, 64:128, :], batch_y[:, 64:128, 64:128, :]

    def apply_augmentation(self, x, y):
        return apply_augmentation(self,x,y)


class ArraySequence(Sequence):
    def __init__(self, data_dir:Path, batch_size, augment=True, n_augmentations=10,
                 brightness_range=[0.9, 1], poisson=0, subset_fraction=1, validation=False):
        self.data_dir = data_dir
        self.n_augmentations = n_augmentations
        self.batch_size = batch_size
        self.augment = augment
        self.brightness_range = brightness_range
        self.poisson = poisson
        self.generator = GENERATOR
        self.subset_fraction = subset_fraction if not validation else 1.0 

        self.validation = validation
        if validation:
            self.images_file = data_dir / "eval_images_00.tif"
            self.gt_file = data_dir / "eval_gt_00.tif"
        else:
            self.images_file = data_dir / "train_images_00.tif"
            self.gt_file = data_dir / "train_gt_00.tif"
        with tifffile.TiffFile(self.images_file) as tif_input, tifffile.TiffFile(self.gt_file) as tif_gt:
            self.images_array = tif_input.asarray()
            self.gt_array = tif_gt.asarray()
        self.num_samples = self.images_array.shape[0]

        #Correct dimensions for tensorflow
        if len(self.images_array.shape) > 3:
            self.images_array = np.moveaxis(self.images_array, 1, -1)
            self.gt_array = np.expand_dims(self.gt_array, -1)

        print("Number of frames in generator: ", self.num_samples)

        self.on_epoch_end()

    def on_epoch_end(self):
        # Update indices after each epoch
        total_samples = self.num_samples
        subset_size = max(1, int(total_samples * self.subset_fraction))
        self.indices = np.random.choice(total_samples, subset_size, replace=False) if not self.validation else np.arange(total_samples)

    def __len__(self):
        return int(np.ceil(len(self.indices) * self.n_augmentations / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = []
        batch_y = []
        start_index = (idx * self.batch_size) % len(self.indices)
        i = start_index

        while True:
            if i >= len(self.indices):
                i = 0

            index = self.indices[i]
            x = self.images_array[index]
            y = self.gt_array[index]

            if x.ndim == 2:
                x = np.expand_dims(x, axis=-1)
                y = np.expand_dims(y, axis=-1)

            if self.augment and not self.validation:
                x, y = self.apply_augmentation(x, y)
            # else:
            #     x_size = 128
            #     x = x[56:56+x_size, 56:56+x_size, :]
            #     y = y[56:56+x_size, 56:56+x_size, :]

            batch_x.append(x)
            batch_y.append(y)

            if len(batch_x) >= self.batch_size:
                break
            i += 1

        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        return batch_x, batch_y

    def apply_augmentation(self, x, y):
        return apply_augmentation(self,x,y)