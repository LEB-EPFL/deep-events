import numpy as np
import tifffile
import os
from pathlib import Path
from keras.utils import Sequence
from keras.preprocessing.image import ImageDataGenerator



GENERATOR = ImageDataGenerator(
            horizontal_flip=True,
            rotation_range=30,
        )
BRIGHTNESS_ADJUST = [0, 1]

def apply_augmentation(self, x, y):
    seed = np.random.randint(0, 1e7)
    # seed = np.random.RandomState(seed=None)
    params = self.generator.get_random_transform(x.shape, seed=seed)
    x = self.generator.apply_transform(x, params)
    bright = np.random.default_rng().uniform(self.brightness_range[0], self.brightness_range[1], size=(1))
    x = x*bright
    y = self.generator.apply_transform(y, params)
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

                    if len(batch_x) >= self.batch_size:
                        break


                if len(batch_x) >= self.batch_size:
                    break

        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        return batch_x, batch_y

    def apply_augmentation(self, x, y):
        return apply_augmentation(self,x,y)



class ArraySequence(Sequence):
    def __init__(self, data_dir:Path, batch_size, augment=True, n_augmentations=10,
                 brightness_range=[0.9, 1]):
        self.data_dir = data_dir
        self.n_augmentations = n_augmentations
        self.batch_size = batch_size
        self.augment = augment
        self.brightness_range = brightness_range
        self.generator = GENERATOR

        self.images_file = data_dir / "train_images_00.tif"
        self.gt_file = data_dir / "train_gt_00.tif"
        with tifffile.TiffFile(self.images_file) as tif_input, tifffile.TiffFile(self.gt_file) as tif_gt:
            self.num_samples = len(tif_input.pages)
            self.images_array = tif_input.asarray()
            self.gt_array = tif_gt.asarray()
        print("Number of frames in generator: ", self.num_samples)

    def __len__(self):
        return int(np.ceil(self.num_samples * self.n_augmentations / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = []
        batch_y = []
        start_index = (idx * self.batch_size) % self.num_samples
        # print("\n start", start_index)
        # end_index = min((idx + 1) * self.batch_size, self.num_samples)
        i = -1
        while True:
            i += 1
            #reset if we went over the total number of frames
            i = 0 if i >= self.num_samples else i
            if start_index <= 0:
                x = self.images_array[i]
                y = self.gt_array[i]
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

            if len(batch_x) >= self.batch_size:
                break

        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        return batch_x, batch_y

    def apply_augmentation(self, x, y):
        return apply_augmentation(self,x,y)