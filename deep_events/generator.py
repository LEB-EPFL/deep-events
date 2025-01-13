import numpy as np
import tifffile
import os
from pathlib import Path
from keras.utils import Sequence
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
from deep_events.augmentation import elastic_transform_3d_as_2d_slices, warp_2d_slice, tf_apply_augmentation

# set seeds
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

GENERATOR = ImageDataGenerator(
            horizontal_flip=True,
            vertical_flip=True,
            rotation_range=30,
            width_shift_range=20,
            height_shift_range=20,
        )

def apply_augmentation(self, xs, ys, x_size=128, y_size=128, performance=False):
    crop_pos = (xs.shape[-3] - x_size)//2
    new_x = np.zeros_like(xs[:, crop_pos:crop_pos+x_size, crop_pos:crop_pos+x_size, :])
    new_y = np.zeros_like(ys[:, crop_pos:crop_pos+x_size, crop_pos:crop_pos+x_size, :])
    for i, (x, y) in enumerate(zip(xs, ys)):
    # while True: #dummy loop
        params = self.generator.get_random_transform(x.shape[:-1])
        bright = np.random.default_rng().uniform(self.brightness_range[0], self.brightness_range[1], size=(1))
        # print('SHAPE', x.shape)
        if len(x.shape) > 3:
            # print('CHANNEL AUG')
            for c in range(x.shape[-1]):
                x[..., c] = self.generator.apply_transform(x[..., c], params)
                if not performance:
                    x[..., c] = x[..., crop_pos:crop_pos+x_size, crop_pos:crop_pos+x_size, c]
                x[..., c] = x[..., c]/x[..., c].max()
            y[..., 0] = self.generator.apply_transform(y[..., 0], params)
            if not performance:
                y[..., 0] = y[..., crop_pos:crop_pos+x_size, crop_pos:crop_pos+x_size, 0]
        else:
            # print("NON-CHANNEL AUG")
            x = self.generator.apply_transform(x, params)
            y = self.generator.apply_transform(y, params)
            if not performance:
                x = x[crop_pos:crop_pos+x_size, crop_pos:crop_pos+x_size, :]
                y = y[crop_pos:crop_pos+x_size, crop_pos:crop_pos+x_size, :]
            #renormalize
            x = x/x.max()
            # fig, ax = plt.subplots(1, x.shape[2])
            # for idx in range(x.shape[2]):
            #     ax[idx].imshow(x[..., idx])
            # plt.show()

        if self.poisson > 0.01:
            intensity_scale = 100 / self.poisson  # Higher noise_level means fewer photons
            gaussian_std = 0.01 * self.poisson   # Higher noise_level means more Gaussian noise
            poisson_noisy = np.random.poisson(x * intensity_scale) / intensity_scale
            gaussian_noisy = poisson_noisy + np.random.normal(0, gaussian_std, x.shape)
            x = np.clip(gaussian_noisy, 0, 1)

        x = x*bright
        gamma = np.random.uniform(low=0.8, high=1.2)
        x = x ** gamma
        shift_val = np.random.uniform(-0.1, 0.1)
        x = np.clip(x + shift_val, 0, 1)


        if not performance:
            x, dx, dy = elastic_transform_3d_as_2d_slices(x, alpha=10.0, sigma=3.0, order=1)
            y[..., 0] = warp_2d_slice(y[..., 0], dx, dy, order=0)
        # break
        new_x[i] = x
        new_y[i] = y
    return new_x, new_y
    # return x, y


class ArraySequence(Sequence):
    def __init__(self, data_dir:Path, batch_size, augment=True, n_augmentations=10,
                 brightness_range=[0.9, 1], poisson=0, subset_fraction=1, validation=False, t_size=1):
        self.data_dir = data_dir
        self.n_augmentations = n_augmentations
        self.batch_size = batch_size
        self.augment = augment
        self.brightness_range = brightness_range
        self.poisson = poisson
        self.generator = GENERATOR
        self.subset_fraction = subset_fraction if not validation else 1.0 
        self.t_size = t_size
        self.last_frame = False
        self.performance = False

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
            self.gt_array = np.moveaxis(self.gt_array, 1, -1)

        print("Number of frames in generator: ", self.num_samples)

        self.on_epoch_end()

    def on_epoch_end(self):
        pass
        # Update indices after each epoch
        # total_samples = self.num_samples
        # subset_size = max(1, int(total_samples * self.subset_fraction))
        # self.indices = np.random.choice(total_samples, subset_size, replace=False) if not self.validation else np.arange(total_samples)

    def __len__(self):
        return self.num_samples# * self.n_augmentations / float(self.batch_size)))

    def __getitem__(self, idx):
        # batch_x = []
        # batch_y = []
        # start_index = (idx * self.batch_size) % len(self.indices)
        # i = start_index

        # while True:
            # if i >= len(self.indices):
            #     i = 0

        
        x = self.images_array[idx]
        y = self.gt_array[idx]

        if x.ndim == 2:
            x = np.expand_dims(x, axis=-1)
            y = np.expand_dims(y, axis=-1)
        y_all = y

        # temporal subsample
        if self.last_frame:
            last_frame = self.last_frame
        elif self.validation:
            last_frame = -1
        else:
            last_frame = tf.random.uniform([], -5, 0, dtype=tf.int32)
        # print('last_frame', last_frame)

        x_frames = list(range(-self.t_size+last_frame+1, last_frame+1))
        if tf.random.uniform([], 0, 1, dtype=tf.float32) < 0.1 and self.t_size > 1 and not self.validation and len(x_frames) > 1 and not self.performance:
            # in 10% of cases drop a frame
            to_drop = tf.random.uniform([], 0, len(x_frames)-1, dtype=tf.int32)
            x_frames.remove(x_frames[to_drop])
            x_frames = [-self.t_size+last_frame] + x_frames
        y = y_all[..., x_frames[-1]]
        j = 1
        while y_all.max() > 0.5 and y.max() < 0.5:
            y = y_all[..., x_frames[-1] + j]
            j += 1
            # fig, axs = plt.subplots(1, 3)
            # fig.set_size_inches(15, 5)
            # axs[0].imshow(x[..., x_frames[-1]])
            # axs[1].imshow(y, clim=(0, 1))
            # axs[2].imshow(x[..., -1])
            # print(x_frames)
            # print(x.shape)
            # plt.show()
        x = x[..., x_frames]
        y = np.expand_dims(y, -1)
        # if self.augment and not self.validation:
        #     x, y = self.apply_augmentation(x, y)
        # elif self.performance:
        #     x, y = self.apply_augmentation(x, y)
        # else:
        #     x_size = 128
        #     x = x[56:56+x_size, 56:56+x_size, :]
        #     y = y[56:56+x_size, 56:56+x_size, :]

        # batch_x.append(x)
        # batch_y.append(y)

        # if len(batch_x) >= self.batch_size or self.performance:
            # break
            #i += 1

        # batch_x = np.array(batch_x, dtype=np.float32)
        # batch_y = np.array(batch_y, dtype=np.float32)
        # return batch_x, batch_y
        return x, y

    # def apply_augmentation(self, x, y):
    #     tf_apply_augmentation(x, y, brightness_range=self.brightness_range, poisson=self.poisson, performance=self.performance)
    #     # return apply_augmentation(self, x, y, performance=self.performance)
    

def create_tf_dataset(data_dir: Path, batch_size, augment=True, 
                      brightness_range=[0.6, 1], poisson=0.1, 
                      validation=False, t_size=1, performance=False):
    """
    Creates a tf.data.Dataset based on the functionality of ArraySequence.
    """

    # Load data
    if validation:
        images_file = data_dir / "eval_images_00.tif"
        gt_file = data_dir / "eval_gt_00.tif"
    else:
        images_file = data_dir / "train_images_00.tif"
        gt_file = data_dir / "train_gt_00.tif"

    images_array = tifffile.imread(images_file)
    gt_array = tifffile.imread(gt_file)

    # Correct dimensions for TensorFlow
    if len(images_array.shape) > 3:
        images_array = np.moveaxis(images_array, 1, -1)
        gt_array = np.moveaxis(gt_array, 1, -1)

    num_samples = images_array.shape[0]
    print(f"Number of frames in dataset: {num_samples}")

    # Convert data to TensorFlow tensors
    images_tensor = tf.convert_to_tensor(images_array, dtype=tf.float32)
    gt_tensor = tf.convert_to_tensor(gt_array, dtype=tf.float32)

    # Create a TensorFlow Dataset
    dataset = tf.data.Dataset.from_tensor_slices((images_tensor, gt_tensor))
    # dataset = dataset.cache()

    # Define temporal subsampling logic
    def temporal_subsample(x, y, t_size=5, validation=False, performance=False):
        num_frames = tf.shape(y)[-1]  # Total number of frames
        validation = tf.constant(validation, dtype=tf.bool)

        # Determine last_frame
        last_frame = tf.cond(
            validation,
            lambda: tf.constant(-1, dtype=tf.int32),
            lambda: tf.random.uniform([], -5, 0, dtype=tf.int32)
        )

        # Create x_frames
        x_frames = tf.range(-t_size + last_frame + 1, last_frame + 1, dtype=tf.int32)
        x_frames = tf.where(x_frames < 0, x_frames + num_frames, x_frames)  # Handle negative indices
        x_frames = tf.clip_by_value(x_frames, 0, num_frames - 1)  # Ensure valid indices

        # Ensure x_frames has exactly `t_size` frames
        x_frames = tf.pad(x_frames, [[0, tf.maximum(0, t_size - tf.shape(x_frames)[0])]])[:t_size]

        # Subsample x and y
        x_subsampled = tf.gather(x, x_frames, axis=-1)
        y_target = y[..., x_frames[-1]]

        return x_subsampled, tf.expand_dims(y_target, -1)



    dataset = dataset.map(temporal_subsample, num_parallel_calls=tf.data.AUTOTUNE)
    # Batch, shuffle, and prefetch
    if not validation:
        dataset = dataset.shuffle(buffer_size=num_samples)
    # dataset = dataset.cache()
    # Apply augmentations if needed
    if augment:
        def augmentation_wrapper(x, y):
            return tf_apply_augmentation(
                x, y,
                x_size=128,
                y_size=128,
                brightness_range=brightness_range,
                poisson=poisson,
                performance=performance
            )

        dataset = dataset.map(augmentation_wrapper, num_parallel_calls=tf.data.AUTOTUNE)


    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


if __name__ == "__main__":
    data_folder = Path(r"X:\Scientific_projects\deep_events_WS\data\original_data\training_data\data_20241227_1500_brightfield_cos7_n1_f1_mito_events")
    seq = ArraySequence(data_folder, 32, t_size=5)
    print(seq.__getitem__(10)[0].shape)
    print(seq.__getitem__(10)[1].shape)
