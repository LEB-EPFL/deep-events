import numpy as np
import tifffile
import os
from keras.utils import Sequence
from keras.preprocessing.image import ImageDataGenerator

class CustomSequence(Sequence):
    def __init__(self, data_dir, input_prefix, gt_prefix, batch_size, augment=True):
        self.data_dir = data_dir
        self.input_prefix = input_prefix
        self.gt_prefix = gt_prefix
        self.batch_size = batch_size
        self.augment = augment
        self.generator = ImageDataGenerator(
            horizontal_flip=True,
            rotation_range=30
        )
        self.file_list = sorted(os.listdir(data_dir))
        self.num_samples = sum([tifffile.TiffFile(os.path.join(data_dir, f)).pages for f in self.file_list if f.startswith(self.input_prefix)])
        
    def __len__(self):
        return int(np.ceil(self.num_samples / float(self.batch_size)))
    
    def __getitem__(self, idx):
        batch_x = []
        batch_y = []
        start_index = idx * self.batch_size
        end_index = min((idx + 1) * self.batch_size, self.num_samples)
        
        for f in self.file_list:
            if not f.startswith(self.input_prefix):
                continue
            input_file = os.path.join(self.data_dir, f)
            gt_file = os.path.join(self.data_dir, f.replace(self.input_prefix, self.gt_prefix))
            with tifffile.TiffFile(input_file) as tif_input, tifffile.TiffFile(gt_file) as tif_gt:
                num_pages = len(tif_input.pages)
                for i in range(num_pages):
                    if start_index <= 0:
                        input_page = tif_input.pages[i]
                        gt_page = tif_gt.pages[i]
                        x = input_page.asarray()
                        y = gt_page.asarray()
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
        seed = np.random.randint(0, 1e7)
        x = self.generator.apply_transform(x, seed)
        y = self.generator.apply_transform(y, seed)
        return x, y
