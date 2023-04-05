from pathlib import Path
import datetime

import tifffile
import tensorflow as tf
import numpy as np

from training_functions import create_model


folder = Path("//lebsrv2.epfl.ch/LEB_SHARED/SHARED/_Lab members/Emily/20230329_FtsW_sfGFP_caulobacter_zeiss/training_data")
nb_filters, first_conv_size, nb_input_channels =  8, 9, 1

tf.keras.backend.clear_session()
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
gpu = tf.device('GPU:1/')

def main():
    latest_folder = get_latest_folder(folder)
    train_imgs = adjust_tf_dimensions(tifffile.imread(latest_folder / "train_images.tif"))
    eval_images = adjust_tf_dimensions(tifffile.imread(latest_folder / "eval_images.tif"))
    train_mask = adjust_tf_dimensions(tifffile.imread(latest_folder / "train_gt.tif"))
    eval_mask = adjust_tf_dimensions(tifffile.imread(latest_folder / "eval_gt.tif"))
    print(train_imgs.shape)

    validation_data = (eval_images, eval_mask)

    print(latest_folder)
    model = create_model(nb_filters, first_conv_size, nb_input_channels)



    with gpu:
        history = model.fit(train_imgs, train_mask,
                            batch_size = 16,
                            epochs = 20,
                            shuffle=True,
                            validation_data = validation_data,
                            verbose=2)


def get_latest_folder(parent_folder:Path):
    subfolders = [f for f in parent_folder.glob('*') if f.is_dir()]
    datetime_format = '%Y%m%d_%H%M'
    subfolders = [f for f in subfolders if f.name.count('_') == 1 and
                  datetime.datetime.strptime(f.name, datetime_format)]
    subfolders.sort(key=lambda x: datetime.datetime.strptime(x.name, datetime_format),
                    reverse=True)
    return Path(subfolders[0]) if subfolders else None


def adjust_tf_dimensions(stack:np.array):
    return np.expand_dims(stack, axis=-1)


if __name__ == "__main__":
    main()