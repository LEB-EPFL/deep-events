from pathlib import Path
import datetime
from multiprocessing import Pool, Lock
import time

from benedict import benedict
import tifffile
import tensorflow as tf
import numpy as np

from training_functions import create_model
from generator import ArraySequence

FOLDER = Path("//lebsrv2.epfl.ch/LEB_SHARED/SHARED/_Lab members/Juan/training_data")
SETTINGS = {"nb_filters": 16,
            "first_conv_size": 32,
            "nb_input_channels": 1,
            "batch_size": 16,
            "epochs": 20,
            "n_augmentations": 10}
NAME = datetime.datetime.now().strftime("%Y%m%d_%H%M")

lock = Lock()
def main():
    tf.keras.backend.clear_session()
    gpus = ['GPU:1/', 'GPU:2/', 'GPU:4/']
    folders = ["20230412_1622", "20230413_1436", "20230413_1750"]
    folders = [FOLDER/folder for folder in folders]
    with Pool(3) as p:
        p.starmap(train, zip(folders, gpus))



def train(folder: Path = None, gpu = 'GPU:0/'):
    
    lock.acquire()
    if folder is None:
        latest_folder = get_latest_folder(FOLDER)
    else:
        latest_folder = folder
    
    batch_generator = ArraySequence(latest_folder, SETTINGS["batch_size"],
                                     n_augmentations=SETTINGS["n_augmentations"])
    eval_images = adjust_tf_dimensions(tifffile.imread(latest_folder / "eval_images_00.tif"))
    eval_mask = adjust_tf_dimensions(tifffile.imread(latest_folder / "eval_gt_00.tif"))

    validation_data = (eval_images, eval_mask)

    print(latest_folder)
    benedict(SETTINGS).to_yaml(filepath = latest_folder / (NAME + "_settings.yaml"))
    model = create_model(SETTINGS["nb_filters"], SETTINGS["first_conv_size"], SETTINGS["nb_input_channels"])

    steps_per_epoch = np.floor(batch_generator.__len__())

    gpu = tf.device(gpu)
    time.sleep(10)
    lock.release()
    
    with gpu:
        history = model.fit(batch_generator,
                            batch_size = SETTINGS["batch_size"],
                            epochs = SETTINGS["epochs"],
                            steps_per_epoch = steps_per_epoch,
                            shuffle=True,
                            validation_data = validation_data,
                            verbose=1)
    model.save(latest_folder / (NAME + "_model.h5"))

# 
# for gpu in tf.config.experimental.list_physical_devices('GPU'):
#     tf.config.experimental.set_memory_growth(gpu, True)


 


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


def test_model():
    import matplotlib.pyplot as plt
    frame = 10
    latest_folder = get_latest_folder(folder)
    model = tf.keras.models.load_model(latest_folder / "model.h5")
    eval_images = adjust_tf_dimensions(tifffile.imread(latest_folder / "eval_images.tif"))
    output = model.predict(np.expand_dims(eval_images[frame],axis=0))
    eval_mask = adjust_tf_dimensions(tifffile.imread(latest_folder / "eval_gt.tif"))
    plt.imshow(output[0, :, :, 0], vmax=0.5)
    plt.figure()
    plt.imshow(eval_mask[frame, :, :, 0])
    plt.figure()
    plt.imshow(eval_images[frame, :, :, 0])
    plt.show()

if __name__ == "__main__":
    main()