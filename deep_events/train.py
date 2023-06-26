from pathlib import Path
import datetime
from multiprocessing import Pool, Lock
import time

from benedict import benedict
import tifffile
import tensorflow as tf
import numpy as np
import os

from deep_events.training_functions import create_model
from deep_events.generator import ArraySequence



FOLDER = Path("//lebsrv2.epfl.ch/LEB_SHARED/SHARED/_Lab members/Juan/training_data")
SETTINGS = {"nb_filters": 16,
            "first_conv_size": 12,
            "nb_input_channels": 1,
            "batch_size": 16,
            "epochs": 20,
            "n_augmentations": 30,
            'brightness_range': [0.6, 1],
            "loss": 'binary_crossentropy'}



def main(): # pragma: no cover
    tf.keras.backend.clear_session()
    gpus = ['GPU:1/', 'GPU:2/', 'GPU:4/']
    folders = ["20230412_1622", "20230413_1436", "20230413_1750"]
    folders = [FOLDER/folder for folder in folders]
    with Pool(3) as p:
        p.starmap(train, zip(folders, gpus))

def distributed_train(folders, gpus, settings=SETTINGS):
    l = Lock()

    if not isinstance(settings, list):
        settings = [settings]*len(folders)

    with Pool(min(3, len(folders)), initializer=init_pool, initargs=(l,)) as p:
        p.starmap(train, zip(folders, gpus, settings))

def init_pool(l: Lock):
    global lock
    lock = l

lock = Lock()

def train(folder: Path = None, gpu = 'GPU:2/', settings: dict = SETTINGS):
    time.sleep(np.random.random()*10)
    lock.acquire()
    print("LOCKED")
    if folder is None:
        latest_folder = get_latest_folder(FOLDER)
    else:
        latest_folder = folder

    logdir = latest_folder.parents[0] / ("logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    batch_generator = ArraySequence(latest_folder, settings["batch_size"],
                                     n_augmentations=settings["n_augmentations"],
                                     brightness_range=settings['brightness_range'])
    eval_images = adjust_tf_dimensions(tifffile.imread(latest_folder / "eval_images_00.tif"))
    eval_mask = adjust_tf_dimensions(tifffile.imread(latest_folder / "eval_gt_00.tif"))

    validation_data = (eval_images, eval_mask)
    time.sleep(1)
    name = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    while Path(latest_folder / (name + "_settings.yaml")).is_file():
        name = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    benedict(settings).to_yaml(filepath = latest_folder / (name + "_settings.yaml"))
    lock.release()
    print("UNLOCKED")


    n_tries = 0
    max_tries = 10
    while n_tries < max_tries:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu[-2]
        model = create_model(settings, eval_images.shape)

        steps_per_epoch = np.floor(batch_generator.__len__())
        print(f"NUMBER OF EPOCHS: {settings['epochs']}" )
        try:
            gpu_device = tf.device(gpu)
            with gpu_device:
                history = model.fit(batch_generator,
                                    batch_size = settings["batch_size"],
                                    epochs = settings["epochs"],
                                    steps_per_epoch = steps_per_epoch,
                                    shuffle=True,
                                    validation_data = validation_data,
                                    verbose = 1,
                                    callbacks = [tensorboard_callback])
            n_tries = max_tries
        except Exception as e:
            print("------------------------------ COULD NOT TRAIN -----------------------------------------")
            print(e)
            gpu = gpu.replace(gpu[-2], str((int(gpu[-2])+1)%6))
            time.sleep(10)
            print(f"Try next GPU: {gpu}")
        n_tries += 1

    tf.keras.models.save_model(model, latest_folder / (name + "_model.h5"), save_traces=True)

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
    return subfolders if subfolders else None


def get_latest(pattern, folder:Path):
    files = [f for f in folder.glob('*') if f.is_file()]
    files = [f for f in files if pattern in f.name]
    files.sort(reverse=True)
    print(files)
    return folder / files[0]


def adjust_tf_dimensions(stack:np.array):
    return np.expand_dims(stack, axis=-1)


def test_model():
    import matplotlib.pyplot as plt
    gpu = tf.device("GPU:4/")
    with gpu:
        frame = 1
        training_folder = Path(get_latest_folder(FOLDER)[0])
        print(training_folder)
        model_dir = get_latest("model", training_folder)
        model = tf.keras.models.load_model(model_dir)
        eval_images = adjust_tf_dimensions(tifffile.imread(training_folder / "eval_images_00.tif"))
        eval_mask = adjust_tf_dimensions(tifffile.imread(training_folder / "eval_gt_00.tif"))
        while True:
            output = model.predict(np.expand_dims(eval_images[frame],axis=0))
            f, axs = plt.subplots(1, 3)
            f.set_size_inches(15,5)
            axs[0].imshow(output[0, :, :, 0], vmax=1)
            axs[0].set_title("prediction")
            axs[1].imshow(eval_mask[frame, :, :, 0])
            axs[1].set_title("ground truth")
            axs[2].imshow(eval_images[frame, :, :, 0])
            plt.show()
            frame = frame + 1

if __name__ == "__main__":
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3" # set the GPU ID
    test_model()