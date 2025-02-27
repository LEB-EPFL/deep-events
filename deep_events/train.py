from pathlib import Path
import datetime
from multiprocessing import Pool, Lock
import time

from benedict import benedict

import numpy as np
import os
import shutil
import random

# set seeds
random.seed(42)
np.random.seed(42)

os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

def adjust_tf_dimensions(stack:np.array):
    if len(stack.shape) < 4:
        return np.expand_dims(stack, axis=-1)
    else:
        return np.moveaxis(stack, 1, -1)


from deep_events.training_functions import get_model_generator
from deep_events.generator import ArraySequence
from deep_events.database.convenience import get_latest_folder
from deep_events import performance
from deep_events.logs import LogImages

FOLDER = Path("//sb-nas1.rcp.epfl.ch/LEB/Scientific_projects/deep_events_WS/data/original_data/training_data/")
SETTINGS = {"nb_filters": 16,
            "first_conv_size": 12,
            "nb_input_channels": 1,
            "batch_size": 32,
            "epochs": 10,
            "n_augmentations": 20,
            'brightness_range': [0.6, 1],
            "loss": 'binary_crossentropy',
            "poisson": 0,
            "subset_fraction": 0.5,
            "initial_learning_rate": 4e-4,
            "n_timepoints": 3}



def main(): # pragma: no cover

    gpus = ['GPU:1/']
    # folders = ["20240806_1401_brightfield_cos7_n3_f1"]
    # folders = [FOLDER/folder for folder in folders]
    # with Pool(3) as p:
    #     p.starmap(train, zip(folders, gpus))
    data_folder = Path(r'\\sb-nas1.rcp.epfl.ch\LEB\Scientific_projects\deep_events_WS\data\original_data\training_data\data_20241227_1500_brightfield_cos7_n1_f1_mito_events')
    folder = Path(r'\\sb-nas1.rcp.epfl.ch\LEB\Scientific_projects\deep_events_WS\data\original_data\training_data\20241227_1526_brightfield_cos7_t0.2_f1_sFalse_mito_events_n753_sFalse')
    train(data_folder, folder)


def distributed_train(data_folders, folders, gpus, settings=SETTINGS):
    l = Lock()

    if not isinstance(settings, list):
        settings = [settings]*len(data_folders)
    distributed = [True]*len(data_folders)
    # os.environ['TF_GPU_ALLOCATOR'] = 0
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
    # folders = [str(x) for x in folders]
    with Pool(min(5, len(data_folders)), initializer=init_pool, initargs=(l,)) as p:
        p.starmap(train, zip(data_folders, folders, gpus, settings, distributed))
    time.sleep(30)  # allow for gpu cleanup
    for folder in set(folders):
        performance.performance_for_folder(folder, general_eval=False)
    time.sleep(30)  # allow for gpu cleanup

def init_pool(l: Lock):
    global lock
    lock = l

lock = Lock()

def train(data_folder: Path = None, folder = None, gpu = 'GPU:2/', settings: dict = SETTINGS, distributed: bool = False):
    if distributed:
        time.sleep(np.random.random()*10)
        lock.acquire()
        print("LOCKED")
    if data_folder is None:
        data_folder = get_latest_folder(FOLDER)

    print('model folder', folder)
    print('data folder', data_folder)
    print(settings)
    logs_dir = folder.parents[0] / (settings.get("log_dir", "logs") + "/scalars/")
    name = short_name = Path(folder).parts[-1][:13]
    logdir = logs_dir / name
    i = 0
    while logdir.exists():
        name = short_name + f"_{i}"
        logdir = logs_dir / name
        i += 1
    print('NAME', name)
    os.mkdir(logdir)
    print(f"Writing settings, logdir {logdir}")

    batch_generator = ArraySequence(data_folder, settings["batch_size"],
                                     n_augmentations=settings["n_augmentations"],
                                     brightness_range=settings['brightness_range'],
                                     poisson=settings["poisson"],
                                     subset_fraction=settings["subset_fraction"],
                                     t_size=settings['n_timepoints'])
    validation_generator = ArraySequence(data_folder, settings["batch_size"],
                                     n_augmentations=settings["n_augmentations"],
                                     brightness_range=settings['brightness_range'],
                                     poisson=settings["poisson"],
                                     validation=True,
                                     t_size=settings['n_timepoints'])


    
    # time.sleep(1)
    # name = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    # while Path(folder / (name + "_settings.yaml")).is_file():
    #     name = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    settings["logdir"] = logdir.name
    settings_folder = str(folder / (name + "_settings.yaml"))
    settings['data_folder'] = data_folder
    print(F"SETTINGS LOCATION: {settings_folder}")
    benedict(settings).to_yaml(filepath = folder / (name + "_settings.yaml"))
    if distributed:
        lock.release()
        print("UNLOCKED")

    n_tries = 0
    max_tries = 10
    while n_tries < max_tries:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu[-2]
        import tensorflow as tf
        tf.random.set_seed(42)
        images_callback = LogImages(logdir, batch_generator, validation_generator, freq=2)
        writer = tf.summary.create_file_writer(str(logdir))
        with writer.as_default():
            for key, value in settings.items():
                tf.summary.text(name=key, data=str(value), step=0)
            writer.flush()
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
        reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1)
        print('val shape', validation_generator.__getitem__(0)[1].shape)
        print('train shape', validation_generator.__getitem__(0)[0].shape)
        model_generator = get_model_generator(settings['model'])
        model = model_generator(settings, batch_generator.__getitem__(0)[0].shape)

        steps_per_epoch = np.floor(batch_generator.__len__())
        print(f"NUMBER OF EPOCHS: {settings['epochs']}")
        try:
            gpu_device = tf.device(gpu)
            with gpu_device:
                history = model.fit(batch_generator,
                                    validation_data = validation_generator,
                                    batch_size = settings["batch_size"],
                                    epochs = settings["epochs"],
                                    steps_per_epoch = steps_per_epoch,
                                    shuffle=True,
                                    verbose = 1,
                                    callbacks = [tensorboard_callback, images_callback, reduce_lr_callback, early_stopping_callback],
                                    # validation_steps=20
                                    )
            n_tries = max_tries
        except Exception as e:
            print("------------------------------ COULD NOT TRAIN -----------------------------------------")
            print(e)
            gpu = gpu.replace(gpu[-2], str((int(gpu[-2])+1)%6))
            time.sleep(10)
            print(f"Try next GPU: {gpu}")
        n_tries += 1

    tf.keras.models.save_model(model, folder / (name + "_model.h5"), save_traces=True)
    





if __name__ == "__main__":
    # gpu = 'GPU:1/'
    # folder = FOLDER / "20241220_1432_brightfield_cos7_n5_f1"

    # train(folder, gpu)
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3" # set the GPU ID
    main()