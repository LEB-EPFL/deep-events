from pathlib import Path
import datetime
from multiprocessing import Pool, Lock, Process
import time
import warnings
# Suppress the specific end-of-development warning from TensorFlow Addons
warnings.filterwarnings(
    "ignore",
    message=".*TensorFlow Addons (TFA) has ended development*",
    category=UserWarning
)

from benedict import benedict
import gc
import numpy as np
import os
import psutil
import random

from deep_events.augmentation import tf_apply_augmentation

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
from deep_events.generator import ArraySequence, apply_augmentation
from deep_events.database.convenience import get_latest_folder
from deep_events import performance

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
    tf.keras.backend.clear_session()
    gpus = ['GPU:1/']
    # folders = ["20240806_1401_brightfield_cos7_n3_f1"]
    # folders = [FOLDER/folder for folder in folders]
    # with Pool(3) as p:
    #     p.starmap(train, zip(folders, gpus))
    data_folder = Path(r'\\sb-nas1.rcp.epfl.ch\LEB\Scientific_projects\deep_events_WS\data\original_data\training_data\data_20241227_1500_brightfield_cos7_n1_f1_mito_events')
    folder = Path(r'\\sb-nas1.rcp.epfl.ch\LEB\Scientific_projects\deep_events_WS\data\original_data\training_data\20241227_1526_brightfield_cos7_t0.2_f1_sFalse_mito_events_n753_sFalse')
    train(data_folder, folder)


def distributed_train(data_folders, folders, gpus, settings=SETTINGS):
    core_groups = [
    list(range(0, 8)),   # First process uses cores 0-7
    list(range(8, 16)),  # Second process uses cores 8-15
    list(range(16, 24)), # Third process uses cores 16-23
    list(range(24, 32)), # Fourth process uses cores 24-31
    list(range(32, 40))  # Fifth process uses cores 32-39
    ]
    global lock
    lock = Lock()
    if not isinstance(settings, list):
        settings = [settings]*len(data_folders)
    distributed = [True]*len(data_folders)
    # os.environ['TF_GPU_ALLOCATOR'] = 0
   
    # folders = [str(x) for x in folders]
    processes = []
    for idx, (data_folder, folder, gpu, core_group, setting) in enumerate(zip(data_folders, folders, gpus, core_groups, settings)):
        p = Process(target=train, args=(data_folder, folder, gpu, core_group, setting, distributed[idx], lock))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    time.sleep(30)  # allow for gpu cleanup
    for folder in set(folders):
        performance.performance_for_folder(folder, general_eval=False)
    time.sleep(30)  # allow for gpu cleanup

def train(data_folder: Path = None, folder = None, gpu = 'GPU:2/', cores = list(range(0, 8)), settings: dict = SETTINGS, distributed: bool = False, lock=None):
    p = psutil.Process()
    p.cpu_affinity(cores)  
    import tensorflow as tf
    # tf.config.threading.set_intra_op_parallelism_threads(8)
    # tf.config.threading.set_inter_op_parallelism_threads(8)
    from deep_events.logs import LogImages
    tf.random.set_seed(42)
    gpu = gpu.split(':')[-1].rstrip('/')
    print('GPU', gpu)
    print('CPU', cores)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu + ",0"
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
    os.environ["PATH"] +=  os.pathsep + "C:/Program Files/NVIDIA Corporation/Nsight Systems 2023.1.2/target-windows-x64"
    os.environ['ENABLE_PROFILER'] = '1'

    # global lock
    if distributed:
        time.sleep(np.random.random()*10)
        lock.acquire()
        print("LOCKED")
    if data_folder is None:
        data_folder = get_latest_folder(FOLDER)

    print(settings)
    logs_dir = folder.parents[0] / (settings.get("log_dir", "logs") + "/scalars/")
    name = short_name = Path(folder).parts[-1][:13]
    logdir = logs_dir / name
    i = 0
    while logdir.exists():
        name = short_name + f"_{i}"
        logdir = logs_dir / name
        i += 1
    print(name)
    print(f"Writing settings, logdir {logdir}")
    writer = tf.summary.create_file_writer(str(logdir))
    with writer.as_default():
        for key, value in settings.items():
            tf.summary.text(name=key, data=str(value), step=0)
        writer.flush()
    if distributed:
        lock.release()
        print("UNLOCKED")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, profile_batch='100,110')
    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1)
    class GarbageCollectionCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            gc.collect()

    # batch_generator = ArraySequence(data_folder, settings["batch_size"],
    #                                  n_augmentations=settings["n_augmentations"],
    #                                  brightness_range=settings['brightness_range'],
    #                                  poisson=settings["poisson"],
    #                                  subset_fraction=settings["subset_fraction"],
    #                                  t_size=settings['n_timepoints'],
    #                                  validation=False,
    #                                  augment = False) # aug handled by Dataset
    # validation_generator = ArraySequence(data_folder, settings["batch_size"],
    #                                  n_augmentations=settings["n_augmentations"],
    #                                  brightness_range=settings['brightness_range'],
    #                                  poisson=settings["poisson"],
    #                                  validation=True,
    #                                  t_size=settings['n_timepoints'])
    
    # def create_tf_dataset(array_sequence, augment=True, settings=None):
    #     def apply_tf_augmentation_wrapper(x, y):
    #         return tf_apply_augmentation(
    #             x, y,
    #             x_size=128,
    #             y_size=128,
    #             brightness_range=settings["brightness_range"],
    #             poisson=settings["poisson"],
    #             performance=False
    #         )
    #     # num_parallel_calls = 50
    #     if settings is None:
    #         raise ValueError("Settings must be provided to determine the number of timepoints and channels.")
    #     # Define the output signature based on how timepoints are stacked into channels
    #     output_signature = (tf.TensorSpec(shape=(None, None, settings['n_timepoints'] * settings['nb_input_channels']), dtype=tf.float32),
    #                         tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32))
    #     print("dataset from generator")
    #     dataset = tf.data.Dataset.from_generator(lambda: array_sequence, output_signature=output_signature)
    #     dataset = dataset.cache()                        # Cache the batched dataset
    #     dataset = dataset.repeat()                       # Repeat cached data for multiple epochs
    #     dataset = dataset.shuffle(buffer_size=10000)
    #     if augment:
    #         # Apply augmentations to individual samples before batching
    #         dataset = dataset.map(
    #             apply_tf_augmentation_wrapper,
    #             num_parallel_calls=tf.data.experimental.AUTOTUNE
    #         )
    #     # Batch augmented samples
    #     dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)  
    #     dataset = dataset.batch(settings["batch_size"])  # Batch after augmentation
    #     # Ensure correct data types
    #     # dataset = dataset.map(lambda x, y: (tf.cast(x, tf.float32), tf.cast(y, tf.float32)))
        # return dataset

    from deep_events.generator import create_tf_dataset
    data_folder = Path("D:/Users/stepp/Desktop/data_20250105_1122_brightfield_cos7_t0.2_f1_sFalse_mito_events_n753_sFalse")
    train_dataset = create_tf_dataset(data_folder, settings["batch_size"], augment=True, t_size=settings["n_timepoints"])
    validation_dataset = create_tf_dataset(data_folder, settings["batch_size"], augment=False, validation=True, t_size=settings["n_timepoints"])
    for batch in train_dataset.take(1):
        print(batch[0].shape)  # Shape of features
        print(batch[1].shape)  # Shape of labels
    for batch in train_dataset.take(1):
        print(batch[0].shape)  # Shape of features
        print(batch[1].shape)  # Shape of labels
    # images_callback = LogImages(logdir, batch_generator, validation_generator, freq=2)
    
    settings["logdir"] = logdir.name
    settings_folder = str(folder / (name + "_settings.yaml"))
    settings['data_folder'] = data_folder
    print(F"SETTINGS LOCATION: {settings_folder}")
    benedict(settings).to_yaml(filepath = folder / (name + "_settings.yaml"))


    n_tries = 0
    max_tries = 10
    while n_tries < max_tries:

        print('train shape', batch[0].shape)
        # print('val shape', val_batch[0].shape)
        model_generator = get_model_generator(settings['model'])
        model = model_generator(settings, batch[0].shape)

        # steps_per_epoch = np.floor(batch_generator.__len__())
        print(f"NUMBER OF EPOCHS: {settings['epochs']}")
        try:
            # tf.profiler.experimental.start(settings['logdir'])
            history = model.fit(train_dataset,
                                validation_data = validation_dataset,
                                # batch_size = settings["batch_size"],
                                epochs = settings["epochs"]*10,
                                # steps_per_epoch = np.ceil((len(batch_generator) * settings["n_augmentations"]) / settings["batch_size"]),
                                # shuffle=True,
                                verbose = 1,
                                callbacks = [tensorboard_callback,  reduce_lr_callback, early_stopping_callback, GarbageCollectionCallback()],# , images_callback],
                                validation_steps=10,
                                )
            # tf.profiler.experimental.stop()
            n_tries = max_tries
        except Exception as e:
            print("------------------------------ COULD NOT TRAIN -----------------------------------------")
            print(e)
            gpu_index = int(gpu)
            gpu = str((gpu_index + 1) % 6)  # Rotate GPU index (assuming 6 GPUs)
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu
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