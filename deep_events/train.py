from pathlib import Path
import datetime
from multiprocessing import Pool, Lock
import time

from benedict import benedict
import tensorflow as tf
import numpy as np
import os

def adjust_tf_dimensions(stack:np.array):
    if len(stack.shape) < 4:
        return np.expand_dims(stack, axis=-1)
    else:
        return np.moveaxis(stack, 1, -1)


from deep_events.training_functions import create_model
from deep_events.generator import ArraySequence
from deep_events.database.convenience import get_latest_folder
from deep_events import performance

FOLDER = Path("//sb-nas1.rcp.epfl.ch/LEB/Scientific_projects/deep_events_WS/data/original_data/training_data/")
SETTINGS = {"nb_filters": 16,
            "first_conv_size": 12,
            "nb_input_channels": 1,
            "batch_size": 32,
            "epochs": 10,
            "n_augmentations": 30,
            'brightness_range': [0.6, 1],
            "loss": 'binary_crossentropy',
            "poisson": 0,
            "subset_fraction": 0.02,
            "initial_learning_rate": 0.5e-3}



def main(): # pragma: no cover
    tf.keras.backend.clear_session()
    gpus = ['GPU:1/']
    folders = ["20240806_1401_brightfield_cos7_n3_f1"]
    folders = [FOLDER/folder for folder in folders]
    with Pool(3) as p:
        p.starmap(train, zip(folders, gpus))

def distributed_train(folders, gpus, settings=SETTINGS):
    l = Lock()

    if not isinstance(settings, list):
        settings = [settings]*len(folders)
    distributed = [True]*len(folders)
    # os.environ['TF_GPU_ALLOCATOR'] = 0
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
    # folders = [str(x) for x in folders]
    with Pool(min(5, len(folders)), initializer=init_pool, initargs=(l,)) as p:
        p.starmap(train, zip(folders, gpus, settings, distributed))
    time.sleep(30)  # allow for gpu cleanup
    for folder in set(folders):
        performance.performance_for_folder(folder, general_eval=False)
    time.sleep(30)  # allow for gpu cleanup

def init_pool(l: Lock):
    global lock
    lock = l

lock = Lock()

def train(folder: Path = None, gpu = 'GPU:2/', settings: dict = SETTINGS, distributed: bool = False):
    if distributed:
        time.sleep(np.random.random()*10)
        lock.acquire()
        print("LOCKED")
    if folder is None:
        folder = get_latest_folder(FOLDER)


    print(folder)
    print(settings)
    logs_dir = folder.parents[0] / "logs/scalars/"
    name = short_name = Path(folder).parts[-1][:13]
    logdir = logs_dir / name
    i = 0
    while logdir.exists():
        name = short_name + f"_{i}"
        logdir = logs_dir / name
        print(name)
        i += 1
    print(f"Writing settings, logdir {logdir}")
    writer = tf.summary.create_file_writer(str(logdir))
    with writer.as_default():
        for key, value in settings.items():
            tf.summary.text(name=key, data=str(value), step=0)
        writer.flush()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)

    batch_generator = ArraySequence(folder, settings["batch_size"],
                                     n_augmentations=settings["n_augmentations"],
                                     brightness_range=settings['brightness_range'],
                                     poisson=settings["poisson"],
                                     subset_fraction=settings["subset_fraction"])
    validation_generator = ArraySequence(folder, settings["batch_size"],
                                     n_augmentations=settings["n_augmentations"],
                                     brightness_range=settings['brightness_range'],
                                     poisson=settings["poisson"],
                                     validation=True)

    images_callback = LogImages(logdir, batch_generator, validation_generator, freq=1)
    

    # time.sleep(1)
    # name = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    # while Path(folder / (name + "_settings.yaml")).is_file():
    #     name = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    settings["logdir"] = logdir.name
    settings_folder = str(folder / (name + "_settings.yaml"))
    print(F"SETTINGS LOCATION: {settings_folder}")
    benedict(settings).to_yaml(filepath = folder / (name + "_settings.yaml"))
    if distributed:
        lock.release()
        print("UNLOCKED")

    n_tries = 0
    max_tries = 10
    while n_tries < max_tries:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu[-2]
        model = create_model(settings, batch_generator.__getitem__(0)[0].shape)

        steps_per_epoch = np.floor(batch_generator.__len__())
        print(f"NUMBER OF EPOCHS: {settings['epochs']}" )
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
                                    callbacks = [tensorboard_callback, images_callback, reduce_lr_callback],
                                    validation_steps=20)
            n_tries = max_tries
        except Exception as e:
            print("------------------------------ COULD NOT TRAIN -----------------------------------------")
            print(e)
            gpu = gpu.replace(gpu[-2], str((int(gpu[-2])+1)%6))
            time.sleep(10)
            print(f"Try next GPU: {gpu}")
        n_tries += 1

    tf.keras.models.save_model(model, folder / (name + "_model.h5"), save_traces=True)
    


class LogImages(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, image_generator, val_generator, freq=1):
        super(LogImages, self).__init__()
        self.log_dir = str(log_dir)
        self.image_generator = image_generator  # The images you want to log
        self.val_generator = val_generator  # Corresponding labels (optional)
        self.freq = freq  # Frequency (in epochs) at which to log images
        self.idxs = [np.linspace(0, self.image_generator.num_samples, 10, dtype=int),
                    np.linspace(0, self.val_generator.num_samples, 10, dtype=int),]

    def on_epoch_end(self, epoch, logs=None):
        # Log images every 'freq' epochs
        if epoch % self.freq == 0:
            with tf.summary.create_file_writer(self.log_dir).as_default():
                for prefix, generator, idxs in zip(['train', 'eval'],
                                             [self.image_generator, self.val_generator],
                                             self.idxs):
                    self.log_images(generator, idxs, prefix, epoch)

    def log_images(self, generator, idxs, prefix, epoch):
        images = []
        labels = []
        predictions = []
        for idx in idxs:
            image, label = generator.__getitem__(idx)
            pred = self.model.predict(image, verbose=0)
            image, label, pred = image[0], label[0], pred[0]
            predictions.append(pred)
            image = image[..., 0]
            image = np.expand_dims(image, -1)
            images.append(image)
            labels.append(label)
        # images, labels, predictions = np.array(images), np.array(labels), np.array(predictions)
        tf.summary.image(prefix + "_images", images, step=epoch, max_outputs=10) # Optionally, log the labels as images too (if they are images)
        tf.summary.image(prefix + "_labels", labels, step=epoch, max_outputs=10)
        tf.summary.image(prefix + "_preds", predictions, step=epoch, max_outputs=10)


if __name__ == "__main__":
    gpu = 'GPU:1/'
    folder = FOLDER / "20240806_1415_brightfield_cos7_n3_f1"

    train(folder, gpu)
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3" # set the GPU ID
