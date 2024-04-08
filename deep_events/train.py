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

FOLDER = Path("//lebnas1.epfl.ch/microsc125/deep_events/data/training_data/")
SETTINGS = {"nb_filters": 16,
            "first_conv_size": 12,
            "nb_input_channels": 1,
            "batch_size": 16,
            "epochs": 20,
            "n_augmentations": 10,
            'brightness_range': [0.6, 1],
            "loss": 'binary_crossentropy',
            "poisson": 0}



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
    # os.environ['TF_GPU_ALLOCATOR'] = ""
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
    # folders = [str(x) for x in folders]
    with Pool(min(5, len(folders)), initializer=init_pool, initargs=(l,)) as p:
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
                                     brightness_range=settings['brightness_range'],
                                     poisson=settings["poisson"])
    validation_generator = ArraySequence(latest_folder, settings["batch_size"],
                                     n_augmentations=settings["n_augmentations"],
                                     brightness_range=settings['brightness_range'],
                                     poisson=settings["poisson"],
                                     validation=True)

    images_callback = LogImages(logdir, batch_generator, batch_generator, freq=1)

    time.sleep(1)
    name = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    while Path(latest_folder / (name + "_settings.yaml")).is_file():
        name = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    settings["logdir"] = logdir.name
    settings_folder = str(latest_folder / (name + "_settings.yaml"))
    print(F"SETTINGS LOCATION: {settings_folder}")
    benedict(settings).to_yaml(filepath = latest_folder / (name + "_settings.yaml"))
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
                                    callbacks = [tensorboard_callback, images_callback],
                                    validation_steps=1)
            n_tries = max_tries
        except Exception as e:
            print("------------------------------ COULD NOT TRAIN -----------------------------------------")
            print(e)
            gpu = gpu.replace(gpu[-2], str((int(gpu[-2])+1)%6))
            time.sleep(10)
            print(f"Try next GPU: {gpu}")
        n_tries += 1

    tf.keras.models.save_model(model, latest_folder / (name + "_model.h5"), save_traces=True)
    performance.main(latest_folder / (name + "_model.h5"))


class LogImages(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, image_generator, label_generator, freq=1):
        super(LogImages, self).__init__()
        self.log_dir = log_dir
        self.image_generator = image_generator  # The images you want to log
        self.label_generator = label_generator  # Corresponding labels (optional)
        self.freq = freq  # Frequency (in epochs) at which to log images
        self.image_idxs = np.linspace(0, self.image_generator.num_samples, 10, dtype=int)

    def on_epoch_end(self, epoch, logs=None):
        # Log images every 'freq' epochs
        if epoch % self.freq == 0:
            with tf.summary.create_file_writer(self.log_dir).as_default():
                images = []
                labels = []
                for idx in self.image_idxs:
                    image, label = self.image_generator.__getitem__(idx)
                    images.append(image)
                    labels.append(label)
                predictions = self.model.predict(images[:])

                tf.summary.image("Images", images, step=epoch, max_outputs=10)

                # Optionally, log the labels as images too (if they are images)
                tf.summary.image("Labels", labels, step=epoch, max_outputs=10)
                tf.summary.image("Preds", predictions, step=epoch, max_outputs=10)



if __name__ == "__main__":
    pass
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3" # set the GPU ID
