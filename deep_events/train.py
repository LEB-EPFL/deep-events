from pathlib import Path
import datetime
from multiprocessing import Pool, Lock
import time

from benedict import benedict
import tensorflow as tf
import numpy as np
import os
import random

# set seeds
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

def adjust_tf_dimensions(stack:np.array):
    if len(stack.shape) < 4:
        return np.expand_dims(stack, axis=-1)
    else:
        return np.moveaxis(stack, 1, -1)


from deep_events.training_functions import create_model, create_recurrent_model, create_bottleneck_convlstm_model
from deep_events.generator import ArraySequence
from deep_events.database.convenience import get_latest_folder
from deep_events import performance

FOLDER = Path("//sb-nas1.rcp.epfl.ch/LEB/Scientific_projects/deep_events_WS/data/original_data/training_data/")
SETTINGS = {"nb_filters": 16,
            "first_conv_size": 12,
            "nb_input_channels": 1,
            "batch_size": 32,
            "epochs": 10,
            "n_augmentations": 5,
            'brightness_range': [0.6, 1],
            "loss": 'binary_crossentropy',
            "poisson": 0,
            "subset_fraction": 0.02,
            "initial_learning_rate": 4e-4}



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
    logs_dir = folder.parents[0] / (settings.get("log_dir", "logs") + "/scalars/")
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
    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1)


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

    images_callback = LogImages(logdir, batch_generator, validation_generator, freq=2)
    

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
        print('val shape', validation_generator.__getitem__(0)[1].shape)
        model = create_bottleneck_convlstm_model(settings, batch_generator.__getitem__(0)[0].shape)

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
    


class LogImages(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, image_generator, val_generator, freq=1, overlay_alpha=0.5, nrows=3, ncols=10, downscale=2):
        super().__init__()
        self.log_dir = str(log_dir)
        self.image_generator = image_generator
        self.val_generator = val_generator
        self.freq = freq
        self.nrows = nrows
        self.ncols = ncols
        self.overlay_alpha = overlay_alpha
        self.downscale = downscale
        self.idxs = [
            np.linspace(0, self.image_generator.num_samples - 1, nrows*ncols, dtype=int),
            np.linspace(0, self.val_generator.num_samples - 1, nrows*ncols, dtype=int),
        ]

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.freq == 0:
            with tf.summary.create_file_writer(self.log_dir).as_default():
                for prefix, generator, idxs in zip(['train','eval'], [self.image_generator,self.val_generator], self.idxs):
                    self.log_images(generator, idxs, prefix, epoch)

    def log_images(self, generator, idxs, prefix, epoch):
        ls= []
        for idx in idxs:
            x, y = generator.__getitem__(idx)
            z = self.model.predict(x, verbose=0)
            x, y, z = x[0, :, :, 0], y[0], z[0]
            xi = self._c2c(x)
            yl = self._om(xi, y, z, (255,0,0), self.overlay_alpha)
            ls.append(yl)
        ls = self._make_montage(np.stack(ls, axis=0))
        ls = ls.astype(np.float32)/255.
        tf.summary.image(prefix+"_labels_overlay", ls[None], step=epoch)

    def _c2c(self, img):
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        if img.shape[-1] == 1:
            img = np.concatenate([img, img, img], axis=-1)
        # img = self._downsample(img, self.downscale)
        return self._n2u(img)

    def _n2u(self, img):
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img*255).clip(0,255)
            else:
                img = np.clip(img, 0, 255)
            img = img.astype(np.uint8)
        return img

    def _om(self, orig, mask, pred, color=(255,0,0), alpha=0.5):
        if len(mask.shape) == 3 and mask.shape[-1] > 1:
            mask = np.mean(mask, axis=-1)
        if len(pred.shape) == 3 and pred.shape[-1] > 1:
            pred = np.mean(pred, axis=-1)
        mask = self._n2u(mask)[:, :, 0]
        pred = self._n2u(pred)[:, :, 0]
        
        # Convert mask from [0..255] to [0..2]
        mask_f = mask.astype(np.float32) / 128.0
        pred_f = pred.astype(np.float32) /128.0
        
        orig_f = orig.astype(np.float32)
        cf  = np.array(color, dtype=np.float32)
        cfg = np.array((0,255,0), dtype=np.float32)
        
        # Red overlay for mask
        c = cf * mask_f[..., None]  # shape (H,W,3), each pixel scaled by mask_f
        # Green overlay for pred
        cp = cfg * pred_f[..., None]
        
        # Weighted sum
        out = (1 - alpha)*orig_f + alpha*c + alpha*cp
        out = np.clip(out, 0, 255).astype(np.uint8)
        return out
        # mask = self._n2u(mask)[:, :, 0]
        # pred = self._n2u(pred)[:, :, 0]
        # c = np.zeros_like(orig, dtype=np.float32)
        # cf = np.array(color, dtype=np.float32)
        # r = (mask > 20)
        # if r.max():
        #     c[r] = np.array([cf *x for x in mask[r]])
        # cp = np.zeros_like(orig, dtype=np.float32)
        # colorp = np.array((0, 255, 0), dtype=np.float32)
        # p = (pred > 20)
        # if p.max():
        #     cp[p] = np.array([colorp *x for x in pred[p]])
        # of = orig.astype(np.float32)
        # o = (1.0 - alpha)*of + alpha*c + alpha*cp
        # return np.clip(o, 0, 255).astype(np.uint8)

    def _downsample(self, img, factor):
        return img[::factor, ::factor, :]

    def _make_montage(self, imgs):
        _, h, w, c = imgs.shape
        out = np.zeros((self.nrows*h, self.ncols*w, c), dtype=imgs.dtype)
        for idx, im in enumerate(imgs):
            if idx >= self.nrows*self.ncols: break
            r = idx // self.ncols
            cc = idx % self.ncols
            if idx == 0:
                im[:20, :20, 0] = np.ones((20,20))*255
                im[:20, :20, 1] = np.ones((20,20))*255
            out[r*h:(r+1)*h, cc*w:(cc+1)*w] = im
        return out



if __name__ == "__main__":
    gpu = 'GPU:1/'
    folder = FOLDER / "20241220_1432_brightfield_cos7_n5_f1"

    train(folder, gpu)
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3" # set the GPU ID
