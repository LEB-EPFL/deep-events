import numpy as np
import tensorflow as tf

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
        x_list, y_list = [], []
        for idx in idxs:
            x, y = generator.__getitem__(idx)
            x_list.append(x[0])  
            y_list.append(y[0])
        x_batch = np.stack(x_list, axis=0)
        y_batch = np.stack(y_list, axis=0)

        # Single batch prediction
        z_batch = self.model.predict(x_batch, verbose=0)
        ls = []
        for i in range(len(idxs)):
            x = x_batch[i, ..., 0]
            y = y_batch[i]
            z = z_batch[i]
            # x, y, z = x_i[0, :, :, 0], y_i[0], z_i[0]
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
