import tensorflow as tf
import tensorflow_addons as tfa

class RandomFlip(tf.keras.layers.Layer):
    def call(self, x):
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_flip_up_down(x)
        return x


class RandomRotation(tf.keras.layers.Layer):
    def __init__(self, max_angle=45):
        super(RandomRotation, self).__init__()
        self.max_angle = max_angle

    def call(self, x):
        # Convert degrees to radians for TensorFlow rotation
        radians = tf.random.uniform([], -self.max_angle, self.max_angle) * (3.14159 / 180)
        return tfa.image.rotate(x, radians)


class RandomZoom(tf.keras.layers.Layer):
    def __init__(self, zoom_range=(0.8, 1.2)):
        super(RandomZoom, self).__init__()
        self.zoom_range = zoom_range

    def call(self, x):
        zoom_factor = tf.random.uniform([], self.zoom_range[0], self.zoom_range[1])
        x = tf.image.resize(x, [int(tf.shape(x)[1] * zoom_factor), int(tf.shape(x)[2] * zoom_factor)])
        return tf.image.resize_with_crop_or_pad(x, tf.shape(x)[1], tf.shape(x)[2])  # Return to original size


class RandomContrast(tf.keras.layers.Layer):
    def __init__(self, contrast_range=(0.5, 1.5)):
        super(RandomContrast, self).__init__()
        self.contrast_range = contrast_range

    def call(self, x):
        return tf.image.random_contrast(x, self.contrast_range[0], self.contrast_range[1])


class RandomGaussianNoise(tf.keras.layers.Layer):
    def __init__(self, stddev=0.05):
        super(RandomGaussianNoise, self).__init__()
        self.stddev = stddev

    def call(self, x):
        noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=self.stddev, dtype=tf.float32)
        return x + noise


class RandomBrightness(tf.keras.layers.Layer):
    def __init__(self, brightness_range=(-0.2, 0.2)):
        super(RandomBrightness, self).__init__()
        self.brightness_range = brightness_range

    def call(self, x):
        return tf.image.random_brightness(x, self.brightness_range[1])

def apply_augmentation_v2(self, x, y):
    # Now we define the full augmentation pipeline
    augmentation_pipeline = tf.keras.Sequential([
        RandomFlip(),                # Randomly flip the image horizontally and vertically
        RandomRotation(max_angle=30), # Randomly rotate the image by up to 30 degrees
        # RandomZoom(zoom_range=(0.9, 1.1)), # Apply random zoom between 90% to 110%
        RandomContrast(contrast_range=(0.8, 1.2)), # Random contrast adjustment
        RandomBrightness(brightness_range=(-0.1, 0.1)), # Random brightness shift
        RandomGaussianNoise(stddev=0.05)  # Add Gaussian noise
    ])
    

