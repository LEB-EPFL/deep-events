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
    
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter

def create_2d_displacement(
    shape: tuple,
    alpha: float = 10.0,
    sigma: float = 3.0,
    random_state: np.random.RandomState = None
):
    """
    Creates a random 2D displacement field (dx, dy) for elastic transforms.

    Args:
        shape (tuple): (H, W) shape for displacement.
        alpha (float): Magnitude scaling for displacement.
        sigma (float): Gaussian filter sigma for smoothing displacement fields.
        random_state (np.random.RandomState): Optional, for reproducibility.

    Returns:
        (dx, dy): 2D arrays of shape (H, W) each.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    H, W = shape

    # Random in range [-1, 1]
    dx = random_state.rand(H, W) * 2 - 1
    dy = random_state.rand(H, W) * 2 - 1

    # Smooth the fields
    dx = gaussian_filter(dx, sigma, mode="reflect") * alpha
    dy = gaussian_filter(dy, sigma, mode="reflect") * alpha

    return dx, dy

def warp_2d_slice(
    slice_2d: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    order: int = 1
) -> np.ndarray:
    """
    Warps one 2D slice with displacement fields dx, dy using map_coordinates.

    Args:
        slice_2d (np.ndarray): shape (H, W) or (H, W, C).
        dx (np.ndarray): shape (H, W) displacement in x-direction.
        dy (np.ndarray): shape (H, W) displacement in y-direction.
        order (int): Interpolation order. 1 for images, 0 for masks.

    Returns:
        np.ndarray: Warped slice, same shape as input slice_2d.
    """
    # If we have (H, W, C), handle each channel separately
    is_multichannel = (slice_2d.ndim == 3 and slice_2d.shape[-1] > 1)

    # Meshgrid for base coordinates
    H, W = slice_2d.shape[:2]
    x_coords, y_coords = np.meshgrid(np.arange(W), np.arange(H))  # note order

    # Flatten the coordinate arrays for map_coordinates
    map_x = (x_coords + dx).flatten()
    map_y = (y_coords + dy).flatten()

    # Bounds check is done with mode='reflect' or 'nearest' in map_coordinates
    if is_multichannel:
        # Warp each channel
        warped_channels = []
        for c in range(slice_2d.shape[-1]):
            channel_img = slice_2d[..., c]
            warped_c = map_coordinates(
                channel_img,
                [map_y, map_x],  # indices is (row_coords, col_coords)
                order=order,
                mode='reflect'
            ).reshape(H, W)
            warped_channels.append(warped_c)
        warped_slice = np.stack(warped_channels, axis=-1)
    else:
        # Single channel
        warped_slice = map_coordinates(
            slice_2d,
            [map_y, map_x],
            order=order,
            mode='reflect'
        ).reshape(H, W)

    return warped_slice

def warp_2d_slices_3d(
    volume_3d: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    order: int = 1
) -> np.ndarray:
    """
    Applies the same (dx, dy) elastic warp to each slice along the last dimension (T).

    Args:
        volume_3d (np.ndarray): shape (H, W, T) or (H, W, T, C).
                                If you have channels, the shape might be (H, W, C, T).
                                Adjust accordingly or reshape first.
        dx, dy (np.ndarray): shape (H, W) displacements.
        order (int): 1 for continuous images, 0 for masks.

    Returns:
        np.ndarray: Warped volume, same shape as volume_3d.
    """
    # We assume shape = (H, W, T) or (H, W, T, C).
    # If you have (H, W, C, T), reorder to (H, W, T, C) first or loop appropriately.

    # Distinguish whether there's a channel dimension
    if volume_3d.ndim == 3:
        # (H, W, T) => no explicit channels
        H, W, T = volume_3d.shape
        warped = np.zeros_like(volume_3d)
        for t in range(T):
            slice_2d = volume_3d[..., t]  # shape (H, W)
            warped_2d = warp_2d_slice(slice_2d, dx, dy, order=order)
            warped[..., t] = warped_2d

    elif volume_3d.ndim == 4:
        # (H, W, T, C)
        H, W, T, C = volume_3d.shape
        warped = np.zeros_like(volume_3d)
        for t in range(T):
            # shape (H, W, C)
            slice_2d = volume_3d[..., t, :]
            warped_2d = warp_2d_slice(slice_2d, dx, dy, order=order)
            warped[..., t, :] = warped_2d
    else:
        raise ValueError("Expected volume_3d to be (H, W, T) or (H, W, T, C).")

    return warped

def elastic_transform_3d_as_2d_slices(
    volume_3d: np.ndarray,
    alpha: float = 10.0,
    sigma: float = 3.0,
    order: int = 1,
    random_state: np.random.RandomState = None
) -> np.ndarray:
    """
    Applies a 2D elastic transform to each slice in the third dimension of volume_3d
    using the same displacement fields, preserving continuity along that dimension.

    Args:
        volume_3d (np.ndarray): shape (H, W, T) or (H, W, T, C).
        alpha (float): magnitude scaling for displacements.
        sigma (float): std dev for Gaussian smoothing.
        order (int): interpolation order (1 for images, 0 for masks).
        random_state (np.random.RandomState): optional, for reproducibility.

    Returns:
        np.ndarray: The warped 3D volume, same shape as volume_3d.
    """
    # Create a single 2D displacement for all slices
    shape_2d = volume_3d.shape[:2]  # (H, W)
    dx, dy = create_2d_displacement(
        shape=shape_2d,
        alpha=alpha,
        sigma=sigma,
        random_state=random_state
    )

    # Apply that displacement to each slice
    warped_3d = warp_2d_slices_3d(volume_3d, dx, dy, order=order)
    return warped_3d, dx, dy