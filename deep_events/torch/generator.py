import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import random
from typing import Tuple, List, Optional, Union, Dict
import tifffile
import os
import logging

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
os.environ['PYTHONHASHSEED'] = '42'

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def apply_augmentation(sequence, x, y, performance=False):
    """
    Apply GPU-accelerated augmentation to image and mask.
    
    Parameters:
    -----------
    sequence : ArraySequence
        The ArraySequence instance
    x : torch.Tensor
        Input image or image sequence
    y : torch.Tensor
        Target mask
    performance : bool
        Whether this is for performance testing
        
    Returns:
    --------
    Tuple[torch.Tensor, torch.Tensor]
        Augmented image and mask
    """
    # Get device from the input tensors
    device = x.device
    
    # No augmentation for performance testing
    if performance:
        return x, y
    
    # Random flip - this operation is safe for any dimension
    if torch.rand(1, device=device).item() > 0.5:
        # Flip across the width dimension (last dimension)
        x = torch.flip(x, dims=[-1])
        y = torch.flip(y, dims=[-1])
    
    # Random 90-degree rotation
    k_rot = torch.randint(0, 4, (1,), device=device).item()  # 0, 1, 2, or 3 (x90 degrees)
    if k_rot > 0:
        # Only attempt rotation if the last two dimensions are equal (square)
        if x.shape[-1] == x.shape[-2]:  # Check if H=W for the spatial dimensions
            if len(x.shape) == 4:  # (Batch, Time/Channel, Height, Width)
                for b in range(x.shape[0]):
                    for c in range(x.shape[1]):
                        x[b, c] = torch.rot90(x[b, c], k=k_rot, dims=[0, 1])
            elif len(x.shape) == 3:  # (Time/Channel, Height, Width)
                for c in range(x.shape[0]):
                    x[c] = torch.rot90(x[c], k=k_rot, dims=[0, 1])
            else:  # (Height, Width)
                x = torch.rot90(x, k=k_rot, dims=[0, 1])
                
            # Apply same rotation to target
            if len(y.shape) == 4:
                for b in range(y.shape[0]):
                    for c in range(y.shape[1]):
                        y[b, c] = torch.rot90(y[b, c], k=k_rot, dims=[0, 1])
            elif len(y.shape) == 3:
                for c in range(y.shape[0]):
                    y[c] = torch.rot90(y[c], k=k_rot, dims=[0, 1])
            elif len(y.shape) == 2:
                y = torch.rot90(y, k=k_rot, dims=[0, 1])
    
    # CENTER CROPPING - Add here to match TensorFlow implementation
    # Define crop size (default 128 like in TensorFlow or make configurable)
    x_size = getattr(sequence, 'crop_size', 128)  # Use sequence.crop_size if available, else 128
    
    # Get dimensions based on tensor shape
    if len(x.shape) == 4:  # (Batch, Time/Channel, H, W)
        _, _, H, W = x.shape
    elif len(x.shape) == 3:  # (Time/Channel, H, W)
        _, H, W = x.shape
    else:  # (H, W)
        H, W = x.shape
    
    # Always resize for consistent dimensions - this is crucial for batching
    if H != x_size or W != x_size:
        # If the image is larger than crop_size, center crop
        if H >= x_size and W >= x_size:
            # Calculate crop positions (center crop)
            crop_pos_h = (H - x_size) // 2
            crop_pos_w = (W - x_size) // 2
            
            # Apply cropping based on tensor dimensions
            if len(x.shape) == 4:  # (Batch, Time/Channel, H, W)
                x = x[:, :, crop_pos_h:crop_pos_h+x_size, crop_pos_w:crop_pos_w+x_size]
                if len(y.shape) == 4:
                    y = y[:, :, crop_pos_h:crop_pos_h+x_size, crop_pos_w:crop_pos_w+x_size]
                elif len(y.shape) == 3:
                    y = y[:, crop_pos_h:crop_pos_h+x_size, crop_pos_w:crop_pos_w+x_size]
                else:  # y is (H, W)
                    y = y[crop_pos_h:crop_pos_h+x_size, crop_pos_w:crop_pos_w+x_size]
            elif len(x.shape) == 3:  # (Time/Channel, H, W)
                x = x[:, crop_pos_h:crop_pos_h+x_size, crop_pos_w:crop_pos_w+x_size]
                if len(y.shape) == 3:
                    y = y[:, crop_pos_h:crop_pos_h+x_size, crop_pos_w:crop_pos_w+x_size]
                else:  # y is (H, W)
                    y = y[crop_pos_h:crop_pos_h+x_size, crop_pos_w:crop_pos_w+x_size]
            else:  # (H, W)
                x = x[crop_pos_h:crop_pos_h+x_size, crop_pos_w:crop_pos_w+x_size]
                y = y[crop_pos_h:crop_pos_h+x_size, crop_pos_w:crop_pos_w+x_size]
        else:
            # If the image is smaller, use interpolation to resize
            import torch.nn.functional as F
            
            if len(x.shape) == 4:  # (Batch, Time/Channel, H, W)
                x_resized = torch.zeros((x.shape[0], x.shape[1], x_size, x_size), device=x.device, dtype=x.dtype)
                for b in range(x.shape[0]):
                    for c in range(x.shape[1]):
                        # Use bilinear interpolation for input images
                        x_resized[b, c] = F.interpolate(
                            x[b, c].unsqueeze(0).unsqueeze(0),  # Add batch & channel dims
                            size=(x_size, x_size),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0).squeeze(0)  # Remove added dims
                x = x_resized
                
                # Resize target with nearest neighbor to preserve class values
                if len(y.shape) == 4:
                    y_resized = torch.zeros((y.shape[0], y.shape[1], x_size, x_size), device=y.device, dtype=y.dtype)
                    for b in range(y.shape[0]):
                        for c in range(y.shape[1]):
                            y_resized[b, c] = F.interpolate(
                                y[b, c].unsqueeze(0).unsqueeze(0),
                                size=(x_size, x_size),
                                mode='nearest'  # Use nearest for masks
                            ).squeeze(0).squeeze(0)
                    y = y_resized
                elif len(y.shape) == 3:
                    y = F.interpolate(
                        y.unsqueeze(0),  # Add batch dim
                        size=(x_size, x_size),
                        mode='nearest'
                    ).squeeze(0)
                else:  # y is (H, W)
                    y = F.interpolate(
                        y.unsqueeze(0).unsqueeze(0),
                        size=(x_size, x_size),
                        mode='nearest'
                    ).squeeze(0).squeeze(0)
            
            elif len(x.shape) == 3:  # (Time/Channel, H, W)
                x_resized = torch.zeros((x.shape[0], x_size, x_size), device=x.device, dtype=x.dtype)
                for c in range(x.shape[0]):
                    x_resized[c] = F.interpolate(
                        x[c].unsqueeze(0).unsqueeze(0),
                        size=(x_size, x_size),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0).squeeze(0)
                x = x_resized
                
                if len(y.shape) == 3:
                    y_resized = torch.zeros((y.shape[0], x_size, x_size), device=y.device, dtype=y.dtype)
                    for c in range(y.shape[0]):
                        y_resized[c] = F.interpolate(
                            y[c].unsqueeze(0).unsqueeze(0),
                            size=(x_size, x_size),
                            mode='nearest'
                        ).squeeze(0).squeeze(0)
                    y = y_resized
                else:  # y is (H, W)
                    y = F.interpolate(
                        y.unsqueeze(0).unsqueeze(0),
                        size=(x_size, x_size),
                        mode='nearest'
                    ).squeeze(0).squeeze(0)
            
            else:  # (H, W)
                x = F.interpolate(
                    x.unsqueeze(0).unsqueeze(0),
                    size=(x_size, x_size),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).squeeze(0)
                
                y = F.interpolate(
                    y.unsqueeze(0).unsqueeze(0),
                    size=(x_size, x_size),
                    mode='nearest'
                ).squeeze(0).squeeze(0)
    
    # Normalize to [0, 1] range
    # Handle different dimension arrangements
    if len(x.shape) == 4:  # (Batch, Time/Channel, H, W)
        for b in range(x.shape[0]):
            for c in range(x.shape[1]):
                if x[b, c].max() > 0:
                    x[b, c] = x[b, c] / x[b, c].max()
    elif len(x.shape) == 3:  # (Time/Channel, H, W)
        for c in range(x.shape[0]):
            if x[c].max() > 0:
                x[c] = x[c] / x[c].max()
    else:  # (H, W)
        if x.max() > 0:
            x = x / x.max()
    
    # Add Poisson noise
    if sequence.poisson > 0:
        intensity_scale = 100 / sequence.poisson
        gaussian_std = 0.01 * sequence.poisson
        
        # Apply noise based on tensor dimensions
        if len(x.shape) == 4:  # (Batch, Time/Channel, H, W)
            for b in range(x.shape[0]):
                for c in range(x.shape[1]):
                    poisson_noisy = torch.poisson(x[b, c] * intensity_scale) / intensity_scale
                    gaussian_noise = torch.randn_like(x[b, c], device=device) * gaussian_std
                    x[b, c] = torch.clamp(poisson_noisy + gaussian_noise, 0, 1)
        elif len(x.shape) == 3:  # (Time/Channel, H, W)
            for c in range(x.shape[0]):
                poisson_noisy = torch.poisson(x[c] * intensity_scale) / intensity_scale
                gaussian_noise = torch.randn_like(x[c], device=device) * gaussian_std
                x[c] = torch.clamp(poisson_noisy + gaussian_noise, 0, 1)
        else:  # (H, W)
            poisson_noisy = torch.poisson(x * intensity_scale) / intensity_scale
            gaussian_noise = torch.randn_like(x, device=device) * gaussian_std
            x = torch.clamp(poisson_noisy + gaussian_noise, 0, 1)

    # Random brightness adjustment
    if sequence.brightness_range is not None:
        brightness_factor = torch.empty(1, device=device).uniform_(
            sequence.brightness_range[0], 
            sequence.brightness_range[1]
        )
        x = x * brightness_factor
    
    # Add gamma correction
    gamma = torch.empty(1, device=device).uniform_(0.8, 1.2)
    x = x.pow(gamma)
    
    # Add random intensity shift
    shift_val = torch.empty(1, device=device).uniform_(-0.1, 0.1)
    x = torch.clamp(x + shift_val, 0, 1)
    
    # Apply elastic transformation
    if not performance and hasattr(sequence, 'elastic_transform') and sequence.elastic_transform:
        from deep_events.torch.augmentation import elastic_transform_3d_as_2d_slices, warp_2d_slice
        
        # Apply the transform without excessive dimension checking
        if len(x.shape) == 3:  # (Time, H, W)
            x, dx, dy = elastic_transform_3d_as_2d_slices(x, alpha=10.0, sigma=3.0, order=1)
            # Apply same transform to mask
            y = warp_2d_slice(y, dx, dy, order=0)
        elif len(x.shape) == 2:  # (H, W)
            from deep_events.torch.augmentation import elastic_transform_2d
            x, dx, dy = elastic_transform_2d(x, alpha=10.0, sigma=3.0, order=1)
            y = warp_2d_slice(y, dx, dy, order=0)
    
    return x, y


class ArraySequence(Dataset):
    """
    PyTorch Dataset for loading and augmenting image sequences for segmentation.
    """
    
    def __init__(self, 
                 data_dir: Path,
                 batch_size: int,
                 augment: bool = True,
                 n_augmentations: int = 10,
                 brightness_range: List[float] = [0.9, 1],
                 poisson: float = 0.0,
                 subset_fraction: float = 1.0,
                 validation: bool = False,
                 t_size: int = 1,
                 last_frame: bool = False,
                 performance: bool = False,
                 device: Optional[torch.device] = None,
                 elastic_transform: bool = True):
        """
        Initialize the dataset.
        
        Parameters:
        -----------
        data_dir : Path
            Directory containing the data
        batch_size : int
            Batch size
        augment : bool
            Whether to apply augmentation
        n_augmentations : int
            Number of augmentations per original image
        brightness_range : List[float]
            Range for brightness augmentation [min, max]
        poisson : float
            Strength of Poisson noise (0 = no noise)
        subset_fraction : float
            Fraction of data to use (1.0 = all data)
        validation : bool
            Whether this is a validation set
        t_size : int
            Number of time points to include
        last_frame : bool or int
            Whether to use only the last frame or a specific frame index
        performance : bool
            Whether this is for performance testing
        device : torch.device
            Device to load data to (if None, will use CPU)
        elastic_transform : bool
            Whether to apply elastic transformation
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.augment = augment
        self.n_augmentations = n_augmentations
        self.brightness_range = brightness_range
        self.poisson = poisson
        self.subset_fraction = subset_fraction if not validation else 1.0
        self.validation = validation
        self.t_size = t_size
        self.last_frame = last_frame
        self.performance = performance
        self.device = device
        self.elastic_transform = elastic_transform

        if self.t_size == 1:
            self.last_frame = True
        
        # Load data from TIFF files
        if validation:
            self.images_file = data_dir / "eval_images_00.tif"
            self.gt_file = data_dir / "eval_gt_00.tif"
        else:
            self.images_file = data_dir / "train_images_00.tif"
            self.gt_file = data_dir / "train_gt_00.tif"
            
        # Load data
        with tifffile.TiffFile(self.images_file) as tif_input, tifffile.TiffFile(self.gt_file) as tif_gt:
            self.images_array = tif_input.asarray()
            self.gt_array = tif_gt.asarray()
            
        # Log data shape
        logger.info(f"Original SHAPE in Sequence: {self.images_array.shape}, validation: {validation}")
        
        # Determine the number of samples
        self.num_samples = self.images_array.shape[0]
        
        # Simplify dimension ordering - just convert to PyTorch format - channels first
        # PyTorch expects: (N, C, H, W) for batches or (C, H, W) for individual samples
        if len(self.images_array.shape) > 3:
            # Move time/channel dimension to position 1 (PyTorch convention)
            self.images_array = np.moveaxis(self.images_array, 1, 1)
            if self.gt_array.ndim > 3 and self.gt_array.shape[1] > 1:
                self.gt_array = np.moveaxis(self.gt_array, 1, 1)
        
        # Log final data shape
        logger.info(f"Final SHAPE OF THE DATA: {self.images_array.shape}")
        
        # Initialize indices
        self.on_epoch_end()
    
    def on_epoch_end(self):
        """Update indices after each epoch."""
        total_samples = self.num_samples
        subset_size = max(1, int(total_samples * self.subset_fraction))
        self.indices = np.random.choice(total_samples, subset_size, replace=False) if not self.validation else np.arange(total_samples)
        
        # For PyTorch, we'll use a synthetic index based on original index + augmentation
        self.full_indices = []
        for idx in self.indices:
            # Original sample
            self.full_indices.append((idx, 0))
            # Augmented samples if augmentation is enabled
            if self.augment and not self.validation:
                for aug_idx in range(1, self.n_augmentations):
                    self.full_indices.append((idx, aug_idx))
    
    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.full_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset with memory optimization.
        
        Parameters:
        -----------
        idx : int
            Index
                
        Returns:
        --------
        Tuple[torch.Tensor, torch.Tensor]
            Input image(s) and target mask
        """
        # Get original index and augmentation index
        if idx >= len(self.full_indices):
            idx = idx % len(self.full_indices)
        
        orig_idx, aug_idx = self.full_indices[idx]
        
        # Load original data (still as numpy arrays)
        x = self.images_array[orig_idx].copy()
        y = self.gt_array[orig_idx].copy()
        
        # Store original target for access to all frames
        y_all = y

        # Temporal subsample
        if self.last_frame is True:
            last_frame = -1
        elif self.last_frame is False:
            if self.validation:
                last_frame = -1
            else:
                last_frame = np.random.randint(-5, 0)
        else:
            # If last_frame is a specific index
            last_frame = self.last_frame
        
        # Memory optimization: Only extract the last frame
        # This is different from before where we extracted all frames in x_frames range
        if len(x.shape) == 3:  # (Time, H, W)
            # Convert negative index to positive
            if last_frame < 0:
                actual_idx = x.shape[0] + last_frame
            else:
                actual_idx = last_frame
            
            # Ensure index is within bounds
            actual_idx = max(0, min(actual_idx, x.shape[0] - 1))
            
            # Select just the last frame (for target prediction)
            last_frame_x = x[actual_idx]
            
            # Memory optimization: Only keep the frames we need
            # For backward compatibility, reshape to (t_size, H, W) even though we only use the last frame
            if self.t_size > 1:
                # Start from the last frame and go backwards to get t_size frames
                start_idx = max(0, actual_idx - self.t_size + 1)
                selected_indices = list(range(start_idx, actual_idx + 1))
                
                # If we don't have enough frames, pad with copies of the first frame
                if len(selected_indices) < self.t_size:
                    padding = [selected_indices[0]] * (self.t_size - len(selected_indices))
                    selected_indices = padding + selected_indices
                
                # Select the frames (limited to t_size)
                selected_indices = selected_indices[-self.t_size:]
                x = x[selected_indices]
            else:
                # For t_size=1, just keep the last frame
                x = np.expand_dims(last_frame_x, axis=0)
        
        # Get the target mask from the last frame
        # Handle different dimension arrangements
        if len(y_all.shape) == 3:  # (Channels/Time, H, W)
            y = y_all[last_frame]
            
            # Ensure we have a mask with some positive pixels if available
            j = 1
            max_idx = y_all.shape[0] - 1
            while j < 5 and y_all.max() > 0.5 and y.max() < 0.5 and last_frame + j <= max_idx:
                y = y_all[last_frame + j]
                j += 1
        elif len(y_all.shape) == 2:  # (H, W)
            y = y_all  # Just use the single frame
        
        # Convert to PyTorch tensors with memory optimization
        x_tensor = torch.from_numpy(x.copy()).float()
        y_tensor = torch.from_numpy(y.copy()).float()
        
        # Ensure y_tensor has shape (C, H, W) with a channel dimension
        if y_tensor.dim() == 2:  # If shape is (H, W)
            y_tensor = y_tensor.unsqueeze(0)  # Add channel dimension to make it (1, H, W)
        
        # Apply augmentation if needed and not validation
        if (self.augment and not self.validation and aug_idx > 0) or self.performance:
            x_tensor, y_tensor = apply_augmentation(self, x_tensor, y_tensor, performance=self.performance)
                
        return x_tensor, y_tensor


class DataLoader:
    """
    A wrapper around DataLoader that integrates with our ArraySequence dataset.
    """
    
    def __init__(self, dataset, batch_size=None, shuffle=True, num_workers=4,
                 pin_memory=None, device=None, drop_last=False, collate_fn=None):
        """
        Initialize the data loader.
        
        Parameters:
        -----------
        dataset : ArraySequence
            The dataset to load from
        batch_size : int, optional
            Batch size (if None, use dataset's batch_size)
        shuffle : bool
            Whether to shuffle the data
        num_workers : int
            Number of worker processes for loading
        pin_memory : bool, optional
            Whether to pin memory for faster GPU transfer.
            If None (default), will be set based on device location.
        device : torch.device
            Device to load data to
        drop_last : bool
            Whether to drop the last incomplete batch
        """
        from torch.utils.data import DataLoader as TorchDataLoader
        
        self.dataset = dataset
        self.batch_size = batch_size if batch_size is not None else dataset.batch_size
        self.device = device
        self.drop_last = drop_last
        
        # If device is provided to the dataset directly, disable pin_memory
        # as we're already moving tensors to the device in __getitem__
        use_pin_memory = False
        if pin_memory is None:
            # Only use pin_memory if tensors stay on CPU in the dataset
            use_pin_memory = (device is not None) and (dataset.device is None)
        else:
            use_pin_memory = pin_memory
            
        # Create standard PyTorch DataLoader
        self.dataloader = TorchDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=use_pin_memory,  # Only pin if tensors are on CPU in dataset
            drop_last=drop_last,
            persistent_workers=num_workers > 0,
            collate_fn=collate_fn
        )
        
        # Set up iterator
        self.iterator = None
    
    def __iter__(self):
        """Return iterator."""
        self.iterator = iter(self.dataloader)
        return self
    
    def __next__(self):
        """Get next batch and move to device."""
        batch = next(self.iterator)
        # Only move to device here if not already moved in dataset
        if self.device is not None and self.dataset.device is None:
            batch = [t.to(self.device, non_blocking=True) for t in batch]
        return batch
    
    def __len__(self):
        """Return the number of batches."""
        return len(self.dataloader)


class DataLoaderWithEpochEnd:
    """
    Wrapper for DataLoader that calls on_epoch_end after each epoch.
    """
    
    def __init__(self, dataset, batch_size=None, shuffle=True, num_workers=4,
                 pin_memory=None, device=None, collate_fn=None):
        """
        Initialize the wrapper.
        
        Parameters:
        -----------
        dataset : ArraySequence
            The dataset to load from
        batch_size : int, optional
            Batch size (if None, use dataset's batch_size)
        shuffle : bool
            Whether to shuffle the data
        num_workers : int
            Number of worker processes
        pin_memory : bool, optional
            Whether to pin memory (if None, determined automatically)
        device : torch.device
            Device to load data to
        """
        self.dataset = dataset
        self.dataloader = DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            device=device,
            collate_fn=collate_fn
        )
    
    def __iter__(self):
        """Return dataloader iterator."""
        return iter(self.dataloader)
    
    def __len__(self):
        """Return the number of batches."""
        return len(self.dataloader)
    
    def on_epoch_end(self):
        """Call on_epoch_end on the dataset."""
        self.dataset.on_epoch_end()