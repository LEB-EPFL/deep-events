from pathlib import Path
import datetime
import time
import logging
import os
import shutil
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast
from torch.cuda.amp import GradScaler

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
os.environ['PYTHONHASHSEED'] = '42'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Default settings
FOLDER = Path("//sb-nas1.rcp.epfl.ch/LEB/Scientific_projects/deep_events_WS/data/original_data/training_data/")
SETTINGS = {
    "nb_filters": 16,
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
    "n_timepoints": 3,
    "model": "bottleneck_lstm"
}

class TensorboardLogger:
    """
    Helper class for TensorBoard logging with debugging and explicit flushing.
    """
    def __init__(self, log_dir):
        """Initialize TensorBoard logger with proper directory creation."""
        # Convert to Path object if string
        if isinstance(log_dir, str):
            log_dir = Path(log_dir)
            
        # Ensure directory exists
        os.makedirs(log_dir, exist_ok=True)
        
        # Log the exact path being used
        logger.info(f"Initializing TensorBoard logger at: {os.path.abspath(log_dir)}")
        
        # Create SummaryWriter
        self.writer = SummaryWriter(log_dir=str(log_dir))
        
        # Test write to ensure it's working
        self.writer.add_text("initialization", f"TensorBoard initialized at {datetime.datetime.now()}", 0)
        self.writer.flush()
        
    def add_scalar(self, tag, value, step):
        """Add scalar value with explicit flush."""
        self.writer.add_scalar(tag, value, step)
        # Periodic flush to ensure data is written
        if step % 5 == 0:
            self.writer.flush()
            
    def add_scalars(self, main_tag, tag_scalar_dict, step):
        """Add multiple scalars with explicit flush."""
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)
        if step % 5 == 0:
            self.writer.flush()
    
    def add_image(self, tag, img_tensor, step):
        """Add image with explicit flush."""
        self.writer.add_image(tag, img_tensor, step)
        self.writer.flush()  # Always flush images
        
    def add_images(self, tag, img_tensor, step):
        """Add images with explicit flush."""
        self.writer.add_images(tag, img_tensor, step)
        self.writer.flush()  # Always flush images
        
    def add_text(self, tag, text_string, step):
        """Add text with explicit flush."""
        self.writer.add_text(tag, text_string, step)
        self.writer.flush()  # Always flush text
    
    def close(self):
        """Close the writer with final flush."""
        self.writer.flush()  # Ensure final flush
        self.writer.close()

def save_dict_to_yaml(d: Dict, filepath: Path):
    """Save dictionary to YAML file in a format compatible with PyYAML safe_load."""
    import yaml
    
    # Convert Benedict dict to regular dict if needed
    if hasattr(d, 'dict') and callable(getattr(d, 'dict')):
        d = d.dict()
    
    # Ensure all values are YAML-serializable
    clean_dict = {}
    for key, value in d.items():
        # Convert Path objects to strings
        if isinstance(value, Path):
            clean_dict[key] = str(value)
        # Convert numpy arrays to lists
        elif isinstance(value, np.ndarray):
            clean_dict[key] = value.tolist()
        # Handle nested dictionaries
        elif isinstance(value, dict):
            clean_dict[key] = save_dict_to_yaml(value, None)
        else:
            clean_dict[key] = value
    
    # If filepath is None, return the cleaned dict instead of saving
    if filepath is None:
        return clean_dict
    
    with open(filepath, 'w') as f:
        yaml.dump(clean_dict, f, default_flow_style=False)

def save_model(model, optimizer, scaler, epoch, settings, val_loss, folder, name, is_best=False, is_final=False):
    """
    Save model with mixed precision support.
    
    Parameters:
    -----------
    model : nn.Module
        The trained model
    optimizer : torch.optim
        Optimizer state
    scaler : GradScaler
        Mixed precision scaler state
    epoch : int
        Current epoch
    settings : dict
        Training settings
    val_loss : float
        Validation loss
    folder : Path
        Folder to save to
    is_best : bool
        Whether this is the best model so far
    is_final : bool
        Whether this is the final model
    """
    # Determine filename suffix
    suffix = "_best" if is_best else "_final" if is_final else f"_e{epoch}"
    model_path = folder / f"{name}_model.pt"
    
    # Clean settings for serialization
    clean_settings = {}
    for key, value in settings.items():
        if isinstance(value, Path):
            clean_settings[key] = str(value)
        elif isinstance(value, np.ndarray):
            clean_settings[key] = value.tolist()
        else:
            clean_settings[key] = value
    
    # Save model with all states
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'epoch': epoch,
        'settings': clean_settings,
        'val_loss': val_loss,
    }, model_path)
    
    return model_path

def custom_collate_fn(batch):
    """
    Custom collate function to handle inconsistent tensor sizes.
    This should be passed to DataLoader's collate_fn parameter.
    
    Parameters:
    -----------
    batch : list
        List of (data, target) tuples from a Dataset
        
    Returns:
    --------
    tuple
        (batched_data, batched_targets) with consistent sizes
    """
    # Extract data and target tensors from the batch
    data, targets = zip(*batch)
    
    # Check if we have tensors with consistent dimensions
    data_shapes = set(x.shape for x in data)
    target_shapes = set(y.shape for y in targets)
    
    # If dimensions are already consistent, use default stacking
    if len(data_shapes) == 1 and len(target_shapes) == 1:
        return torch.stack(data, 0), torch.stack(targets, 0)
    
    # If dimensions are inconsistent, resize to a common size
    # First determine the common size to use (max height/width or median)
    if len(data_shapes) > 1:
        # Get the most common shape (most tensors already have this shape)
        from collections import Counter
        data_shape_counter = Counter(tuple(x.shape) for x in data)
        most_common_shape = data_shape_counter.most_common(1)[0][0]
        
        # Extract target dimensions
        if len(most_common_shape) == 4:  # (Batch, Channel, H, W)
            C, H, W = most_common_shape[1], most_common_shape[2], most_common_shape[3]
        elif len(most_common_shape) == 3:  # (Channel, H, W)
            C, H, W = most_common_shape[0], most_common_shape[1], most_common_shape[2]
        
        # Resize all tensors to the most common size
        resized_data = []
        for x in data:
            if x.shape != most_common_shape:
                # Handle different dimension patterns
                if len(x.shape) == 4:  # (Batch, Channel, H, W)
                    x_resized = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
                elif len(x.shape) == 3:  # (Channel, H, W)
                    if x.shape[0] != C:  # Different channel count
                        # This is an edge case - you might need more complex handling
                        # For now, we'll just use what we have and hope channels are compatible
                        pass
                    x_resized = F.interpolate(x.unsqueeze(0), size=(H, W), mode='bilinear', 
                                              align_corners=False).squeeze(0)
                resized_data.append(x_resized)
            else:
                resized_data.append(x)
        
        # Similarly resize targets, but use nearest neighbor for masks
        resized_targets = []
        for y in targets:
            if y.shape != most_common_shape:
                if len(y.shape) == 3:  # (Channel, H, W)
                    y_resized = F.interpolate(y.unsqueeze(0), size=(H, W), mode='nearest').squeeze(0)
                elif len(y.shape) == 4:  # (Batch, Channel, H, W)
                    y_resized = F.interpolate(y, size=(H, W), mode='nearest')
                resized_targets.append(y_resized)
            else:
                resized_targets.append(y)
        
        # Stack the resized tensors
        return torch.stack(resized_data, 0), torch.stack(resized_targets, 0)
    
    # Fallback case - should not happen if the above logic is robust
    return torch.stack(data, 0), torch.stack(targets, 0)

def distributed_train(data_folders, folders, devices, settings=SETTINGS):
    """
    Train multiple models in parallel.
    
    Parameters:
    -----------
    data_folders : List[Path]
        List of folders containing training data
    folders : List[Path]
        List of folders where to save the models
    devices : List[int]
        List of GPU device IDs to use
    settings : Dict or List[Dict]
        Settings for training, can be a single dict for all models or a list of dicts
    """
    if not isinstance(settings, list):
        settings = [settings] * len(data_folders)
    
    # Calculate workers per model
    workers_per_model = 5
    
    # Set up CUDA for better performance
    torch.backends.cudnn.benchmark = True
    
    # Convert Path objects to strings in all settings dicts
    for i in range(len(settings)):
        # Deep copy the settings to avoid modifying the original
        current_settings = settings[i].copy() if isinstance(settings[i], dict) else settings.copy()
        
        # Add workers_per_model to settings
        current_settings['dataloader_workers'] = workers_per_model
        
        # Make data_folder serializable
        if 'data_folder' in current_settings and isinstance(current_settings['data_folder'], Path):
            current_settings['data_folder'] = str(current_settings['data_folder'])
        
        # Update settings
        settings[i] = current_settings
    
    # Pass the worker count to each training process
    with ThreadPoolExecutor(max_workers=len(data_folders)) as executor:
        futures = []
        for i in range(len(data_folders)):
            futures.append(executor.submit(
                train, data_folders[i], folders[i], devices[i], settings[i], True
            ))
        
        # Wait for all futures to complete
        for future in futures:
            future.result()

    # Evaluate performance for each model if the module exists
    for folder in set(folders):
        from deep_events.torch import performance
        performance.performance_for_folder(folder, general_eval=False)


def train(data_folder: Optional[Path] = None, folder: Optional[Path] = None, 
          device_id: int = 0, settings: Dict = SETTINGS, distributed: bool = False):
    """
    Optimized training function for better GPU utilization.
    """
    if settings is None:
        settings = {}
    
    # Set device
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Free memory from any previous runs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # Set performance-focused CUDA options
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
        torch.backends.cudnn.deterministic = False  # Sacrifice determinism for speed
        torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere+ GPUs
        torch.backends.cudnn.allow_tf32 = True  # Allow TF32 for cuDNN
    
    # Initialize data generators
    try:
        from deep_events.torch.generator import ArraySequence
    except ImportError:
        logger.error("Could not import ArraySequence, please ensure deep_events is installed correctly")
        raise
    
   
    # Optimize number of workers based on CPU count
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    # PyTorch suggests 4 workers per GPU is a good starting point
    optimal_workers = min(cpu_count, 8)  # Don't exceed 8 workers to avoid overwhelming the system
    settings['dataloader_workers'] = settings.get('dataloader_workers', optimal_workers)
    logger.info(f"Using {settings['dataloader_workers']} DataLoader workers")
    
    # Initialize datasets with optimized settings
    train_dataset = ArraySequence(
        data_folder, 
        settings["batch_size"],
        n_augmentations=settings["n_augmentations"],
        brightness_range=settings['brightness_range'],
        poisson=settings["poisson"],
        subset_fraction=settings["subset_fraction"],
        t_size=settings['n_timepoints'],
        device=None  # Keep data on CPU for proper pin_memory usage
    )
    
    val_dataset = ArraySequence(
        data_folder, 
        settings["batch_size"],
        brightness_range=settings['brightness_range'],
        poisson=settings["poisson"],
        validation=True,
        t_size=settings['n_timepoints'],
        device=None  # Keep data on CPU for proper pin_memory usage
    )
    
    # Create data loaders with optimized settings
    train_loader = DataLoader(
        train_dataset, 
        batch_size=settings["batch_size"],
        shuffle=True,
        num_workers=settings['dataloader_workers'],
        collate_fn=custom_collate_fn,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for better performance
        prefetch_factor=2,  # Prefetch 2 batches per worker for smoother pipeline
        persistent_workers=True  # Keep workers alive between epochs
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=settings["batch_size"] * 2,  # Use larger batch size for validation (no gradients needed)
        shuffle=False,
        num_workers=settings['dataloader_workers'] // 2,  # Fewer workers for validation
        collate_fn=custom_collate_fn,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    # Initialize model, optimizer, loss function
    try:
        from deep_events.torch.training_functions import get_model_generator
    except ImportError:
        logger.error("Could not import get_model_generator, please ensure deep_events is installed correctly")
        raise
        
    model_generator = get_model_generator(settings.get('model', 'unet'))
    model, optimizer, loss_fn, metrics = model_generator(settings, train_dataset[0][0].shape)
    model = model.to(device)
    
    # Initialize gradient scaler for mixed precision
    scaler = GradScaler()
    logger.info("Mixed precision training enabled")
    
    # Initialize tensorboard logger - reduce logging frequency for better performance
    if isinstance(folder, str):
        folder = Path(folder)
    
    logs_dir = folder.parents[0] / (settings.get("log_dir", "logs") + "/scalars/")
    name = short_name = Path(folder).parts[-1][:13]
    logdir = logs_dir / name
    i = 0
    while logdir.exists():
        name = short_name + f"_{i}"
        logdir = logs_dir / name
        i += 1
    os.makedirs(logdir, exist_ok=True)
    
    # Create enhanced logger with reduced logging frequency
    tb_logger = TensorboardLogger(logdir)
    logger.info(f"TensorBoard logs will be saved to: {os.path.abspath(logdir)}")
    
    # Log settings
    for key, value in settings.items():
        tb_logger.add_text(key, str(value), 0)
    
    # Initialize OneCycleLR scheduler
    total_steps = settings["epochs"] * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=settings["initial_learning_rate"],
        total_steps=total_steps,
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=1000.0,
        verbose=False
    )
    
    # Initialize early stopping
    best_val_loss = float('inf')
    best_model_state = None
    patience = settings.get("patience", 25)
    patience_counter = 0
    
    # Reduce save frequency for better performance
    save_frequency = settings.get("save_frequency", 10)
    
    # Reduce validation frequency for better throughput
    val_frequency = settings.get("val_frequency", 1)  # Validate every N epochs
    
    # Reduce logging frequency for console output
    log_batch_frequency = settings.get("log_batch_frequency", 50)  # Log every N batches
    
    # Training loop with performance optimizations
    total_epochs = settings["epochs"]
    for epoch in range(total_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_metrics_values = {name: 0.0 for name in metrics}
        batch_count = 0
        
        # Use of a progress bar to reduce logging overhead
        from tqdm import tqdm
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        # Pre-allocate GPU memory for metrics
        metric_cache = {}
        
        # Start timer for measuring epoch time
        start_time = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(train_iterator):
            # Non-blocking transfer to GPU
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Forward pass
            optimizer.zero_grad(set_to_none=True)  # Faster than setting to zero
            with autocast('cuda'):
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
            
            # Backward pass 
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            # Update scheduler without sync point
            scheduler.step()
            
            # Accumulate metrics without blocking GPU
            # Avoid calling .item() every iteration which causes synchronization
            loss_val = loss.detach()  # Detach without transferring to CPU
            train_loss += loss_val  # Keep on GPU until end of epoch
            batch_count += 1
            
            # Only log periodically to reduce overhead
            if batch_idx % log_batch_frequency == 0:
                try:
                    loss_cpu = loss.item()  # Now we synchronize, but less frequently
                    logger.info(f'Epoch {epoch} Batch {batch_idx}/{len(train_loader)} '
                           f'Loss: {loss_cpu:.6f} LR: {scheduler.get_last_lr()[0]:.6f}')
                except:
                    pass  # Don't crash if logging fails
        
        # Calculate final metrics on CPU at end of epoch (single sync point)
        train_loss_cpu = train_loss.sum().item() / batch_count
        
        # Only perform validation at specified frequency
        run_validation = (epoch % val_frequency == 0) or (epoch == total_epochs - 1)
        
        if run_validation:
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_metrics_values = {name: 0.0 for name in metrics}
            val_batch_count = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    # Non-blocking transfer
                    inputs = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    
                    # Forward pass
                    with autocast('cuda'):
                        outputs = model(inputs)
                        loss = loss_fn(outputs, targets)
                    
                    # Keep on GPU until end of validation
                    val_loss += loss.detach()
                    val_batch_count += 1
            
            # Calculate validation metrics on CPU (single sync point)
            val_loss_cpu = val_loss.sum().item() / val_batch_count
            
            # Print epoch summary (reduced metrics for speed)
            logger.info(f'Epoch {epoch} summary (time: {time.time() - start_time:.2f}s):')
            logger.info(f'  Train loss: {train_loss_cpu:.6f}')
            logger.info(f'  Val loss: {val_loss_cpu:.6f}')
            
            # Log metrics to tensorboard (reduced frequency)
            tb_logger.add_scalar('train_loss', train_loss_cpu, epoch)
            tb_logger.add_scalar('val_loss', val_loss_cpu, epoch)
            tb_logger.add_scalar('learning_rate', scheduler.get_last_lr()[0], epoch)
            
            # Check early stopping and save model
            if val_loss_cpu < best_val_loss:
                best_val_loss = val_loss_cpu
                patience_counter = 0
                
                # Asynchronously save best model (use a background thread)
                # def save_best_model():
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                #     save_model(model, optimizer, scaler, epoch, settings, val_loss_cpu, folder, is_best=True)
                #     logger.info(f"Saved new best model with val_loss: {val_loss_cpu:.6f}")
                
                # Only spawn thread if not distributed to avoid conflicts
                # if not distributed:
                #     try:
                #         import threading
                #         threading.Thread(target=save_best_model).start()
                #     except:
                #         # Fall back to synchronous save if threading fails
                #         save_best_model()
                # else:
                #     save_best_model()
            else:
                patience_counter += 1
                # Only save periodically
                # if epoch % save_frequency == 0:
                #     save_model(model, optimizer, scaler, epoch, settings, val_loss_cpu, folder, is_best=False)
                    
                if patience_counter >= patience:
                    logger.info(f'Early stopping triggered after {epoch+1} epochs')
                    break
        else:
            # When skipping validation, just log training metrics
            logger.info(f'Epoch {epoch} summary (time: {time.time() - start_time:.2f}s):')
            logger.info(f'  Train loss: {train_loss_cpu:.6f}')
            tb_logger.add_scalar('train_loss', train_loss_cpu, epoch)
            tb_logger.add_scalar('learning_rate', scheduler.get_last_lr()[0], epoch)
        
        # Call on_epoch_end to update dataset indices
        train_dataset.on_epoch_end()
        val_dataset.on_epoch_end()
        
        # Log images only occasionally to reduce overhead
        if epoch % 5 == 0 and run_validation:
            with torch.no_grad():
                # Get a batch of validation data
                for val_inputs, val_targets in val_loader:
                    val_inputs = val_inputs.to(device, non_blocking=True)
                    val_targets = val_targets.to(device, non_blocking=True)
                    
                    # Generate predictions
                    with autocast('cuda'):
                        val_outputs = model(val_inputs)
                        val_outputs_sigmoid = torch.sigmoid(val_outputs)
                    
                    # Only log a few images
                    n_images = min(8, val_inputs.size(0))
                    
                    if val_inputs.size(1) > 1 and len(val_inputs.shape) == 4:
                        display_inputs = val_inputs[:n_images, 0:1]
                    else:
                        display_inputs = val_inputs[:n_images]
                    
                    display_outputs = val_outputs_sigmoid[:n_images]
                    display_targets = val_targets[:n_images]
                    
                    # Add to tensorboard
                    tb_logger.add_images('inputs', display_inputs, epoch)
                    tb_logger.add_images('predictions', display_outputs, epoch)
                    tb_logger.add_images('targets', display_targets, epoch)
                    break  # Just log one batch
    
    # Save final model
    if best_model_state is not None and patience_counter >= patience:
        print("Saving best model")
        model.load_state_dict(best_model_state)
    
    # save settings to yaml file
    settings['data_dir'] = data_folder
    save_dict_to_yaml(settings, folder / f'{name}_settings.yaml')
    save_model(model, optimizer, scaler, epoch, settings, val_loss_cpu if run_validation else train_loss_cpu, 
              folder, name, is_final=True)
    logger.info(f"Training completed in {time.time() - start_time:.2f}s.")
    
    # Close tensorboard logger
    tb_logger.close()
    
    return True

def main():
    """Main function to execute training."""
    # Example local paths for testing
    data_folder = Path('./data/training_data')
    folder = Path('./models')
    train(data_folder, folder)

if __name__ == "__main__":
    main()