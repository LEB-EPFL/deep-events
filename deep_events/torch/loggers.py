import torch
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
import os
import logging

class LogImages:
    """
    Class for logging images to TensorBoard during training.
    This integrates with the PyTorch training loop to periodically log:
    - Input images
    - Model predictions
    - Ground truth masks
    - Difference maps (prediction vs ground truth)
    """
    
    def __init__(self, log_dir: Path, train_dataset, val_dataset, freq: int = 1):
        """
        Initialize the image logger.
        
        Parameters:
        -----------
        log_dir : Path
            Directory where to save TensorBoard logs
        train_dataset : torch.utils.data.Dataset
            Training dataset
        val_dataset : torch.utils.data.Dataset
            Validation dataset
        freq : int
            Frequency of logging (in epochs)
        """
        self.writer = SummaryWriter(log_dir=str(log_dir))
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.log_dir = log_dir
        self.freq = freq
        
        # Create a figure directory
        self.figure_dir = log_dir / "figures"
        os.makedirs(self.figure_dir, exist_ok=True)
        
        # Sample indices for consistent visualization
        self.train_indices = np.random.choice(
            len(train_dataset), size=min(4, len(train_dataset)), replace=False
        )
        self.val_indices = np.random.choice(
            len(val_dataset), size=min(4, len(val_dataset)), replace=False
        )
        
        logging.info(f"Image logger initialized at {log_dir}")
    
    def log_batch(self, model: torch.nn.Module, device: torch.device, epoch: int):
        """
        Log sample images, predictions, and targets to TensorBoard.
        
        Parameters:
        -----------
        model : torch.nn.Module
            The model to generate predictions
        device : torch.device
            Device on which to run the model
        epoch : int
            Current epoch number
        """
        if epoch % self.freq != 0:
            return
        
        logging.info(f"Logging images for epoch {epoch}")
        
        # Set model to evaluation mode
        model.eval()
        
        # Log training samples
        self._log_dataset_samples("train", model, device, epoch, self.train_indices, self.train_dataset)
        
        # Log validation samples
        self._log_dataset_samples("val", model, device, epoch, self.val_indices, self.val_dataset)
    
    def _log_dataset_samples(self, name: str, model: torch.nn.Module, device: torch.device, 
                            epoch: int, indices: np.ndarray, dataset):
        """
        Log samples from a dataset to TensorBoard.
        
        Parameters:
        -----------
        name : str
            Name prefix for the logs (e.g., 'train', 'val')
        model : torch.nn.Module
            The model to generate predictions
        device : torch.device
            Device on which to run the model
        epoch : int
            Current epoch number
        indices : np.ndarray
            Indices of samples to log
        dataset : torch.utils.data.Dataset
            Dataset from which to take samples
        """
        # Create figure for combined visualization
        fig = plt.figure(figsize=(15, 5 * len(indices)))
        
        # Process each sample
        with torch.no_grad():
            for i, idx in enumerate(indices):
                # Get sample
                inputs, target = dataset[idx]
                inputs = inputs.unsqueeze(0).to(device)  # Add batch dimension
                target = target.unsqueeze(0).to(device)  # Add batch dimension
                
                # Generate prediction
                prediction = model(inputs)
                
                # Convert to numpy for plotting
                if inputs.shape[1] > 1:  # If multi-channel (e.g., multiple time steps)
                    input_np = inputs[0, 0].cpu().numpy()  # Take first channel of first batch
                else:
                    input_np = inputs[0, 0].cpu().numpy()  # Take first channel of first batch
                
                prediction_np = prediction[0, 0].cpu().numpy()
                target_np = target[0, 0].cpu().numpy()
                diff_np = np.abs(prediction_np - target_np)
                
                # Plot the sample
                ax1 = fig.add_subplot(len(indices), 4, i*4 + 1)
                ax1.imshow(input_np, cmap='gray')
                ax1.set_title(f"Input (Sample {idx})")
                ax1.axis('off')
                
                ax2 = fig.add_subplot(len(indices), 4, i*4 + 2)
                ax2.imshow(prediction_np, cmap='gray')
                ax2.set_title(f"Prediction")
                ax2.axis('off')
                
                ax3 = fig.add_subplot(len(indices), 4, i*4 + 3)
                ax3.imshow(target_np, cmap='gray')
                ax3.set_title(f"Ground Truth")
                ax3.axis('off')
                
                ax4 = fig.add_subplot(len(indices), 4, i*4 + 4)
                ax4.imshow(diff_np, cmap='hot')
                ax4.set_title(f"Absolute Difference")
                ax4.axis('off')
                
                # Also log to TensorBoard individually
                self.writer.add_image(f"{name}/sample_{idx}/input", 
                                     inputs[0, 0:1], epoch)
                self.writer.add_image(f"{name}/sample_{idx}/prediction", 
                                     prediction[0], epoch)
                self.writer.add_image(f"{name}/sample_{idx}/target", 
                                     target[0], epoch)
                self.writer.add_image(f"{name}/sample_{idx}/diff", 
                                     torch.abs(prediction[0] - target[0]), epoch)
        
        # Save the combined figure
        plt.tight_layout()
        
        # Save as PNG
        fig_path = self.figure_dir / f"{name}_samples_epoch_{epoch}.png"
        plt.savefig(fig_path)
        
        # Log figure to TensorBoard
        self._figure_to_tensorboard(fig, f"{name}/combined_samples", epoch)
        
        plt.close(fig)
    
    def _figure_to_tensorboard(self, figure, tag, step):
        """
        Convert matplotlib figure to TensorBoard image.
        
        Parameters:
        -----------
        figure : matplotlib.figure.Figure
            Figure to convert
        tag : str
            Tag for TensorBoard
        step : int
            Step/epoch for TensorBoard
        """
        # Save the plot to a buffer
        buf = io.BytesIO()
        figure.savefig(buf, format='png')
        buf.seek(0)
        
        # Convert to PIL Image
        image = Image.open(buf)
        
        # Convert to numpy array
        image_np = np.array(image)
        
        # Log to TensorBoard
        self.writer.add_image(tag, image_np, step, dataformats='HWC')
    
    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close()


class MetricsLogger:
    """
    Class for logging metrics during training.
    Records and reports various metrics, both to TensorBoard and to a CSV file.
    """
    
    def __init__(self, log_dir: Path, metrics: Dict[str, Callable]):
        """
        Initialize the metrics logger.
        
        Parameters:
        -----------
        log_dir : Path
            Directory where to save logs
        metrics : Dict[str, Callable]
            Dictionary of metric names and functions
        """
        self.writer = SummaryWriter(log_dir=str(log_dir))
        self.log_dir = log_dir
        self.metrics = metrics
        
        # Initialize metric history
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': []
        }
        
        for metric_name in metrics:
            self.history[f'train_{metric_name}'] = []
            self.history[f'val_{metric_name}'] = []
        
        logging.info(f"Metrics logger initialized at {log_dir}")
    
    def log_metrics(self, epoch: int, train_loss: float, val_loss: float, 
                   train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """
        Log metrics for an epoch.
        
        Parameters:
        -----------
        epoch : int
            Current epoch number
        train_loss : float
            Training loss
        val_loss : float
            Validation loss
        train_metrics : Dict[str, float]
            Dictionary of training metrics
        val_metrics : Dict[str, float]
            Dictionary of validation metrics
        """
        # Update history
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        
        for metric_name, value in train_metrics.items():
            self.history[f'train_{metric_name}'].append(value)
            
        for metric_name, value in val_metrics.items():
            self.history[f'val_{metric_name}'].append(value)
        
        # Log to TensorBoard
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        self.writer.add_scalar('Loss/val', val_loss, epoch)
        
        for metric_name, value in train_metrics.items():
            self.writer.add_scalar(f'Metrics/{metric_name}/train', value, epoch)
            
        for metric_name, value in val_metrics.items():
            self.writer.add_scalar(f'Metrics/{metric_name}/val', value, epoch)
        
        # Save history to CSV
        self._save_history_to_csv()
        
        # Create and save plots
        if epoch % 5 == 0 or epoch == 1:
            self._create_and_save_plots()
    
    def _save_history_to_csv(self):
        """Save training history to a CSV file."""
        import pandas as pd
        
        # Convert history to DataFrame
        df = pd.DataFrame(self.history)
        
        # Save to CSV
        csv_path = self.log_dir / "training_history.csv"
        df.to_csv(csv_path, index=False)
    
    def _create_and_save_plots(self):
        """Create and save metric plots."""
        # Create loss plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.history['epoch'], self.history['train_loss'], label='Train Loss')
        ax.plot(self.history['epoch'], self.history['val_loss'], label='Val Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True)
        
        # Save loss plot
        loss_plot_path = self.log_dir / "loss_plot.png"
        fig.savefig(loss_plot_path)
        plt.close(fig)
        
        # Create a plot for each metric
        for metric_name in self.metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(self.history['epoch'], self.history[f'train_{metric_name}'], 
                   label=f'Train {metric_name}')
            ax.plot(self.history['epoch'], self.history[f'val_{metric_name}'], 
                   label=f'Val {metric_name}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_name)
            ax.set_title(f'Training and Validation {metric_name}')
            ax.legend()
            ax.grid(True)
            
            # Save metric plot
            metric_plot_path = self.log_dir / f"{metric_name}_plot.png"
            fig.savefig(metric_plot_path)
            plt.close(fig)
        
        # Create a combined metrics plot
        fig, axes = plt.subplots(len(self.metrics), 1, figsize=(10, 5 * len(self.metrics)), 
                               sharex=True)
        
        if len(self.metrics) == 1:
            axes = [axes]
        
        for i, metric_name in enumerate(self.metrics):
            axes[i].plot(self.history['epoch'], self.history[f'train_{metric_name}'], 
                        label=f'Train {metric_name}')
            axes[i].plot(self.history['epoch'], self.history[f'val_{metric_name}'], 
                        label=f'Val {metric_name}')
            axes[i].set_ylabel(metric_name)
            axes[i].set_title(f'Training and Validation {metric_name}')
            axes[i].legend()
            axes[i].grid(True)
        
        axes[-1].set_xlabel('Epoch')
        plt.tight_layout()
        
        # Save combined metrics plot
        combined_plot_path = self.log_dir / "combined_metrics_plot.png"
        fig.savefig(combined_plot_path)
        plt.close(fig)
    
    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close()


if __name__ == "__main__":
    # Test the loggers
    import tempfile
    import torch.nn.functional as F
    
    # Define test metrics
    def accuracy(y_pred, y_true):
        """Binary accuracy."""
        y_pred_binary = (y_pred > 0.5).float()
        correct = (y_pred_binary == y_true).float().sum()
        return correct / y_true.numel()
    
    def dice_coef(y_pred, y_true, smooth=1):
        """Dice coefficient."""
        y_pred_flat = y_pred.view(-1)
        y_true_flat = y_true.view(-1)
        intersection = (y_pred_flat * y_true_flat).sum()
        return (2. * intersection + smooth) / (y_pred_flat.sum() + y_true_flat.sum() + smooth)
    
    # Create temp directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_dir = Path(tmp_dir)
        
        # Create metrics logger
        metrics_logger = MetricsLogger(
            log_dir=log_dir,
            metrics={
                'accuracy': accuracy,
                'dice_coef': dice_coef
            }
        )
        
        # Log some fake metrics
        for epoch in range(10):
            train_loss = 1.0 - 0.05 * epoch
            val_loss = 1.2 - 0.04 * epoch
            
            train_metrics = {
                'accuracy': 0.7 + 0.02 * epoch,
                'dice_coef': 0.6 + 0.03 * epoch
            }
            
            val_metrics = {
                'accuracy': 0.65 + 0.02 * epoch,
                'dice_coef': 0.55 + 0.03 * epoch
            }
            
            metrics_logger.log_metrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_metrics=train_metrics,
                val_metrics=val_metrics
            )
        
        metrics_logger.close()
        
        print(f"MetricsLogger test complete. Files saved to {log_dir}")