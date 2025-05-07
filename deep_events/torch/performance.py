from pathlib import Path
import os
import re
import csv
from typing import Callable, List, Tuple, Dict, Union, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import logging

import tifffile
import torch
import torch.nn as nn
import numpy as np
import yaml
from skimage.filters import threshold_otsu
from skimage import measure

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FOLDER = Path("W:/deep_events/data/original_data/training_data")
SETTINGS_NAME = 'testing'
SETTINGS_NAME = 'f1_settings'
SETTINGS_NAME = 'settings'
# MCC_WEIGHTS = [5, 1, 15, 1]
# MCC_WEIGHTS = [5, 1, 5, 1]
# MCC_WEIGHTS = [5, 3, 10, 1]
MCC_WEIGHTS = [5, 5, 10, 1]
beta = 0.1

# Utility functions for metrics calculation
def get_precision(tp, fp):
    """Calculate precision."""
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def get_tpr(tp, fn):
    """Calculate true positive rate (recall)."""
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def get_f1_score(precision, recall):
    """Calculate F1 score."""
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

def get_mcc(tp, fp, fn, tn):
    """Calculate Matthews correlation coefficient."""
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return (tp * tn - fp * fn) / denominator if denominator > 0 else 0

def get_weighted_mcc(tp, fp, fn, tn, w_tp=1, w_tn=1, w_fp=1, w_fn=1):
    """Calculate weighted Matthews correlation coefficient."""
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return (w_tp * tp * w_tn * tn - w_fp * fp * w_fn * fn) / denominator if denominator > 0 else 0

def get_kappa(tp, fp, fn, tn):
    """Calculate Cohen's kappa."""
    total = tp + fp + fn + tn
    po = (tp + tn) / total
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (total ** 2)
    return (po - pe) / (1 - pe) if (1 - pe) > 0 else 0

# Find latest folder/file utility
def get_latest_folder(folder: Path, pattern: str = "*"):
    """Find latest folder matching the pattern."""
    def folder_date_sort(x):
        return int(x.parts[-1][:8])
    folders = [x for x in Path(folder).glob(pattern) if x.is_dir()]
    folders = sorted(folders, key=folder_date_sort, reverse=True)
    return [f for f in folders]

def get_latest(file_type: str, folder: Path):
    """Find latest file of a given type in a folder."""
    files = list(folder.glob("*" + file_type + "*"))
    def date_sort(x):
        try:
            return int(x.parts[-1][:8])
        except ValueError:
            return 0
    files = sorted(files, key=date_sort, reverse=True)
    return files[0]

# Label images for event detection
def label(images, threshold):
    """Label connected components in binary masks."""
    labels = []
    for i in range(images.shape[0]):
        im = images[i].squeeze()
        # Apply threshold to create binary mask
        binary = im > threshold
        # Label connected components
        label_img = measure.label(binary)
        labels.append(label_img)
    return np.array(labels)

# Evaluation functions
def whole_ev_eval(masks, preds, plot, eval_frames, threshold=None):
    """Evaluate on whole event level."""
    if threshold is None:
        threshold = threshold_otsu(np.array(preds))
    TP, FP, FN, TN = 0, 0, 0, 0
    tp_values, fp_values, fn_values, tn_values = [], [], [], []
    
    for gt, pred in zip(masks, preds):
        max_pred = pred.max()
        max_gt = gt.max() 
        if max_gt > 0.5:
            if max_pred > threshold:
                TP += 1
                tp_values.append(max_pred)
            else:
                FN += 1
                fn_values.append(max_pred)
        else:
            if max_pred > threshold:
                FP += 1
                fp_values.append(max_pred)
            else:
                TN += 1
                tn_values.append(max_pred)
    return [TP, FP, FN, TN, tp_values, fp_values, fn_values, tn_values, threshold]

def event_loc_fp_eval(eval_mask, pred_output_test, plot, eval_frames, threshold):
    """Evaluate with event localization focus."""
    all_preds = label(pred_output_test, threshold) 
    all_true_labels = label(eval_mask, 0.7)
    TP, FP, FN, TN = np.uint32(0), np.uint32(0), np.uint32(0), np.uint32(0)
    tp_values, fp_values, fn_values, tn_values = [], [], [], []
    
    for labels, preds, maps in zip(all_true_labels, all_preds, pred_output_test):
        # Adjust dimensions based on your PyTorch model output structure
        if len(preds.shape) > 2:
            preds = preds[0]
        if len(maps.shape) > 2:
            maps = maps[0]
            
        pred_copy = preds.copy()
        
        for event in np.unique(labels)[1:]:
            pred_labels = preds[labels==event]
            values = maps[labels==event]
            if pred_labels.max() > 0:
                TP += 1
                tp_values.append(values.max())
                for tp in np.unique(pred_labels)[1:]:
                    pred_copy[pred_copy == tp] = 0
            else:
                FN += 1
                fn_values.append(values.max())
                
        for event in np.unique(pred_copy)[1:]:
            FP += 1
            values = maps[pred_copy==event]
            fp_values.append(values.max())
            
    return [TP, FP, FN, TN, tp_values, fp_values, fn_values, tn_values, threshold]

def frame_wise_eval(eval_mask, pred_output_test, plot, eval_frames, threshold):
    """Evaluate on a frame-by-frame basis."""
    all_preds = label(pred_output_test, threshold) 
    all_true_labels = label(eval_mask, 0.5)
    TP, FP, FN, TN = np.uint32(0), np.uint32(0), np.uint32(0), np.uint32(0)
    tp_values, fp_values, fn_values, tn_values = [], [], [], []
    
    for labels, preds, maps in zip(all_true_labels, all_preds, pred_output_test):
        # Adjust dimensions based on your PyTorch model output structure
        if len(preds.shape) > 2:
            preds = preds[0]
        if len(maps.shape) > 2:
            maps = maps[0]
            
        value = maps.max()
        if labels.max() > 0:
            if preds.max() > 0:
                TP += 1
                tp_values.append(value)
            else: 
                FN += 1
                fn_values.append(value)
        else:
            if preds.max() > 0:
                FP += 1
                fp_values.append(value)
            else:
                TN += 1
                tn_values.append(value)
                
    return [TP, FP, FN, TN, tp_values, fp_values, fn_values, tn_values, threshold]

def pixel_wise_eval(eval_mask, pred_output_test, plot, eval_frames, threshold):
    """Evaluate on a pixel-wise basis."""
    predictions = pred_output_test
    ground_truth = eval_mask
    
    # Handle dimensionality to ensure we compare 2D arrays
    if len(predictions.shape) > 3:
        predictions = predictions.reshape(predictions.shape[0], -1)
    if len(ground_truth.shape) > 3:
        ground_truth = ground_truth.reshape(ground_truth.shape[0], -1)
    
    tolerance = 0.05
    mask = ground_truth > 0.05
    predictions = predictions[mask]
    ground_truth = ground_truth[mask]
    
    TP = int(np.sum(np.abs(predictions - ground_truth) <= tolerance))  # Matches within tolerance
    FP = int(np.sum((predictions > ground_truth + tolerance)))  # Predictions too high outside tolerance
    FN = int(np.sum((predictions < ground_truth - tolerance)))
    TN = int(predictions.size - (TP + FP + FN))  # Remaining are true negatives
    
    tp_values, fp_values, fn_values, tn_values = [], [], [], []
    return [TP, FP, FN, TN, tp_values, fp_values, fn_values, tn_values, threshold]

def fission_stats_stack(true_labels, pred_labels):
    """
    Calculate fission statistics for a stack of images.
    Implementation placeholder - detailed implementation would depend on the specific metrics needed.
    """
    # This would need to be implemented based on the specific requirements
    # For now, returning a basic structure
    return [0, 0, 0, 0]  # [TP, FP, FN, TN]

def event_loc_eval(eval_mask, pred_output_test, plot, *_):
    """Evaluate event localization."""
    labels = label(pred_output_test, 0.7)
    true_labels = label(eval_mask, 0.7)

    if plot:
        try:
            import random
            to_plot = random.sample(range(labels.shape[0]), labels.shape[0])
            frame = 0
            while True:
                f, axs = plt.subplots(1, 3)
                f.set_size_inches(25,10)
                axs[0].imshow(labels[to_plot[frame]], vmax=1, cmap='Reds')
                axs[0].set_title("prediction")
                axs[1].imshow(true_labels[to_plot[frame]])
                axs[1].set_title("ground truth")
                frame += 1
                plt.show()
        except KeyboardInterrupt:
            pass

    return fission_stats_stack(true_labels, labels)

def optimize_threshold(eval_mask, pred_output_test, model_path, eval_event_frames, eval_func=whole_ev_eval):
    """Find the optimal threshold for prediction."""
    results = {
        "thresholds": [], 
        "precision": [], 
        "recall": [], 
        "f1": [], 
        "mcc": [], 
        "kappa": [],
        "specificity": [], 
        "npv": [], 
        "w_mcc": [], 
        f"f_beta": [], 
        'beta': beta
    }
    
    for threshold in tqdm(np.linspace(0.05, 1, 96)):
        stats = eval_func(eval_mask, pred_output_test, False, eval_event_frames, threshold)
        
        # Calculate precision
        try:
            precision = get_precision(stats[0], stats[1])
        except:
            precision = 0
            
        # Calculate recall
        try:
            recall = get_tpr(stats[0], stats[2])
        except:
            recall = 0
            
        # Calculate F1 score
        try:
            f1 = round(get_f1_score(precision, recall)*100)/100
        except:
            f1 = 0
            
        # Calculate other metrics
        specificity = round(stats[3] / (stats[3] + stats[1]) * 100)/100 if (stats[3] + stats[1]) != 0 else 0
        npv = round(stats[3] / (stats[3] + stats[2]) * 100)/100 if (stats[3] + stats[2]) != 0 else 0
        
        # Calculate F-beta score
        beta_squared = beta ** 2
        if precision + recall == 0:
            fbeta = 0
        else:
            fbeta = (1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall)
            
        # Round values
        precision = round(precision*100)/100
        recall = round(recall*100)/100 
        mcc = round(get_mcc(*stats[:4])*100)/100
        w_mcc = round(get_weighted_mcc(*stats[:4], *MCC_WEIGHTS)*100)/100
        kappa = round(get_kappa(*stats[:4])*100)/100
        fbeta = round(fbeta*100)/100
        
        # Store results
        results[f"f_beta"].append(fbeta)
        results["thresholds"].append(threshold)
        results["precision"].append(precision)
        results["specificity"].append(specificity)
        results["npv"].append(npv)
        results["recall"].append(recall)
        results["f1"].append(f1)
        results["mcc"].append(mcc)
        results["w_mcc"].append(w_mcc)
        results["kappa"].append(kappa)
        
    # Find optimal threshold based on F-beta score
    fbeta_threshold = results["thresholds"][results[f"f_beta"].index(max(results[f"f_beta"]))]
    f1_threshold = results["thresholds"][results[f"f1"].index(max(results[f"f1"]))]
    threshold = fbeta_threshold
    
    # Save plots if needed
    if isinstance(model_path, (str, Path)):
        plt.figure()
        scores = ["f1", "mcc", "w_mcc", "recall", "precision", f"f_beta"]
        for score in scores:
            plt.plot(results["thresholds"], results[score])
        plt.axvline(threshold)
        plt.legend(scores)
        
        # Convert model_path to string if it's a Path
        model_path_str = str(model_path) if isinstance(model_path, Path) else model_path
        plt.savefig(model_path_str.replace("model.pt", "perf.png"))
        plt.close()
        
        # Save results to YAML - with proper serialization
        try:
            with open(model_path_str.replace('model.pt', f'perf_data_{SETTINGS_NAME}.yaml'), 'w') as f:
                yaml.dump(results, f, default_flow_style=False)
        except Exception as e:
            logger.error(f"Error saving performance data: {e}")
    
    return threshold

def load_model_safely(model_path, device=None):
    """
    Load a PyTorch model safely using weights_only=True.
    
    Args:
        model_path: Path to the model file
        device: Device to load the model on
        
    Returns:
        Loaded model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # First try to load with weights_only=True
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        
        # If it's a state dict, we need to create a model and load the weights
        if isinstance(checkpoint, dict):
            # Load settings to determine model type
            settings_path = str(model_path).replace('model.pt', 'settings.yaml')

            with open(settings_path, 'r') as f:
                settings = yaml.safe_load(f)
            

            from deep_events.torch.training_functions import get_model_generator
            model_type = settings.get('model', 'temporal_unet')
            logger.info(f"Creating model of type {model_type}")
            
            # Get model generator and create model
            model_generator = get_model_generator(model_type)
            
            # Determine input shape from settings
            t_size = settings.get('n_timepoints', 1)
            input_shape = (1, t_size, 256, 256)  # Default shape if not specified
            
            # Create model
            model, _, _, _ = model_generator(settings, input_shape)
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Maybe the checkpoint is already a state dict
                model.load_state_dict(checkpoint)
            
            logger.info(f"Successfully loaded model from {model_path}")
            return model
                    
        else:
            # If checkpoint is already a model, return it
            return checkpoint
            
    except Exception as e:
        logger.warning(f"Error loading model with weights_only=True: {e}")
        # Fall back to unsafe loading with warning
        logger.warning("Falling back to unsafe loading method")
        return torch.load(model_path, map_location=device)

def predict(images, model, device=None):
    """
    Run predictions with PyTorch model.
    
    Args:
        images: Input images as numpy array
        model: PyTorch model
        device: Device to run inference on
        
    Returns:
        Predictions as numpy array
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
    # Convert numpy array to PyTorch tensor
    if isinstance(images, np.ndarray):
        if images.ndim == 3:  # Single image with channels
            images = np.expand_dims(images, axis=0)  # Add batch dimension
            
        # PyTorch expects BCHW format
        if images.shape[1] != 1 and images.shape[1] != 3:  # If channel dim is not in position 1
            # Assume BHWC format and convert to BCHW
            images = np.transpose(images, (0, 3, 1, 2))
            
        images_tensor = torch.from_numpy(images).float()
    else:
        images_tensor = images
    
    # Move to device
    images_tensor = images_tensor.to(device)
    
    predictions = []
    with torch.no_grad():
        # Process in batches if the input is large
        batch_size = 16  # Adjust based on your GPU memory
        num_samples = images_tensor.size(0)
        
        for i in range(0, num_samples, batch_size):
            batch = images_tensor[i:i+batch_size]
            batch_pred = model(batch)
            predictions.append(batch_pred.cpu().numpy())
    
    # Concatenate results and return
    return np.concatenate(predictions, axis=0)

def main(model_dir: Union[Path, str] = None, write_yaml: bool = True, plot: bool = False, 
         general_eval: bool = False, eval_func: Callable = whole_ev_eval, save_hist: bool = True):
    """
    Main function to evaluate PyTorch model performance.
    
    Args:
        model_dir: Path to the model file
        write_yaml: Whether to write results to YAML file
        plot: Whether to create plots
        general_eval: Whether to use general evaluation
        eval_func: Evaluation function to use
        save_hist: Whether to save histogram data
    
    Returns:
        F1 score
    """
    no_details = True
    model_dir = Path(model_dir) if model_dir is not None else None
    
    # Find model if not specified
    if model_dir is None:
        training_folder = Path(get_latest_folder(FOLDER)[0])
        model_dir = Path(get_latest("model", training_folder))
        print(model_dir)
    elif str(model_dir)[-3:] != '.pt':
        training_folder = model_dir
        model_dir = list(training_folder.glob('*model.pt'))[0]
        print(model_dir)
    else:
        training_folder = model_dir.parent
    
    # Load settings
    settings_path = str(model_dir).replace('model.pt', 'settings.yaml')
    try:
        with open(settings_path, 'r') as f:
            settings = yaml.safe_load(f)
        if settings is None:
            settings = {}
    except Exception as e:
        logger.warning(f"Could not load settings from {settings_path}: {e}")
        settings = {}
    
    # Get data folder
    data_folder = Path(settings.get('data_dir', training_folder))
    
    # Get number of frames
    frames = settings.get('n_timepoints', 1)
    if frames == 1 and '_n' in str(training_folder):
        try:
            frames = int(str(training_folder).split('_n')[1][0])
            logger.info(f'n timepoints from folder name: {frames}')
        except (ValueError, IndexError):
            pass
    
    # For general evaluation
    if general_eval:
        parts = model_dir.parent.name.split("_")[2:]
        # Remove subset, so that we check on full validation data
        subset = [re.match(r"s0\..*", x) for x in parts]
        parts = [x for x, sub in zip(parts, subset) if not sub]
        pattern = "*" + "_".join(parts)

        logger.info(f"General eval pattern: {pattern}")
        logger.info(f"Model parent dir: {model_dir.parents[1]}")
        training_folder = Path(get_latest_folder(model_dir.parents[1], pattern)[0])
        logger.info(f"Training folder: {training_folder}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model_safely(model_dir, device)
    model.eval()
    
    # Create data loader for evaluation

    from deep_events.torch.generator import ArraySequence
   
    eval_seq = ArraySequence(
        data_dir=data_folder, 
        batch_size=1, 
        n_augmentations=1, 
        validation=True, 
        t_size=frames
    )
    eval_seq.performance = True

    try:
        with open(data_folder / 'eval_events.csv', 'r') as read_obj: 
            csv_reader = csv.reader(read_obj) 
            eval_event_frames = list(csv_reader)
            for i in range(len(eval_event_frames)):
                eval_event_frames[i] = [int(x) for x in eval_event_frames[i]]
    except FileNotFoundError:
        eval_func = event_loc_eval
        eval_event_frames = []
        no_details = True
    
    n_repeats = 3
    
    # Create arrays to store masks and images
    masks = []
    images = []
    
    # Collect data from DataLoader
    for i in tqdm(range(len(eval_seq) * n_repeats)):
        idx = i % len(eval_seq)
        
        for f in range(-5, 0):
            # Set last_frame if available
            if hasattr(eval_seq, 'last_frame'):
                eval_seq.last_frame = f
                image, mask = eval_seq[idx]
            else:
                # Fallback for SimpleDataset
                image = eval_seq.images[idx]
                mask = eval_seq.masks[idx]
            
            # Convert PyTorch tensors to numpy arrays if needed
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
                
            masks.append(mask)
            images.append(image)
    
    # Reshape arrays for processing
    masks = np.array(masks)
    images = np.array(images)
    
    logger.info(f"Images shape: {images.shape}")
    logger.info(f"Masks shape: {masks.shape}")
    
    # Run predictions
    try:
        # First try processing all at once
        logger.info("Processing all predictions at once")
        pred = predict(images, model)
    except Exception as e:
        # Fall back to processing in parts
        logger.warning(f"Error during prediction, trying in parts: {e}")
        pred = []
        for img in images:
            img_tensor = torch.from_numpy(img).float().unsqueeze(0).to(device)
            pred.append(predict(img_tensor, model))
        pred = np.vstack(pred)
    
    logger.info(f"Predictions shape: {pred.shape}")
    
    # Reshape for evaluation
    try:
        pred = pred.reshape(len(eval_seq) * n_repeats, 5, *mask.shape)
        masks = np.squeeze(masks)
    except Exception as e:
        logger.warning(f"Error reshaping predictions: {e}")
        # Try a different reshape approach
        if len(pred.shape) == 4:  # (samples, channel, height, width)
            pred = pred.transpose(0, 2, 3, 1)  # -> (samples, height, width, channel)
        masks = masks.squeeze()
    
    # Find optimal threshold
    threshold = optimize_threshold(masks, pred, model_dir, eval_event_frames, eval_func)
    
    # Evaluate model
    stats = eval_func(masks, pred, False, eval_event_frames, threshold)
    logger.info(f"Evaluation stats (TP, FP, FN, TN): {stats[:4]}")
    
    # Calculate metrics
    precision = get_precision(stats[0], stats[1])
    recall = get_tpr(stats[0], stats[2])
    f1 = get_f1_score(precision, recall)
    
    # Calculate F-beta
    beta_squared = beta ** 2
    fbeta = (1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall) if precision + recall > 0 else 0
    
    # Round values
    fbeta = round(fbeta * 100) / 100
    precision = round(precision * 100) / 100
    recall = round(recall * 100) / 100
    f1 = round(f1 * 100) / 100
    
    # Generate summary
    summary = f"""
    {model_dir}
    precision {precision}
    recall {recall}
    f1 {f1}
    f_beta {fbeta}
    threshold {threshold}
    """
    
    # Add additional metrics if available
    if not no_details:
        mcc = get_mcc(*stats[:4])
        kappa = get_kappa(*stats[:4])
        mcc = round(mcc * 100) / 100
        kappa = round(kappa * 100) / 100
        w_mcc = get_weighted_mcc(*stats[:4], *MCC_WEIGHTS)
        w_mcc = round(w_mcc * 100) / 100
        summary = summary + f"mcc {mcc}\n    w_mcc {w_mcc}\n    kappa {kappa}\n"
    
    print(summary)
    
    # Write results to YAML file
    if write_yaml:
        settings_path = str(model_dir).replace("model.pt", f"{SETTINGS_NAME}.yaml")
        
        # Load existing settings if available
        if os.path.exists(settings_path):
            try:
                with open(settings_path, 'r') as f:
                    yaml_settings = yaml.safe_load(f)
                    if yaml_settings is None:
                        yaml_settings = {}
            except Exception:
                yaml_settings = {}
        else:
            yaml_settings = {}
        
        # Update performance metrics
        yaml_settings["performance"] = {
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "fbeta": float(fbeta),
            "beta": float(beta)
        }
        
        if not no_details:
            yaml_settings["performance"].update({
                "mcc": float(mcc),
                "w_mcc": float(w_mcc),
                "w_mcc_weights": {
                    "tp": float(MCC_WEIGHTS[0]), 
                    "tn": float(MCC_WEIGHTS[1]),
                    "fp": float(MCC_WEIGHTS[2]), 
                    "fn": float(MCC_WEIGHTS[3])
                },
                "kappa": float(kappa)
            })
            
        yaml_settings["eval_data"] = str(training_folder)
        yaml_settings["performance"]["eval"] = "v1" if no_details else "v2"
        
        # Save to YAML with proper error handling
        try:
            with open(settings_path, 'w') as f:
                yaml.dump(yaml_settings, f, default_flow_style=False)
        except Exception as e:
            logger.error(f"Error saving performance settings: {e}")
    
    # Save histogram data if requested
    if save_hist and not no_details:
        try:
            plt.figure()
            bins = np.arange(0., 1.001, .05)
            plt.hist(stats[4], alpha=.5, label="tp_values", bins=bins, color="green", edgecolor="green", linewidth=2)
            plt.hist(stats[5], alpha=.5, label="fp_values", bins=bins, color="red", edgecolor="red", linewidth=2)
            plt.hist(stats[6], alpha=.5, label="fn_values", bins=bins, color="red", edgecolor="green", linewidth=2)
            plt.hist(stats[7], alpha=.5, label="tn_values", bins=bins, color="green", edgecolor="red", linewidth=2)
            plt.axvline(x=stats[8])
            plt.legend(loc='upper right')
            plt.savefig(str(model_dir).replace("model.pt", "hist.png"))
            plt.close()
            
            # Save histogram data
            hist_data = {'pos': stats[4] + stats[6], 'neg': stats[5] + stats[7]}
            with open(str(model_dir).replace("model.pt", "hist_data_local.yaml"), 'w') as f:
                yaml.dump(hist_data, f)
        except Exception as e:
            logger.error(f"Error saving histogram data: {e}")
    
    return f1

def performance_for_folder(folder: Path, general_eval=True, older_than=0):
    """
    Evaluate all models in a folder.
    
    Args:
        folder: Folder containing models
        general_eval: Whether to use general evaluation
        older_than: Only process models older than this
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    models = list(folder.rglob("*_model.pt"))
    
    for model in models:
        try:
            if older_than > 0:
                model_date = int(model.parts[-1][:len(str(older_than))])
                if model_date < older_than:
                    continue
            
            logger.info('\033[1m' + str(model) + '\033[0m')
            main(model, general_eval=general_eval)
        except Exception as e:
            logger.error(f"Error evaluating model {model}: {e}")

def visual_eval(training_folder, model_name=None):
    """
    Visualize model predictions.
    
    Args:
        training_folder: Folder containing training data
        model_name: Optional specific model name
    """
    import matplotlib.pyplot as plt
    frame = 1
    
    # Find model
    if model_name is None:
        model_dir = Path(get_latest("model", training_folder))
    else:
        model_dir = training_folder / model_name
    
    logger.info(f"Model: {model_dir}")
    
    # Load model safely
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model_safely(model_dir, device)
    model.eval()
    
    # Load data
    eval_images = tifffile.imread(training_folder / "eval_images_00.tif")
    eval_mask = tifffile.imread(training_folder / "eval_gt_00.tif")
    
    # Prepare for PyTorch (BCHW format)
    if eval_images.ndim == 3:  # (frames, height, width)
        eval_images = np.expand_dims(eval_images, axis=1)
    elif eval_images.ndim == 4 and eval_images.shape[1] > 3:  # (frames, time, height, width)
        eval_images = np.transpose(eval_images, (0, 3, 1, 2))
    
    # Interactive visualization
    while True:
        try:
            with torch.no_grad():
                input_tensor = torch.from_numpy(eval_images[frame:frame+1]).float().to(device)
                output = model(input_tensor)
                output_np = output.cpu().numpy()[0, 0]  # Assuming output shape is (batch, channel, height, width)
            
            f, axs = plt.subplots(1, 3)
            f.set_size_inches(15, 5)
            axs[0].imshow(output_np, vmax=1, cmap='Reds')
            axs[0].set_title("prediction")
            
            if eval_mask.ndim == 3:
                mask = eval_mask[frame]
            elif eval_mask.ndim == 4:
                mask = eval_mask[frame, 0]
            
            axs[1].imshow(mask)
            axs[1].set_title("ground truth")
            
            if eval_images.ndim == 4:
                img = eval_images[frame, 0]
            else:
                img = eval_images[frame]
                
            axs[2].imshow(img, cmap='gray')
            axs[2].set_title("input image")
            
            plt.show()
            frame = frame + 1
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error in visualization: {e}")
            frame = frame + 1
            if frame >= len(eval_images):
                break

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "3"
    
    # Example usage for individual model evaluation
    # main(Path("path/to/your/model.pt"))
    
    # Example usage for folder evaluation
    # performance_for_folder(Path("path/to/models/folder"))
    
    # Example configurations for batch evaluation
    main_folder = Path("x:/Scientific_projects/deep_events_WS/data/original_data/training_data/")
    dirs = []
    
    # Example: Evaluate specific model folders
    dirs.extend(list(main_folder.glob('20250312_1022*')))
    # dirs.extend(list(main_folder.glob('20250206_*')))
    
    for folder in dirs:
        models = list(folder.glob("*_model.pt"))
        for mod in models:
            print(mod)
            try:
                f1 = main(folder/mod, write_yaml=True, plot=True, save_hist=True, eval_func=whole_ev_eval)
            except Exception as e:
                logger.error(f"Error evaluating {mod}: {e}")