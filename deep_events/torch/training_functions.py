import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import random
import os
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
os.environ['PYTHONHASHSEED'] = '42'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class SoftFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(SoftFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        logger.info(f'Soft Focal Loss settings: alpha={alpha}, gamma={gamma}')
        
    def forward(self, y_pred, y_true):
        # Check if we're receiving logits or sigmoid values
        if torch.max(y_pred) > 1.0 or torch.min(y_pred) < 0.0:
            # We have raw logits, use binary_cross_entropy_with_logits
            bce = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')
            # For modulating factor, we need sigmoid probabilities
            y_pred_sigmoid = torch.sigmoid(y_pred)
            pt = y_true * y_pred_sigmoid + (1 - y_true) * (1 - y_pred_sigmoid)
        else:
            # We have values in [0,1] range (already sigmoid'd)
            # Clip to avoid numerical issues
            eps = 1e-7
            y_pred = torch.clamp(y_pred, eps, 1.0 - eps)
            bce = -y_true * torch.log(y_pred) - (1 - y_true) * torch.log(1 - y_pred)
            pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        
        # Apply focal loss formula
        alpha_factor = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        focal_weight = (1 - pt) ** self.gamma
        loss = alpha_factor * focal_weight * bce
        
        return torch.mean(loss)

# UNet Model
class UNet(nn.Module):
    def __init__(self, in_channels, nb_filters=16, first_conv_size=12, dropout_rate=0, use_sigmoid=False):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.nb_filters = nb_filters
        self.dropout_rate = dropout_rate
        self.use_sigmoid = use_sigmoid
        
        # Encoder path
        self.down0_conv1 = nn.Conv2d(in_channels, nb_filters, kernel_size=first_conv_size, padding=first_conv_size//2)
        self.down0_bn1 = nn.BatchNorm2d(nb_filters, momentum=0.1, eps=1e-5)
        self.down0_conv2 = nn.Conv2d(nb_filters, nb_filters, kernel_size=first_conv_size, padding=first_conv_size//2)
        self.down0_bn2 = nn.BatchNorm2d(nb_filters, momentum=0.1, eps=1e-5)
        self.down0_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.down0_dropout = nn.Dropout(dropout_rate)
        
        # Down 1
        self.down1_conv1 = nn.Conv2d(nb_filters, nb_filters*2, kernel_size=3, padding=1)
        self.down1_bn1 = nn.BatchNorm2d(nb_filters*2, momentum=0.1, eps=1e-5)
        self.down1_conv2 = nn.Conv2d(nb_filters*2, nb_filters*2, kernel_size=3, padding=1)
        self.down1_bn2 = nn.BatchNorm2d(nb_filters*2, momentum=0.1, eps=1e-5)
        self.down1_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.down1_dropout = nn.Dropout(dropout_rate)
        
        # Center
        self.center_conv1 = nn.Conv2d(nb_filters*2, nb_filters*4, kernel_size=3, padding=1)
        self.center_bn1 = nn.BatchNorm2d(nb_filters*4, momentum=0.1, eps=1e-5)
        self.center_conv2 = nn.Conv2d(nb_filters*4, nb_filters*4, kernel_size=3, padding=1)
        self.center_bn2 = nn.BatchNorm2d(nb_filters*4, momentum=0.1, eps=1e-5)
        self.center_dropout = nn.Dropout(dropout_rate)
        
        # Decoder path
        self.up1_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up1_conv1 = nn.Conv2d(nb_filters*4 + nb_filters*2, nb_filters*2, kernel_size=3, padding=1)
        self.up1_bn1 = nn.BatchNorm2d(nb_filters*2, momentum=0.1, eps=1e-5)
        self.up1_conv2 = nn.Conv2d(nb_filters*2, nb_filters*2, kernel_size=3, padding=1)
        self.up1_bn2 = nn.BatchNorm2d(nb_filters*2, momentum=0.1, eps=1e-5)
        self.up1_dropout = nn.Dropout(dropout_rate)
        
        # Up 0
        self.up0_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up0_conv1 = nn.Conv2d(nb_filters*2 + nb_filters, nb_filters, kernel_size=3, padding=1)
        self.up0_bn1 = nn.BatchNorm2d(nb_filters, momentum=0.1, eps=1e-5)
        self.up0_conv2 = nn.Conv2d(nb_filters, nb_filters, kernel_size=3, padding=1)
        self.up0_bn2 = nn.BatchNorm2d(nb_filters, momentum=0.1, eps=1e-5)
        self.up0_dropout = nn.Dropout(dropout_rate)
        
        # Output layer
        self.outputs = nn.Conv2d(nb_filters, 1, kernel_size=1)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights to match TensorFlow behavior"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _crop_or_pad_to_match(self, x, target_tensor):
        """
        Crop or pad tensor x to match the spatial dimensions of target_tensor.
        
        Args:
            x: Tensor to adjust
            target_tensor: Tensor with the target dimensions
            
        Returns:
            Tensor with adjusted dimensions
        """
        if x.shape[2:] == target_tensor.shape[2:]:
            return x  # No adjustment needed
        
        # Get target dimensions
        _, _, target_h, target_w = target_tensor.shape
        
        # Get current dimensions
        _, _, curr_h, curr_w = x.shape
        
        # Check if we need to crop or pad
        if curr_h > target_h:  # Need to crop height
            # Calculate how much to crop from top and bottom (center crop)
            crop_h = curr_h - target_h
            crop_top = crop_h // 2
            x = x[:, :, crop_top:crop_top + target_h, :]
        elif curr_h < target_h:  # Need to pad height
            # Calculate how much to pad on top and bottom (center pad)
            pad_h = target_h - curr_h
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            x = F.pad(x, (0, 0, pad_top, pad_bottom))
        
        # Now adjust width same way
        _, _, curr_h, curr_w = x.shape  # Get updated dimensions after height adjustment
        
        if curr_w > target_w:  # Need to crop width
            crop_w = curr_w - target_w
            crop_left = crop_w // 2
            x = x[:, :, :, crop_left:crop_left + target_w]
        elif curr_w < target_w:  # Need to pad width
            pad_w = target_w - curr_w
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            x = F.pad(x, (pad_left, pad_right, 0, 0))
        
        return x

    def forward(self, x):
        """
        Forward pass with robust size handling for skip connections.
        """
        # Store input size for final output sizing
        input_size = (x.size(2), x.size(3))
        
        # Encoder
        down0 = F.relu(self.down0_bn1(self.down0_conv1(x)))
        down0 = self.down0_dropout(down0)
        down0 = F.relu(self.down0_bn2(self.down0_conv2(down0)))
        down0 = self.down0_dropout(down0)
        down0_pool = self.down0_pool(down0)
        
        # Down 1
        down1 = F.relu(self.down1_bn1(self.down1_conv1(down0_pool)))
        down1 = self.down1_dropout(down1)
        down1 = F.relu(self.down1_bn2(self.down1_conv2(down1)))
        down1 = self.down1_dropout(down1)
        down1_pool = self.down1_pool(down1)
        
        # Center
        center = F.relu(self.center_bn1(self.center_conv1(down1_pool)))
        center = self.center_dropout(center)
        center = F.relu(self.center_bn2(self.center_conv2(center)))
        center = self.center_dropout(center)
        
        # Decoder
        # Up 1 - Handle potential size mismatch
        up1 = self.up1_upsample(center)
        
        # Adjust up1 to match down1's spatial dimensions
        up1 = self._crop_or_pad_to_match(up1, down1)
        
        # Now concatenate safely
        up1 = torch.cat([down1, up1], dim=1)
        up1 = F.relu(self.up1_bn1(self.up1_conv1(up1)))
        up1 = self.up1_dropout(up1)
        up1 = F.relu(self.up1_bn2(self.up1_conv2(up1)))
        up1 = self.up1_dropout(up1)
        
        # Up 0 - Handle potential size mismatch
        up0 = self.up0_upsample(up1)
        
        # Adjust up0 to match down0's spatial dimensions
        up0 = self._crop_or_pad_to_match(up0, down0)
        
        # Now concatenate safely
        up0 = torch.cat([down0, up0], dim=1)
        up0 = F.relu(self.up0_bn1(self.up0_conv1(up0)))
        up0 = self.up0_dropout(up0)
        up0 = F.relu(self.up0_bn2(self.up0_conv2(up0)))
        up0 = self.up0_dropout(up0)
        
        # Output
        outputs = self.outputs(up0)
        
        # Ensure output matches input spatial dimensions if needed
        if outputs.size(2) != input_size[0] or outputs.size(3) != input_size[1]:
            outputs = F.interpolate(outputs, size=input_size, 
                                  mode='bilinear', align_corners=True)
        
        # Apply sigmoid if requested
        if self.use_sigmoid:
            outputs = torch.sigmoid(outputs)
            outputs = torch.clamp(outputs, min=0, max=1)
            
        return outputs

def create_model(settings, data_shape, printSummary=False):
    """Create a standard UNet model compatible with mixed precision"""
    nb_filters, firstConvSize = settings["nb_filters"], settings["first_conv_size"]
    
    # Determine input channels
    if len(data_shape) == 4:  # (batch, time, H, W)
        input_channels = data_shape[1]
    elif len(data_shape) == 3:  # (time, H, W)
        input_channels = data_shape[0]
    else:  # (H, W)
        input_channels = 1
    
    # Update settings
    settings["nb_input_channels"] = input_channels
    
    # Determine if we should use sigmoid in the model
    # For mixed precision, we generally don't want to use sigmoid in the model
    use_sigmoid = False
    
    # Create model
    dropout_rate = settings.get("dropout_rate", 0)
    model = UNet(
        in_channels=input_channels,
        nb_filters=nb_filters, 
        first_conv_size=firstConvSize, 
        dropout_rate=dropout_rate,
        use_sigmoid=use_sigmoid
    )
    
    # Define optimizer with weight decay matching TensorFlow
    weight_decay = 1e-4  # Match TensorFlow's l2 regularization
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=settings["initial_learning_rate"],
        weight_decay=weight_decay
    )
    
    # Use loss functions that work with logits
    loss_fn = get_loss_function(settings)
    
    # Define metrics
    metrics = {
        'mse': nn.MSELoss()
    }
    
    if printSummary:
        logger.info(str(model))
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Total parameters: {total_params}")
    
    return model, optimizer, loss_fn, metrics

def get_loss_function(settings=None):
    if settings is None:
        return nn.BCELoss()
    
    loss = settings["loss"]
    if loss == "soft_focal":
        logger.info('Using soft focal loss')
        return SoftFocalLoss(settings['weight']['alpha'], settings['weight']['gamma'])
    elif loss == "mse":
        return nn.MSELoss()
    elif loss == "binary_crossentropy":
        return nn.BCELoss()
    else:
        raise NotImplementedError(f"{loss} has not been implemented as loss function in this framework.")

def get_model_generator(model_name='unet'):
    from deep_events.torch import lstm_models
    if model_name == 'unet':
        return create_model
    elif model_name == 'bottleneck_lstm':
        return lstm_models.create_bottleneck_convlstm_model


# # def create_model(settings, data_shape, printSummary=False):
#     """Create a standard UNet model for 2D or temporal data."""
#     nb_filters, firstConvSize = settings["nb_filters"], settings["first_conv_size"]
    
#     # Correctly determine input channels based on data shape
#     # This is crucial to handle both [batch, channels, H, W] and [channels, H, W] shapes
    
#     logger.info(f"Original data shape: {data_shape}")
    
#     # Case 1: Temporal input with batch dimension [batch, time, H, W]
#     if len(data_shape) == 4:
#         input_channels = data_shape[1]
#         logger.info(f"4D input detected (likely [batch, channels/time, H, W]): Using {input_channels} input channels")
    
#     # Case 2: Single sample with channels [channels, H, W]
#     elif len(data_shape) == 3:
#         # NOTE: This is the key fix - don't mistakenly interpret the first dimension as batch size
#         input_channels = data_shape[0]
#         logger.info(f"3D input detected [channels, H, W]: Using {input_channels} input channels")
    
#     # Case 3: Just a 2D image [H, W]
#     else:
#         input_channels = 1
#         logger.info(f"2D input detected [H, W]: Using {input_channels} input channels")
    
#     # Update settings with the correct channel count
#     settings["nb_input_channels"] = input_channels
    
#     # Log all details for debugging
#     logger.info(f"Creating UNet with:")
#     logger.info(f"  - Input channels: {input_channels}")
#     logger.info(f"  - Number of filters: {nb_filters}")
#     logger.info(f"  - First conv size: {firstConvSize}")
    
#     # Determine final activation
#     final_activation = "sigmoid"
    
#     # Create model
#     dropout_rate = settings.get("dropout_rate", 0)
#     model = UNet(
#         in_channels=input_channels,
#         nb_filters=nb_filters, 
#         first_conv_size=firstConvSize, 
#         dropout_rate=dropout_rate,
#         final_activation=final_activation
#     )
    
#     # Define optimizer
#     optimizer = torch.optim.Adam(model.parameters(), lr=settings["initial_learning_rate"], weight_decay=1e-4)
    
#     # Define loss
#     loss_fn = get_loss_function(settings)
    
#     # Define metrics
#     metrics = {
#         'mse': nn.MSELoss()
#     }
    
#     if printSummary:
#         logger.info(str(model))
#         total_params = sum(p.numel() for p in model.parameters())
#         logger.info(f"Total parameters: {total_params}")
        
#         # Print first conv layer details to verify dimensions
#         first_conv = model.down0_conv1
#         logger.info(f"First conv layer weight shape: {first_conv.weight.shape}")
#         logger.info(f"Should expect input with {input_channels} channels")
    
#     return model, optimizer, loss_fn, metrics