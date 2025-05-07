import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Union, Optional
import numpy as np

from deep_events.torch.training_functions import get_loss_function


class ConvLSTMCell(nn.Module):
    """
    ConvLSTM cell implementation for PyTorch.
    """
    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True):
        """
        Initialize ConvLSTM cell.
        
        Parameters:
        -----------
        input_channels: int
            Number of channels in input tensor
        hidden_channels: int
            Number of channels in hidden state
        kernel_size: int or tuple
            Size of the convolutional kernel
        bias: bool
            Whether to add bias or not
        """
        super(ConvLSTMCell, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        # Combined convolution for all gates
        self.conv = nn.Conv2d(
            in_channels=self.input_channels + self.hidden_channels,
            out_channels=4 * self.hidden_channels,  # For input, forget, cell, and output gates
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, input_tensor, hidden_state=None):
        """
        Forward pass of ConvLSTM cell.
        
        Parameters:
        -----------
        input_tensor: torch.Tensor
            Input tensor of shape (batch, channels, height, width)
        hidden_state: Tuple[torch.Tensor, torch.Tensor]
            Previous hidden and cell states
            
        Returns:
        --------
        hidden_state: Tuple[torch.Tensor, torch.Tensor]
            New hidden and cell states
        """
        batch_size, _, height, width = input_tensor.size()
        
        # Initialize hidden state if not provided
        if hidden_state is None:
            h_state = torch.zeros(batch_size, self.hidden_channels, height, width, device=input_tensor.device)
            c_state = torch.zeros(batch_size, self.hidden_channels, height, width, device=input_tensor.device)
        else:
            h_state, c_state = hidden_state
        
        # Concatenate input and previous hidden state
        combined = torch.cat([input_tensor, h_state], dim=1)
        
        # Calculate all gates in one convolution
        conv_output = self.conv(combined)
        
        # Split convolution output into gates
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_output, self.hidden_channels, dim=1)
        
        # Apply activations
        i = torch.sigmoid(cc_i)  # Input gate
        f = torch.sigmoid(cc_f)  # Forget gate
        o = torch.sigmoid(cc_o)  # Output gate
        g = torch.tanh(cc_g)     # Cell input
        
        # Update cell state
        c_next = f * c_state + i * g
        
        # Update hidden state
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next


class ConvLSTM(nn.Module):
    """
    ConvLSTM layer with multiple timesteps.
    """
    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers=1, 
                 batch_first=True, bias=True, return_all_layers=False):
        """
        Initialize ConvLSTM.
        
        Parameters:
        -----------
        input_channels: int
            Number of channels in input tensor
        hidden_channels: int or List[int]
            Number of channels in hidden state for each layer
        kernel_size: int or List[int]
            Size of the convolutional kernel for each layer
        num_layers: int
            Number of ConvLSTM layers
        batch_first: bool
            If True, input and output tensors are (batch, time, channels, height, width)
        bias: bool
            Whether to add bias in ConvLSTM cells
        return_all_layers: bool
            If True, return hidden states of all layers, else only the last layer
        """
        super(ConvLSTM, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels if isinstance(hidden_channels, list) else [hidden_channels] * num_layers
        self.kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size] * num_layers
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        # Create ConvLSTM cells
        cell_list = []
        for i in range(self.num_layers):
            cur_input_channels = self.input_channels if i == 0 else self.hidden_channels[i-1]
            cell_list.append(
                ConvLSTMCell(
                    input_channels=cur_input_channels,
                    hidden_channels=self.hidden_channels[i],
                    kernel_size=self.kernel_size[i],
                    bias=self.bias
                )
            )
        
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Forward pass of ConvLSTM.
        
        Parameters:
        -----------
        input_tensor: torch.Tensor
            Input tensor of shape (batch, time, channels, height, width) if batch_first=True
            else (time, batch, channels, height, width)
        hidden_state: List[Tuple[torch.Tensor, torch.Tensor]]
            List of previous hidden and cell states for each layer
            
        Returns:
        --------
        layer_output_list: List[torch.Tensor]
            List of output tensors for each layer
        last_state_list: List[Tuple[torch.Tensor, torch.Tensor]]
            List of final hidden and cell states for each layer
        """
        # If batch_first, transpose to (time, batch, channels, height, width)
        if self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        
        # Get dimensions
        time_steps, batch_size, _, height, width = input_tensor.size()
        
        # Initialize hidden states if not provided
        if hidden_state is None:
            hidden_state = []
            for i in range(self.num_layers):
                device = input_tensor.device
                h = torch.zeros(batch_size, self.hidden_channels[i], height, width, device=device)
                c = torch.zeros(batch_size, self.hidden_channels[i], height, width, device=device)
                hidden_state.append((h, c))
        
        layer_output_list = []
        last_state_list = []
        
        # Iterate through layers
        cur_layer_input = input_tensor
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            
            # Iterate through time steps
            for t in range(time_steps):
                h, c = self.cell_list[layer_idx](cur_layer_input[t], (h, c))
                output_inner.append(h)
            
            # Stack time steps
            layer_output = torch.stack(output_inner, dim=0)
            cur_layer_input = layer_output
            
            layer_output_list.append(layer_output)
            last_state_list.append((h, c))
        
        # Return outputs based on return_all_layers flag
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]
        
        # If batch_first, transpose back to (batch, time, channels, height, width)
        if self.batch_first:
            layer_output_list = [layer_output.permute(1, 0, 2, 3, 4) for layer_output in layer_output_list]
        
        return layer_output_list, last_state_list


class BottleneckConvLSTM(nn.Module):
    """
    Bottleneck ConvLSTM model for segmentation.
    """
    def __init__(self, in_channels, nb_filters=16, first_conv_size=12, dropout_rate=0, 
                 final_activation="sigmoid"):
        """
        Initialize Bottleneck ConvLSTM.
        
        Parameters:
        -----------
        in_channels: int
            Number of input channels (time steps)
        nb_filters: int
            Number of base filters
        first_conv_size: int
            Size of first convolution kernel
        dropout_rate: float
            Dropout rate
        final_activation: str
            Activation for final layer
        """
        super(BottleneckConvLSTM, self).__init__()
        
        self.in_channels = in_channels
        self.nb_filters = nb_filters
        self.dropout_rate = dropout_rate
        self.final_activation = final_activation
        
        # Encoder path
        # Down 0
        self.down0_conv1 = nn.Conv2d(1, nb_filters, kernel_size=first_conv_size, padding=first_conv_size//2)
        self.down0_bn1 = nn.BatchNorm2d(nb_filters, momentum=0.99, eps=0.001)
        self.down0_conv2 = nn.Conv2d(nb_filters, nb_filters, kernel_size=first_conv_size, padding=first_conv_size//2)
        self.down0_bn2 = nn.BatchNorm2d(nb_filters, momentum=0.99, eps=0.001)
        self.down0_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.down0_dropout = nn.Dropout(dropout_rate)
        
        # Down 1
        self.down1_conv1 = nn.Conv2d(nb_filters, nb_filters*2, kernel_size=3, padding=1)
        self.down1_bn1 = nn.BatchNorm2d(nb_filters*2, momentum=0.99, eps=0.001)
        self.down1_conv2 = nn.Conv2d(nb_filters*2, nb_filters*2, kernel_size=3, padding=1)
        self.down1_bn2 = nn.BatchNorm2d(nb_filters*2, momentum=0.99, eps=0.001)
        self.down1_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.down1_dropout = nn.Dropout(dropout_rate)
        
        # Center ConvLSTM
        self.center_convlstm = ConvLSTM(
            input_channels=nb_filters*2,
            hidden_channels=nb_filters*4,
            kernel_size=3,
            num_layers=1,
            batch_first=True,
            bias=True,
            return_all_layers=False
        )
        
        self.center_bn = nn.BatchNorm2d(nb_filters*4, momentum=0.99, eps=0.001)
        self.center_dropout = nn.Dropout(dropout_rate)
        
        # Decoder path
        # Up 1
        self.up1_conv1 = nn.Conv2d(nb_filters*4 + nb_filters*2, nb_filters*2, kernel_size=3, padding=1)
        self.up1_bn1 = nn.BatchNorm2d(nb_filters*2, momentum=0.99, eps=0.001)
        self.up1_conv2 = nn.Conv2d(nb_filters*2, nb_filters*2, kernel_size=3, padding=1)
        self.up1_bn2 = nn.BatchNorm2d(nb_filters*2, momentum=0.99, eps=0.001)
        self.up1_dropout = nn.Dropout(dropout_rate)
        
        # Up 0
        self.up0_conv1 = nn.Conv2d(nb_filters*2 + nb_filters, nb_filters, kernel_size=3, padding=1)
        self.up0_bn1 = nn.BatchNorm2d(nb_filters, momentum=0.99, eps=0.001)
        self.up0_conv2 = nn.Conv2d(nb_filters, nb_filters, kernel_size=3, padding=1)
        self.up0_bn2 = nn.BatchNorm2d(nb_filters, momentum=0.99, eps=0.001)
        self.up0_dropout = nn.Dropout(dropout_rate)
        
        # Output
        self.outputs = nn.Conv2d(nb_filters, 1, kernel_size=1, padding=0)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass of Bottleneck ConvLSTM.
        
        Parameters:
        -----------
        x: torch.Tensor
            Input tensor of shape (batch, time, height, width) or (batch, time, channels, height, width)
            
        Returns:
        --------
        outputs: torch.Tensor
            Output tensor of shape (batch, 1, height, width)
        """
        # Ensure input is in the right format for processing
        if len(x.shape) == 4:  # (batch, time, height, width)
            # Add channel dimension
            x = x.unsqueeze(2)  # (batch, time, 1, height, width)
        
        input_size = (x.size(3), x.size(4))
        batch_size, time_steps = x.size(0), x.size(1)
        
        # Process each time step through encoder
        down0_list = []
        down1_list = []
        down1_pool_list = []
        
        for t in range(time_steps):
            # Extract time step t
            x_t = x[:, t, :, :, :]
            if x_t.size(1) == 1:
                x_t = x_t.squeeze(1)  # Remove channel dim if it's 1
            
            # Down 0
            down0 = self.down0_conv1(x_t)
            down0 = self.down0_bn1(down0)
            down0 = F.relu(down0)
            down0 = self.down0_dropout(down0)
            down0 = self.down0_conv2(down0)
            down0 = self.down0_bn2(down0)
            down0 = F.relu(down0)
            down0 = self.down0_dropout(down0)
            down0_list.append(down0)
            down0_pool = self.down0_pool(down0)
            
            # Down 1
            down1 = self.down1_conv1(down0_pool)
            down1 = self.down1_bn1(down1)
            down1 = F.relu(down1)
            down1 = self.down1_dropout(down1)
            down1 = self.down1_conv2(down1)
            down1 = self.down1_bn2(down1)
            down1 = F.relu(down1)
            down1 = self.down1_dropout(down1)
            down1_list.append(down1)
            down1_pool = self.down1_pool(down1)
            down1_pool_list.append(down1_pool)
        
        # Stack bottleneck features for ConvLSTM
        bottleneck_features = torch.stack(down1_pool_list, dim=1)  # (batch, time, channels, height, width)
        
        # Apply ConvLSTM to bottleneck
        lstm_output, _ = self.center_convlstm(bottleneck_features)
        lstm_output = lstm_output[0]  # Get output from last layer
        
        # Use last time step for decoding
        center = lstm_output[:, -1]  # (batch, channels, height, width)
        center = self.center_bn(center)
        center = F.relu(center)
        center = self.center_dropout(center)
        
        # Get features from last time step for skip connections
        down0 = down0_list[-1]
        down1 = down1_list[-1]
        
        # Decoder path
        # Up 1
        up1 = F.interpolate(center, size=(down1.size(2), down1.size(3)), mode='bilinear', align_corners=True)
        up1 = torch.cat([down1, up1], dim=1)  # Skip connection
        up1 = self.up1_conv1(up1)
        up1 = self.up1_bn1(up1)
        up1 = F.relu(up1)
        up1 = self.up1_dropout(up1)
        up1 = self.up1_conv2(up1)
        up1 = self.up1_bn2(up1)
        up1 = F.relu(up1)
        up1 = self.up1_dropout(up1)
        
        # Up 0
        up0 = F.interpolate(up1, size=(down0.size(2), down0.size(3)), mode='bilinear', align_corners=True)
        up0 = torch.cat([down0, up0], dim=1)  # Skip connection
        up0 = self.up0_conv1(up0)
        up0 = self.up0_bn1(up0)
        up0 = F.relu(up0)
        up0 = self.up0_dropout(up0)
        up0 = self.up0_conv2(up0)
        up0 = self.up0_bn2(up0)
        up0 = F.relu(up0)
        up0 = self.up0_dropout(up0)
        
        # Output
        outputs = self.outputs(up0)
        if self.final_activation == 'sigmoid':
            outputs = torch.sigmoid(outputs)
        
        # Ensure output size matches input size
        outputs = F.interpolate(outputs, size=input_size, mode='bilinear', align_corners=True)
        outputs = torch.clamp(outputs, min=0, max=1)
            
        return outputs


def create_bottleneck_convlstm_model(settings, data_shape, printSummary=False):
    """
    Create a Bottleneck ConvLSTM model based on settings.
    
    Parameters:
    -----------
    settings: dict
        Dictionary containing model settings
    data_shape: tuple
        Shape of the input data
    printSummary: bool
        Whether to print model summary
        
    Returns:
    --------
    model, optimizer, loss_fn, metrics
    """
    nb_filters, firstConvSize = settings["nb_filters"], settings["first_conv_size"]
    
    # Determine input channels (time steps)
    if len(data_shape) == 5:  # (batch, time, channels, height, width)
        in_channels = data_shape[1]
    elif len(data_shape) == 4:  # (batch, channels, height, width)
        in_channels = data_shape[1]
    else:
        raise ValueError(f"Unexpected data shape: {data_shape}")
    
    settings["nb_input_channels"] = in_channels
    
    # Determine final activation
    if settings["loss"] == "soft_dice":
        final_activation = "sigmoid"
    else:
        final_activation = "sigmoid"
    
    # Create model
    dropout_rate = settings.get("dropout_rate", 0)
    model = BottleneckConvLSTM(
        in_channels=in_channels,
        nb_filters=nb_filters,
        first_conv_size=firstConvSize,
        dropout_rate=dropout_rate,
        final_activation=final_activation
    )
    
    # Define optimizer with weight decay (L2 regularization)
    weight_decay = 1e-4  # Equivalent to TensorFlow's l2(1e-4)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=settings["initial_learning_rate"],
        weight_decay=weight_decay
    )
    
    # Define loss
    loss_fn = get_loss_function(settings)
    
    # Define metrics (these will be used during training)
    metrics = {
        'mse': nn.MSELoss(),
        'wmse': WMSELoss(pos_weight=3.0),
        'wbce': WBCELoss(pos_weight=3.0)
    }
    
    if printSummary:
        print(model)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params}")
    
    return model, optimizer, loss_fn, metrics


class DeepTemporalUNet(nn.Module):
    """
    Deep Temporal UNet model with multiple ConvLSTM layers.
    """
    def __init__(self, in_channels, nb_filters=16, first_conv_size=12, dropout_rate=0,
                 final_activation="sigmoid"):
        """
        Initialize Deep Temporal UNet.
        
        Parameters:
        -----------
        in_channels: int
            Number of input channels (time steps)
        nb_filters: int
            Number of base filters
        first_conv_size: int
            Size of first convolution kernel
        dropout_rate: float
            Dropout rate
        final_activation: str
            Activation for final layer
        """
        super(DeepTemporalUNet, self).__init__()
        
        self.in_channels = in_channels
        self.nb_filters = nb_filters
        self.dropout_rate = dropout_rate
        self.final_activation = final_activation
        
        # Down 0 ConvLSTM - Process all time steps
        self.down0_convlstm = ConvLSTM(
            input_channels=1,
            hidden_channels=nb_filters,
            kernel_size=first_conv_size,
            num_layers=1,
            batch_first=True,
            bias=True,
            return_all_layers=False
        )
        self.down0_bn = nn.BatchNorm2d(nb_filters, momentum=0.99, eps=0.001)
        self.down0_dropout = nn.Dropout(dropout_rate)
        self.down0_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Down 1 ConvLSTM
        self.down1_convlstm = ConvLSTM(
            input_channels=nb_filters,
            hidden_channels=nb_filters*2,
            kernel_size=3,
            num_layers=1,
            batch_first=True,
            bias=True,
            return_all_layers=False
        )
        self.down1_bn = nn.BatchNorm2d(nb_filters*2, momentum=0.99, eps=0.001)
        self.down1_dropout = nn.Dropout(dropout_rate)
        self.down1_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Center ConvLSTM
        self.center_convlstm = ConvLSTM(
            input_channels=nb_filters*2,
            hidden_channels=nb_filters*4,
            kernel_size=3,
            num_layers=1,
            batch_first=True,
            bias=True,
            return_all_layers=False
        )
        self.center_bn = nn.BatchNorm2d(nb_filters*4, momentum=0.99, eps=0.001)
        self.center_dropout = nn.Dropout(dropout_rate)
        
        # Decoder - traditional convolutions from here
        # Up 1
        self.up1_conv1 = nn.Conv2d(nb_filters*4 + nb_filters*2, nb_filters*2, kernel_size=3, padding=1)
        self.up1_bn1 = nn.BatchNorm2d(nb_filters*2, momentum=0.99, eps=0.001)
        self.up1_conv2 = nn.Conv2d(nb_filters*2, nb_filters*2, kernel_size=3, padding=1)
        self.up1_bn2 = nn.BatchNorm2d(nb_filters*2, momentum=0.99, eps=0.001)
        self.up1_dropout = nn.Dropout(dropout_rate)
        
        # Up 0
        self.up0_conv1 = nn.Conv2d(nb_filters*2 + nb_filters, nb_filters, kernel_size=3, padding=1)
        self.up0_bn1 = nn.BatchNorm2d(nb_filters, momentum=0.99, eps=0.001)
        self.up0_conv2 = nn.Conv2d(nb_filters, nb_filters, kernel_size=3, padding=1)
        self.up0_bn2 = nn.BatchNorm2d(nb_filters, momentum=0.99, eps=0.001)
        self.up0_dropout = nn.Dropout(dropout_rate)
        
        # Output
        self.outputs = nn.Conv2d(nb_filters, 1, kernel_size=1, padding=0)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass of Deep Temporal UNet.
        
        Parameters:
        -----------
        x: torch.Tensor
            Input tensor of shape (batch, time, height, width) or (batch, time, channels, height, width)
            
        Returns:
        --------
        outputs: torch.Tensor
            Output tensor of shape (batch, 1, height, width)
        """
        # Ensure input is in the right format for processing
        if len(x.shape) == 4:  # (batch, time, height, width)
            # Add channel dimension
            x = x.unsqueeze(2)  # (batch, time, 1, height, width)
        
        input_size = (x.size(3), x.size(4))
        
        # Process entire sequence through ConvLSTM layers
        # Down 0
        down0_output, _ = self.down0_convlstm(x)
        down0_output = down0_output[0][:, -1]  # Get output from last time step
        down0 = self.down0_bn(down0_output)
        down0 = F.relu(down0)
        down0 = self.down0_dropout(down0)
        down0_pool = self.down0_pool(down0)
        
        # Reshape for next ConvLSTM
        down0_pool = down0_pool.unsqueeze(1)  # Add time dimension back (batch, 1, channels, height, width)
        
        # Down 1
        down1_output, _ = self.down1_convlstm(down0_pool)
        down1_output = down1_output[0][:, -1]  # Get output from last time step
        down1 = self.down1_bn(down1_output)
        down1 = F.relu(down1)
        down1 = self.down1_dropout(down1)
        down1_pool = self.down1_pool(down1)
        
        # Reshape for next ConvLSTM
        down1_pool = down1_pool.unsqueeze(1)  # Add time dimension back
        
        # Center
        center_output, _ = self.center_convlstm(down1_pool)
        center_output = center_output[0][:, -1]  # Get output from last time step
        center = self.center_bn(center_output)
        center = F.relu(center)
        center = self.center_dropout(center)
        
        # Decoder path
        # Up 1
        up1 = F.interpolate(center, size=(down1.size(2), down1.size(3)), mode='bilinear', align_corners=True)
        up1 = torch.cat([down1, up1], dim=1)  # Skip connection
        up1 = self.up1_conv1(up1)
        up1 = self.up1_bn1(up1)
        up1 = F.relu(up1)
        up1 = self.up1_dropout(up1)
        up1 = self.up1_conv2(up1)
        up1 = self.up1_bn2(up1)
        up1 = F.relu(up1)
        up1 = self.up1_dropout(up1)
        
        # Up 0
        up0 = F.interpolate(up1, size=(down0.size(2), down0.size(3)), mode='bilinear', align_corners=True)
        up0 = torch.cat([down0, up0], dim=1)  # Skip connection
        up0 = self.up0_conv1(up0)
        up0 = self.up0_bn1(up0)
        up0 = F.relu(up0)
        up0 = self.up0_dropout(up0)
        up0 = self.up0_conv2(up0)
        up0 = self.up0_bn2(up0)
        up0 = F.relu(up0)
        up0 = self.up0_dropout(up0)
        
        # Output
        outputs = self.outputs(up0)
        if self.final_activation == 'sigmoid':
            outputs = torch.sigmoid(outputs)
        
        # Ensure output size matches input size
        outputs = F.interpolate(outputs, size=input_size, mode='bilinear', align_corners=True)
        outputs = torch.clamp(outputs, min=0, max=1)
            
        return outputs


def create_deep_temporal_unet(settings, data_shape, printSummary=False):
    """
    Create a Deep Temporal UNet model based on settings.
    
    Parameters:
    -----------
    settings: dict
        Dictionary containing model settings
    data_shape: tuple
        Shape of the input data
    printSummary: bool
        Whether to print model summary
        
    Returns:
    --------
    model, optimizer, loss_fn, metrics
    """
    nb_filters, firstConvSize = settings["nb_filters"], settings["first_conv_size"]
    
    # Determine input channels (time steps)
    if len(data_shape) == 5:  # (batch, time, channels, height, width)
        in_channels = data_shape[1]
    elif len(data_shape) == 4:  # (batch, channels, height, width)
        in_channels = data_shape[1]
    else:
        raise ValueError(f"Unexpected data shape: {data_shape}")
    
    settings["nb_input_channels"] = in_channels
    
    # Determine final activation
    if settings["loss"] == "soft_dice":
        final_activation = "sigmoid"
    else:
        final_activation = "sigmoid"
    
    # Create model
    dropout_rate = settings.get("dropout_rate", 0)
    model = DeepTemporalUNet(
        in_channels=in_channels,
        nb_filters=nb_filters,
        first_conv_size=firstConvSize,
        dropout_rate=dropout_rate,
        final_activation=final_activation
    )
    
    # Define optimizer with weight decay (L2 regularization)
    weight_decay = 1e-4  # Equivalent to TensorFlow's l2(1e-4)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=settings["initial_learning_rate"],
        weight_decay=weight_decay
    )
    
    # Define loss
    loss_fn = get_loss_function(settings)
    
    # Define metrics (these will be used during training)
    metrics = {
        'mse': nn.MSELoss(),
        'wmse': WMSELoss(pos_weight=3.0),
        'wbce': WBCELoss(pos_weight=3.0)
    }
    
    if printSummary:
        print(model)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params}")
    
    return model, optimizer, loss_fn, metrics


if __name__ == "__main__":
    # Test the implementation
    import torch
    import torch.nn as nn
    import torch.optim as optim
    
    # Create dummy data
    batch_size = 4
    time_steps = 5
    height, width = 64, 64
    x = torch.rand(batch_size, time_steps, 1, height, width)
    
    # Create model
    model = BottleneckConvLSTM(
        in_channels=time_steps,
        nb_filters=16,
        first_conv_size=3,
        dropout_rate=0.2,
        final_activation='sigmoid'
    )
    
    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Check if output dimensions are correct
    assert output.shape == (batch_size, 1, height, width), "Output shape is incorrect"
    
    # Test with different input format
    x2 = torch.rand(batch_size, time_steps, height, width)
    output2 = model(x2)
    print(f"Input shape (alternative format): {x2.shape}")
    print(f"Output shape: {output2.shape}")
    
    # Check if output dimensions are correct
    assert output2.shape == (batch_size, 1, height, width), "Output shape is incorrect"
    
    print("All tests passed!")