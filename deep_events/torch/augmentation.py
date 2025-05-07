import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Union

def create_2d_displacement(
    shape: Tuple[int, int],
    alpha: float = 10.0,
    sigma: float = 3.0,
    device: Optional[torch.device] = None,
    random_state: Optional[np.random.RandomState] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Creates a random 2D displacement field for elastic transforms using PyTorch.
    
    Args:
        shape: (H, W) shape for displacement
        alpha: Magnitude scaling for displacement
        sigma: Gaussian filter sigma for smoothing displacement fields
        device: PyTorch device to create tensors on
        random_state: Optional, for reproducibility with NumPy
        
    Returns:
        Tuple of displacement fields (dx, dy) as PyTorch tensors
    """
    H, W = shape
    
    # Create random displacement fields
    if random_state is not None:
        # Use NumPy with random_state for reproducibility
        dx_np = random_state.rand(H, W) * 2 - 1
        dy_np = random_state.rand(H, W) * 2 - 1
        dx = torch.from_numpy(dx_np).float()
        dy = torch.from_numpy(dy_np).float()
    else:
        # Use PyTorch's random functions
        dx = torch.rand(H, W, device=device) * 2 - 1
        dy = torch.rand(H, W, device=device) * 2 - 1
    
    # Apply Gaussian blur to smooth the fields
    # Convert to batched format for conv2d
    dx = dx.unsqueeze(0).unsqueeze(0)
    dy = dy.unsqueeze(0).unsqueeze(0)
    
    # Create Gaussian kernel
    kernel_size = int(4 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure odd kernel size
    
    # Create a 1D Gaussian kernel
    coords = torch.arange(kernel_size, device=device) - (kernel_size - 1) / 2
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    # Create 2D separable kernel
    kernel_x = kernel_1d.view(1, 1, kernel_size, 1).repeat(1, 1, 1, 1)
    kernel_y = kernel_1d.view(1, 1, 1, kernel_size).repeat(1, 1, 1, 1)
    
    # Apply separable convolution (faster than full 2D)
    dx = F.conv2d(dx, kernel_x, padding=(kernel_size//2, 0))
    dx = F.conv2d(dx, kernel_y, padding=(0, kernel_size//2))
    
    dy = F.conv2d(dy, kernel_x, padding=(kernel_size//2, 0))
    dy = F.conv2d(dy, kernel_y, padding=(0, kernel_size//2))
    
    # Scale by alpha
    dx = dx.squeeze() * alpha
    dy = dy.squeeze() * alpha
    
    return dx, dy

def elastic_transform_2d(
    image: torch.Tensor,
    alpha: float = 10.0,
    sigma: float = 4.0,
    order: int = 1,
    random_state: Optional[np.random.RandomState] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Elastic deformation of 2D images using PyTorch.
    
    Args:
        image: Input image tensor of shape (H, W) or (C, H, W)
        alpha: Scale factor for deformation
        sigma: Smoothing factor
        order: Interpolation order (1=linear, 0=nearest)
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (transformed image, dx, dy)
    """
    is_batched = False
    
    # Handle different input dimensions
    if len(image.shape) == 4:  # (B, C, H, W)
        is_batched = True
        B, C, H, W = image.shape
    elif len(image.shape) == 3:  # (C, H, W)
        C, H, W = image.shape
    else:  # (H, W)
        H, W = image.shape
        image = image.unsqueeze(0)  # Add channel dim
        C = 1
    
    # Add batch dimension if not batched
    if not is_batched:
        image = image.unsqueeze(0)
    
    # Create displacement fields
    device = image.device
    dx, dy = create_2d_displacement((H, W), alpha, sigma, device, random_state)
    
    # Create sampling grid
    y_coords, x_coords = torch.meshgrid(torch.arange(H, device=device), 
                                        torch.arange(W, device=device),
                                        indexing="ij")
    
    # Apply displacement
    y_mapped = y_coords + dy
    x_mapped = x_coords + dx
    
    # Normalize coordinates to [-1, 1] for grid_sample
    y_mapped = 2.0 * (y_mapped / (H - 1)) - 1.0
    x_mapped = 2.0 * (x_mapped / (W - 1)) - 1.0
    
    # Stack coordinates into a grid
    grid = torch.stack([x_mapped, y_mapped], dim=-1)
    grid = grid.unsqueeze(0).repeat(image.size(0), 1, 1, 1)
    
    # Apply grid sampling with specified interpolation mode
    mode = 'bilinear' if order == 1 else 'nearest'
    transformed = F.grid_sample(
        image, 
        grid, 
        mode=mode, 
        padding_mode='reflection',
        align_corners=True
    )
    
    # Return to original dimensions
    if not is_batched:
        transformed = transformed.squeeze(0)
        if C == 1 and len(image.shape) == 3:
            transformed = transformed.squeeze(0)
    
    return transformed, dx, dy

def elastic_transform_3d_as_2d_slices(
    volume: torch.Tensor,
    alpha: float = 10.0,
    sigma: float = 4.0,
    order: int = 1,
    random_state: Optional[np.random.RandomState] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply elastic deformation to a 3D volume by treating it as a stack of 2D slices.
    The same deformation field is applied to all slices.
    
    Args:
        volume: Input volume tensor of shape (T, H, W) or (B, T, H, W)
        alpha: Scale factor for deformation
        sigma: Smoothing factor
        order: Interpolation order (1=linear, 0=nearest)
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (transformed volume, dx, dy)
    """
    is_batched = False
    
    # Handle different input dimensions
    if len(volume.shape) == 4:  # (B, T, H, W)
        is_batched = True
        B, T, H, W = volume.shape
    else:  # (T, H, W)
        T, H, W = volume.shape
    
    # Create displacement fields on the first slice
    device = volume.device
    dx, dy = create_2d_displacement((H, W), alpha, sigma, device, random_state)
    
    # Create sampling grid
    y_coords, x_coords = torch.meshgrid(torch.arange(H, device=device), 
                                        torch.arange(W, device=device),
                                        indexing="ij")
    
    # Apply displacement
    y_mapped = y_coords + dy
    x_mapped = x_coords + dx
    
    # Normalize coordinates to [-1, 1] for grid_sample
    y_mapped = 2.0 * (y_mapped / (H - 1)) - 1.0
    x_mapped = 2.0 * (x_mapped / (W - 1)) - 1.0
    
    # Stack coordinates into a grid
    grid = torch.stack([x_mapped, y_mapped], dim=-1)
    
    # Reshape volume for grid_sample
    if is_batched:
        # Reshape from (B, T, H, W) to (B*T, 1, H, W)
        volume_reshaped = volume.reshape(B*T, 1, H, W)
        grid_expanded = grid.unsqueeze(0).repeat(B*T, 1, 1, 1)
    else:
        # Reshape from (T, H, W) to (T, 1, H, W)
        volume_reshaped = volume.unsqueeze(1)
        grid_expanded = grid.unsqueeze(0).repeat(T, 1, 1, 1)
    
    # Apply grid sampling with specified interpolation mode
    mode = 'bilinear' if order == 1 else 'nearest'
    transformed = F.grid_sample(
        volume_reshaped, 
        grid_expanded, 
        mode=mode, 
        padding_mode='reflection',
        align_corners=True
    )
    
    # Reshape back to original format
    if is_batched:
        transformed = transformed.reshape(B, T, H, W)
    else:
        transformed = transformed.squeeze(1)  # Back to (T, H, W)
    
    return transformed, dx, dy

def warp_2d_slice(
    image: torch.Tensor,
    dx: torch.Tensor,
    dy: torch.Tensor,
    order: int = 1
) -> torch.Tensor:
    """
    Warp a 2D slice using pre-computed displacement fields.
    
    Args:
        image: Input image tensor of shape (H, W) or (C, H, W)
        dx: X displacement field of shape (H, W)
        dy: Y displacement field of shape (H, W)
        order: Interpolation order (1=linear, 0=nearest)
        
    Returns:
        Warped image
    """
    is_batched = False
    device = image.device
    
    # Handle different input dimensions
    if len(image.shape) == 4:  # (B, C, H, W)
        is_batched = True
        B, C, H, W = image.shape
    elif len(image.shape) == 3:  # (C, H, W)
        C, H, W = image.shape
    else:  # (H, W)
        H, W = image.shape
        image = image.unsqueeze(0)  # Add channel dim
        C = 1
    
    # Add batch dimension if not batched
    if not is_batched:
        image = image.unsqueeze(0)  # Add batch dim
    
    # Create sampling grid
    y_coords, x_coords = torch.meshgrid(torch.arange(H, device=device), 
                                        torch.arange(W, device=device),
                                        indexing="ij")
    
    # Apply displacement
    y_mapped = y_coords + dy
    x_mapped = x_coords + dx
    
    # Normalize coordinates to [-1, 1] for grid_sample
    y_mapped = 2.0 * (y_mapped / (H - 1)) - 1.0
    x_mapped = 2.0 * (x_mapped / (W - 1)) - 1.0
    
    # Stack coordinates into a grid
    grid = torch.stack([x_mapped, y_mapped], dim=-1)
    grid = grid.unsqueeze(0).repeat(image.size(0), 1, 1, 1)
    
    # Apply grid sampling with specified interpolation mode
    mode = 'bilinear' if order == 1 else 'nearest'
    transformed = F.grid_sample(
        image, 
        grid, 
        mode=mode, 
        padding_mode='reflection',
        align_corners=True
    )
    
    # Return to original dimensions
    if not is_batched:
        transformed = transformed.squeeze(0)
        if C == 1 and len(image.shape) == 3:
            transformed = transformed.squeeze(0)
    
    return transformed