import torch
import torch.nn.functional as F
import math
from evaluation.post_process import calculate_hr
import random

# Motion augmentation
# def jitter_affine(x, max_angles=2.0, max_translate=0.03):
#     """
#         Apply random affine transformations to the input tensor
#     """
#     N, D, C, H, W = x.shape
#     x = x.contiguous().view(N * D, C, H, W)
    
#     # Generate random angles and translations
#     angles = (torch.rand(N * D, device=x.device) * 2 - 1) * max_angles * math.pi / 180.0
#     tx = (torch.rand(N * D, device=x.device) * 2 - 1) * max_translate
#     ty = (torch.rand(N * D, device=x.device) * 2 - 1) * max_translate
    
#     affine_matrices = torch.zeros(N * D, 2, 3, device=x.device)
#     affine_matrices[:, 0, 0] = torch.cos(angles)
#     affine_matrices[:, 0, 1] = -torch.sin(angles)
#     affine_matrices[:, 0, 2] = tx
#     affine_matrices[:, 1, 0] = torch.sin(angles)
#     affine_matrices[:, 1, 1] = torch.cos(angles)
#     affine_matrices[:, 1, 2] = ty
    
#     grid = F.affine_grid(affine_matrices, x.size(), align_corners=False)
#     jittered_x = F.grid_sample(x, grid, mode='bilinear', padding_mode='reflection', align_corners=False)
    
#     jittered_x = jittered_x.view(N, D, C, H, W)
    
#     return jittered_x


def jitter_affine(x, max_angles=2.0, max_translate=0.03, knot_step=15):
    """
    Apply affine transformations with smooth temporal trajectory to the input tensor.
    """
    N, D, C, H, W = x.shape
    device = x.device

    # Number of knots along the temporal axis (1 knot every knot_step frames)
    num_knots = max(2, D // knot_step + 1)
    
    # Randomly sample angles and translations at knot points: shape (N, 3, num_knots)
    knots = torch.rand(N, 3, num_knots, device=device) * 2 - 1

    # Interpolate the knot parameters to get smooth parameters for each frame: shape (N, 3, D)
    smooth_params = F.interpolate(knots, size=D, mode='linear', align_corners=True)
    
    angles = (smooth_params[:, 0, :] * max_angles * math.pi / 180.0).reshape(N * D)
    tx = (smooth_params[:, 1, :] * max_translate).reshape(N * D)
    ty = (smooth_params[:, 2, :] * max_translate).reshape(N * D)
    
    x_flat = x.contiguous().view(N * D, C, H, W)
    
    affine_matrices = torch.zeros(N * D, 2, 3, device=device)
    affine_matrices[:, 0, 0] = torch.cos(angles)
    affine_matrices[:, 0, 1] = -torch.sin(angles)
    affine_matrices[:, 0, 2] = tx
    affine_matrices[:, 1, 0] = torch.sin(angles)
    affine_matrices[:, 1, 1] = torch.cos(angles)
    affine_matrices[:, 1, 2] = ty
    
    grid = F.affine_grid(affine_matrices, x_flat.size(), align_corners=False)
    jittered_x = F.grid_sample(x_flat, grid, mode='bilinear', padding_mode='reflection', align_corners=False)
    
    return jittered_x.view(N, D, C, H, W)


# Frame rate augmentation
def frame_drop(x, min_ratio=0.85):
    """
    Simulate frame drop by replacing each dropped frame
    with the value of the previous valid frame.
    The first and last frames are always kept.
    """
    N, D, C, H, W = x.shape
    device = x.device

    keep_ratio = torch.empty(1, device=device).uniform_(min_ratio, 0.95).item()
    D_keep = max(2, int(D * keep_ratio))

    # Randomly select frames, always keeping first and last frames
    inner_indices = torch.randperm(D - 2, device=device)[:D_keep - 2] + 1

    valid_t = torch.cat((
        torch.tensor([0], device=device, dtype=torch.long),
        inner_indices,
        torch.tensor([D - 1], device=device, dtype=torch.long)
    ))

    valid_t, _ = torch.sort(valid_t)

    # Construct frame indices using previous valid frame
    frame_indices = torch.arange(D, device=device)

    # Find the previous valid frame for each temporal position
    positions = torch.searchsorted(valid_t, frame_indices, right=True) - 1

    # Replace dropped frames with their previous valid frame
    source_indices = valid_t[positions]

    output_x = x[:, source_indices, :, :, :]

    return output_x


# def frame_drop(x, min_ratio=0.8):
#     """
#     Randomly drop frames from the input tensor.
#     """
#     N, D, C, H, W = x.shape

#     drop_ratio = torch.empty(1, device=x.device).uniform_(min_ratio, 0.95).item()
#     D_downsampled = int(D * drop_ratio)

#     inner_indices = torch.randperm(D - 2, device=x.device)[:D_downsampled - 2] + 1
#     first_index = torch.tensor([0], device=x.device)
#     last_index = torch.tensor([D - 1], device=x.device)

#     selected_indices = torch.cat((first_index, inner_indices, last_index))
#     selected_indices, _ = torch.sort(selected_indices)

#     sampled_x = x[:, selected_indices, :, :, :]  # (N, D_downsampled, C, H, W)

#     # Move temporal dimension to the last axis
#     sampled_x = sampled_x.permute(0, 2, 3, 4, 1)  # (N, C, H, W, D_downsampled)

#     # Treat each pixel's temporal sequence independently
#     sampled_x = sampled_x.reshape(-1, 1, D_downsampled)

#     # Interpolate only along the temporal dimension
#     interpolated_x = F.interpolate(
#         sampled_x,
#         size=D,
#         mode='linear',
#         align_corners=False
#     )

#     # Restore original dimensions
#     output_x = interpolated_x.reshape(N, C, H, W, D)
#     output_x = output_x.permute(0, 4, 1, 2, 3)  # (N, D, C, H, W)

#     return output_x


# def frame_drop(x, min_ratio=0.8):
#     """
#         Randomly drop frames from the input tensor
#     """
#     N, D, C, H, W = x.shape
    
#     drop_ratio = torch.empty(1, device=x.device).uniform_(min_ratio, 0.95).item()
#     D_downsampled = int(D * drop_ratio)
    
#     inner_indices = torch.randperm(D-2, device=x.device)[:D_downsampled-2] + 1
#     first_index = torch.tensor([0], device=x.device)
#     last_index = torch.tensor([D-1], device=x.device)
    
#     selected_indices = torch.cat((first_index, inner_indices, last_index))
#     selected_indices, _ = torch.sort(selected_indices)
    
#     sampled_x = x[:, selected_indices, :, :, :]
#     sampled_x = sampled_x.permute(0, 2, 1, 3, 4)  # (N, C, D_downsampled, H, W)
    
#     interpolated_x = F.interpolate(sampled_x, size=(D, H, W), mode='trilinear', align_corners=False)
    
#     output_x = interpolated_x.permute(0, 2, 1, 3, 4)  # (N, D, C, H, W)
#     return output_x


# Gamma augmentation
def gamma_correction(x, gamma_range=(0.8, 2.2), eps=1e-6):
    """
        Apply random gamma correction to the input tensor
    """
    N, D, C, H, W = x.shape
    gamma_vals = torch.empty((N, 1, 1, 1, 1), device=x.device).uniform_(gamma_range[0], gamma_range[1])
    
    x_min = x.amin(dim=(1, 2, 3, 4), keepdim=True)
    x_max = x.amax(dim=(1, 2, 3, 4), keepdim=True)
    x_norm = (x - x_min) / (x_max - x_min + eps)
    
    x_gamma = torch.pow(torch.clamp(x_norm, min=eps, max=1.0), gamma_vals)
    
    mean = x_gamma.mean(dim=(1, 2, 3, 4), keepdim=True)
    std = x_gamma.std(dim=(1, 2, 3, 4), keepdim=True)
    x_out = (x_gamma - mean) / (std + eps)
    
    return x_out


# Light & Skin Tone augmentation
def light_skin_tone_adjustment(x, noise_level=0.4):
    """
        Apply random light and skin tone adjustment to the input tensor
    """
    N, D, C, H, W = x.shape
    
    identity = torch.eye(3, device=x.device).unsqueeze(0).expand(N, 3, 3)
    noise = torch.rand((N, 3, 3), device=x.device) - 0.5
    
    transform_matrices = (1 - noise_level) * identity + noise_level * noise
    
    x_light = torch.einsum('ndchw,njc->ndjhw', x, transform_matrices)

    mean = x_light.mean(dim=(1, 2, 3, 4), keepdim=True)
    std = x_light.std(dim=(1, 2, 3, 4), keepdim=True)
    x_out = (x_light - mean) / (std + 1e-6)
    
    return x_out


# Time Delay augmentation
def time_delay(x, labels, fs=30, max_delay=0.5):
    ...
    

def facial_expression_aug(x, max_warp=0.05, knot_step=10):
    """
    Simulate non-rigid facial muscle movements (e.g., talking, facial expressions)
    via anatomically weighted spatio-temporal elastic grid warping (without shading).
    """
    N, D, C, H, W = x.shape
    device = x.device
    
    # 1. Create a coarse 5x5 spatial control grid and temporal control knots
    grid_size = 5
    num_knots = max(2, D // knot_step + 1)
    
    # Generate random (dx, dy) displacements in [-1, 1] at control knots
    # Shape: (N, 2, num_knots, 5, 5)
    disp_knots = torch.rand(N, 2, num_knots, grid_size, grid_size, device=device) * 2 - 1
    
    # 2. Apply anatomical spatial weighting mask on the 5x5 grid:
    # - Outer boundary (indices 0 and 4): 0.0 to anchor image borders.
    # - Row 1 (forehead/eyes): 0.2 (minimal deformation to preserve clean rPPG ROI).
    # - Row 2 (cheeks/nose): 0.5 - 0.7 (moderate deformation).
    # - Row 3 (mouth/jaw): 0.8 - 1.0 (strongest deformation during talking/expressions).
    anatomical_mask = torch.tensor([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.2, 0.2, 0.2, 0.0],
        [0.0, 0.7, 0.5, 0.7, 0.0],
        [0.0, 0.8, 1.0, 0.8, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ], device=device).view(1, 1, 1, grid_size, grid_size)
    
    disp_knots = disp_knots * anatomical_mask * max_warp
    
    # 3. Trilinear interpolation for smooth spatio-temporal displacement fields
    # Output shape: (N, 2, D, H, W)
    disp_field = F.interpolate(
        disp_knots.view(N * 2, 1, num_knots, grid_size, grid_size),
        size=(D, H, W),
        mode='trilinear',
        align_corners=True
    ).view(N, 2, D, H, W)
    
    # Reshape to (N * D, H, W, 2) to match F.grid_sample coordinate format
    disp_field = disp_field.permute(0, 2, 3, 4, 1).reshape(N * D, H, W, 2)
    
    # 4. Construct base normalized coordinate grid in [-1, 1] and add displacements
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, H, device=device),
        torch.linspace(-1.0, 1.0, W, device=device),
        indexing='ij'
    )
    base_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0)  # (1, H, W, 2)
    warped_grid = base_grid + disp_field
    
    x_flat = x.contiguous().view(N * D, C, H, W)
    warped_x = F.grid_sample(
        x_flat, warped_grid, mode='bilinear', padding_mode='reflection', align_corners=True
    ).view(N, D, C, H, W)
    
    return warped_x