import torch
import torch.nn.functional as F
import math

# Motion augmentation
def jitter_affine(x, max_angles=2.0, max_translate=0.03):
    """
        Apply random affine transformations to the input tensor
    """
    N, D, C, H, W = x.shape
    x = x.contiguous().view(N * D, C, H, W)
    
    # Generate random angles and translations
    angles = (torch.rand(N * D, device=x.device) * 2 - 1) * max_angles * math.pi / 180.0
    tx = (torch.rand(N * D, device=x.device) * 2 - 1) * max_translate
    ty = (torch.rand(N * D, device=x.device) * 2 - 1) * max_translate
    
    affine_matrices = torch.zeros(N * D, 2, 3, device=x.device)
    affine_matrices[:, 0, 0] = torch.cos(angles)
    affine_matrices[:, 0, 1] = -torch.sin(angles)
    affine_matrices[:, 0, 2] = tx
    affine_matrices[:, 1, 0] = torch.sin(angles)
    affine_matrices[:, 1, 1] = torch.cos(angles)
    affine_matrices[:, 1, 2] = ty
    
    grid = F.affine_grid(affine_matrices, x.size(), align_corners=False)
    jittered_x = F.grid_sample(x, grid, mode='bilinear', padding_mode='reflection', align_corners=False)
    
    jittered_x = jittered_x.view(N, D, C, H, W)
    
    return jittered_x


# Frame rate augmentation
def frame_drop(x, min_ratio=0.8):
    """
        Randomly drop frames from the input tensor
    """
    N, D, C, H, W = x.shape
    
    drop_ratio = torch.empty(1, device=x.device).uniform_(min_ratio, 0.95).item()
    D_downsampled = int(D * drop_ratio)
    
    inner_indices = torch.randperm(D-2, device=x.device)[:D_downsampled-2] + 1
    first_index = torch.tensor([0], device=x.device)
    last_index = torch.tensor([D-1], device=x.device)
    
    selected_indices = torch.cat((first_index, inner_indices, last_index))
    selected_indices, _ = torch.sort(selected_indices)
    
    sampled_x = x[:, selected_indices, :, :, :]
    sampled_x = sampled_x.permute(0, 2, 1, 3, 4)  # (N, C, D_downsampled, H, W)
    
    interpolated_x = F.interpolate(sampled_x, size=(D, H, W), mode='trilinear', align_corners=False)
    
    output_x = interpolated_x.permute(0, 2, 1, 3, 4)  # (N, D, C, H, W)
    return output_x


# Gamma augmentation
def gamma_correction(x, gamma_range=(0.8, 2.2)):
    """
        Apply random gamma correction to the input tensor
    """
    N, D, C, H, W = x.shape
    gamma_vals = torch.empty((N, 1, 1, 1, 1), device=x.device).uniform_(gamma_range[0], gamma_range[1])
    
    # Normalize the input to [0, 1]
    x_norm = x.float() / 255.0
    x_norm = torch.pow(torch.clamp(x_norm, min=1e-6, max=1.0), gamma_vals)
    x_gamma = x_norm * 255.0
    
    return x_gamma


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
    x_light = torch.clamp(x_light, 0, 255)
    
    return x_light


# Time Delay augmentation
def time_delay(x, labels, fs=30, max_delay=0.5):
    # x shape: (N, D, C, H, W)
    # labels shape: (N, D)
    N, D, C, H, W = x.shape

    max_delay_frames = min(int(max_delay * fs), D - 1)

    delay = torch.randint(1, max_delay_frames + 1, (N,), device=x.device)
    direction = torch.randint(0, 2, (N,), device=x.device)

    x_delayed = torch.zeros_like(x)
    labels_delayed = torch.zeros_like(labels)

    for i in range(N):
        d = delay[i].item()

        if direction[i] == 0:  # forward / shift right
            x_delayed[i, d:] = x[i, :-d]
            x_delayed[i, :d] = x[i, :d].flip(0)
            
            labels_delayed[i, d:] = labels[i, :-d]
            labels_delayed[i, :d] = labels[i, :d].flip(0)

        else:  # backward / shift left
            x_delayed[i, :-d] = x[i, d:]
            x_delayed[i, -d:] = x[i, -d:].flip(0)
            
            labels_delayed[i, :-d] = labels[i, d:]
            labels_delayed[i, -d:] = labels[i, -d:].flip(0)

    return x_delayed, labels_delayed