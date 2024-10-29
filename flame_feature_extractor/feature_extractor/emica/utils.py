import math

import torch
import torch.nn.functional as F


def resample_frames(frames_list, original_fps, target_fps):
    # Step 1: Calculate the total duration of the video
    total_frames = len(frames_list)
    duration = total_frames / original_fps  # in seconds

    # Step 2: Calculate the number of frames at the target FPS
    new_total_frames = int(math.ceil(duration * target_fps))

    resampled_frames = []

    # Step 3: Map each new frame to the corresponding original frame
    for i in range(new_total_frames):
        # Time of the new frame
        t = i / target_fps
        # Corresponding index in the original frames
        orig_index = t * original_fps
        # Round to the nearest index
        index = int(round(orig_index))
        # Ensure index is within bounds
        index = min(index, total_frames - 1)
        # Append the frame to the resampled list
        resampled_frames.append(frames_list[index])

    return resampled_frames


def smooth_tensor(input_tensor, kernel_weight: list[float] = [0.1, 0.2, 0.5, 0.2, 0.1]):
    # input_tensor shape: [B, N, D]
    B, N, D = input_tensor.shape

    # Step 1: Create the filter kernel
    filter_weights = torch.tensor(kernel_weight, dtype=input_tensor.dtype, device=input_tensor.device)
    filter_weights = filter_weights.view(1, 1, -1)  # Shape: [1, 1, K], where K=5

    # Step 2: Reshape input_tensor for convolution
    # Bring the time axis N to be the last dimension
    input_tensor = input_tensor.permute(0, 2, 1)  # Shape: [B, D, N]
    input_tensor = input_tensor.contiguous().view(B * D, 1, N)  # Shape: [B*D, 1, N]

    # Step 3: Apply padding to maintain sequence length
    padding = (len(kernel_weight) - 1) // 2  # For kernel size K=5, padding=2
    input_padded = F.pad(input_tensor, (padding, padding), mode="reflect")

    # Step 4: Perform the convolution
    smoothed = F.conv1d(input_padded, filter_weights)

    # Step 5: Reshape back to original shape
    smoothed = smoothed.view(B, D, N)
    smoothed = smoothed.permute(0, 2, 1)  # Shape: [B, N, D]

    return smoothed


def reconstruct_feature(masked_feature, mask):
    N, D = mask.shape[0], masked_feature.shape[1]
    reconstructed_feature = torch.empty((N, D), dtype=masked_feature.dtype, device=masked_feature.device)

    positions = torch.arange(N, device=masked_feature.device)
    known_positions = positions[mask]
    unknown_positions = positions[~mask]
    known_features = masked_feature

    # Fill in the known positions
    reconstructed_feature[mask] = known_features

    # Perform linear interpolation for the unknown positions
    idx = torch.searchsorted(known_positions, unknown_positions, right=False)
    K = known_positions.shape[0]

    # Clamp indices to handle edge cases
    idx0 = torch.clamp(idx - 1, 0, K - 1)
    idx1 = torch.clamp(idx, 0, K - 1)

    # Get previous and next positions and features
    previous_positions = known_positions[idx0]
    next_positions = known_positions[idx1]
    previous_features = known_features[idx0]
    next_features = known_features[idx1]

    # Compute weights for interpolation
    delta_positions = next_positions - previous_positions
    delta_positions[delta_positions == 0] = 1  # Avoid division by zero
    weights = (unknown_positions - previous_positions) / delta_positions
    weights = weights.unsqueeze(1)  # Shape (M, 1)

    # Compute interpolated features
    interpolated_features = (1 - weights) * previous_features + weights * next_features

    # Assign interpolated features to reconstructed tensor
    reconstructed_feature[~mask] = interpolated_features

    return reconstructed_feature
