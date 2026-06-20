# Copyright Gabriel Fiastre — CaptionFormer (https://github.com/gabfstr/CaptionFormer)

import torch


def forward_pass(scores, low_thresh=0.4, high_thresh=0.6):
    """ Forward hysteresis thresholding pass """
    active = torch.zeros_like(scores, dtype=torch.bool)
    is_active = False

    for i in range(len(scores)):
        if scores[i] >= high_thresh:
            is_active = True
        elif scores[i] < low_thresh:
            is_active = False
        active[i] = is_active

    return active

def backward_pass(scores, low_thresh=0.4, high_thresh=0.6):
    """ Backward hysteresis thresholding pass """
    active = torch.zeros_like(scores, dtype=torch.bool)
    is_active = False

    for i in range(len(scores) - 1, -1, -1):
        if scores[i] >= high_thresh:
            is_active = True
        elif scores[i] < low_thresh:
            is_active = False
        active[i] = is_active

    return active

def filter_short_tracks(active, min_duration=3):
    """ Removes tracks shorter than min_duration """
    active = active.clone()
    count = 0

    for i in range(len(active)):
        if active[i]:
            count += 1
        else:
            if count < min_duration:
                active[i - count : i] = False  # Remove short tracks
            count = 0

    return active


def bidirectional_hysteresis_batch(scores_batch, low_thresh=0.4, high_thresh=0.6, min_duration=3):
    """
    Applies forward & backward passes and filters short tracks for a batch of sequences.
    Args:
        scores_batch (torch.Tensor): A tensor of shape (batch_size, sequence_length) containing score sequences.
        low_thresh (float): Weak activation threshold.
        high_thresh (float): Strong activation threshold.
        min_duration (int): Minimum number of frames to keep an activation.
    Returns:
        torch.Tensor: A tensor of shape (batch_size, sequence_length) with selected active frames.
    """
    batch_size, seq_len = scores_batch.shape
    active_batch = torch.zeros_like(scores_batch, dtype=torch.bool)

    for i in range(batch_size):
        active_fwd = forward_pass(scores_batch[i], low_thresh, high_thresh)
        active_bwd = backward_pass(scores_batch[i], low_thresh, high_thresh)
        active = active_fwd | active_bwd  # Combine both passes
        active = filter_short_tracks(active, min_duration)  # Remove short tracks
        active_batch[i] = active

    return active_batch