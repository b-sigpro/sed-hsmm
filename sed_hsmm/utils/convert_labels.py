from itertools import groupby, product

import torch


def convert_labels(
    y_frame: torch.Tensor,
):
    """
    Converts frame-level labels to event-level labels.

    Args:
        y_frame (torch.Tensor): Frame-wise labels with shape `(num_classes, num_frames)`.
        num_classes (int, optional): Number of classes (default is 10).
        clip_duration (float, optional): duration of each clip in seconds (default is 10).
        label_resolution (float, optional): Time resolution of time frames in seconds (default is 0.064).

    Returns:
        torch.Tensor: Event-based labels with shape [num_classes, 2, num_frames, num_frames].
                      The second dimension indicates inactive or active.
    """

    B, C, T = y_frame.shape

    y_frame_np = y_frame.cpu().numpy()

    y_event = torch.zeros([B, C, 2, T, T], dtype=torch.int)
    for b, c, n in product(range(B), range(C), [0, 1]):
        active_indices = (y_frame_np[b, c] == n).nonzero()[0]

        for _, group in groupby(enumerate(active_indices), lambda _: _[1] - _[0]):
            _, ts = zip(*group, strict=False)

            y_event[b, c, n, ts[-1] - ts[0], ts[-1]] = 1

    return y_event
