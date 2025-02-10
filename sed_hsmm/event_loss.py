import torch
from torch import nn


class EventProbabilityLoss(nn.Module):
    """
    Loss function for event-level prediction.

    This loss computes the negative log-likelihood of predicted event probabilities
    given the ground truth labels. It normalizes the sum of log-probabilities by
    the total number of events to handle varying event counts across clips.

    Forward Args:
        logp_event (torch.Tensor): Log-probabilities of events with shape `(B, C, N, D, T)`.
        y (torch.Tensor): Ground truth event labels with the same shape as `logp_event`,
                          where 1 indicates the presence of an event and 0 indicates absence.

    Returns:
        torch.Tensor: Scalar loss value representing the negative log-likelihood averaged over the batch.
    """

    def forward(self, logp_event: torch.Tensor, y_event: torch.Tensor) -> torch.Tensor:
        num_events = torch.sum(y_event, dim=(-3, -2, -1))
        loss_event = -torch.mean(torch.sum(y_event * logp_event, dim=(-3, -2, -1)) / num_events, dim=-1)

        return loss_event
