import torch
from torch import nn
from torch.nn import functional as fn


class AttentionPoolHead(nn.Module):
    """
    Attention-based pooling head for 1D feature maps.

    Args:
        F (int): Number of input feature channels.
        out_channels (int): Number of output channels after convolution.

    Forward Args:
        x (torch.Tensor): Input tensor of shape (batch_size, F, sequence_length).

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_channels).
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.emb_head = nn.Conv1d(in_channels, out_channels, 1)
        self.att_head = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.emb_head(x)
        att = self.att_head(x)
        return torch.sum(emb * fn.softmax(att, dim=-1), dim=-1)
