import math

import torch
from torch import nn
from torch.nn import functional as fn


class GammaDuration(nn.Module):
    """
    Gamma-based duration modeling for Hidden Semi-Markov Models (HSMM).

    This module models the duration distribution of events using a gamma distribution.
    It computes the log-probabilities of durations for both the inactive (uniform) and
    active (gamma) states. The model allows for a mixture of distributions to capture
    complex duration patterns, and enforces small probabilities for short durations
    to account for noise or outliers.

    Args:
        D (int): Maximum duration.
        K (int): Number of components for HSMM mixtures.
        L (int): Number of components for gamma distributions of durations.
        C (int): Number of output classes.
        shape_factor (float, optional): Factor to scale the shape parameter of the gamma distribution (default is 5).
        d_short (int, optional): Threshold below which durations are forced to have small probabilities (default is 3).
        p_short (int, optional): Penalty applied to short durations (default is 3).
        eps (float, optional): Small epsilon value to avoid numerical instability (default is 1e-6).

    Forward Args:
        None: The forward pass does not require external inputs.

    Returns:
        torch.Tensor: Log-probabilities of durations with shape (K, C, N, D), where N=2 (inactive, active).
    """

    logp_d0: torch.Tensor
    d: torch.Tensor

    def __init__(
        self,
        D: int,
        K: int,
        L: int,
        C: int,
        shape_factor: float = 5,
        d_short: int = 3,
        p_short: int = -4,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.shape_factor = shape_factor
        self.scale_factor = D / self.shape_factor

        self.log_D = math.log(D)

        self.d_short = d_short
        self.p_short = p_short
        self.eps = eps

        self.scale = torch.nn.Parameter(torch.rand(K, L, C, 1))
        self.shape = torch.nn.Parameter(torch.rand(K, L, C, 1))

        self.ratio = torch.nn.Parameter(torch.randn(K, L + 1, C, 1))

        self.register_buffer("logp_d0", torch.full((K, C, D), -math.log(D)))
        self.register_buffer("d", torch.arange(1, D + 1))

    @torch.compile
    def _calculate_gamma_dist(self) -> torch.Tensor:
        scale = self.scale_factor * fn.softplus(self.scale) + self.eps
        shape = self.shape_factor * fn.softplus(self.shape) + self.eps
        return (
            (shape - 1) * torch.log(self.d)[None, None, None, :]
            - self.d[None, None, None, :] / scale
            - shape * torch.log(scale)
            - torch.lgamma(shape)
        )

    def forward(self) -> torch.Tensor:
        # Gamma-based duration probabilities for active states
        logp_d1_gm = self._calculate_gamma_dist()
        logp_d1_gm[..., : self.d_short] = self.p_short - self.log_D  # we set very small values for short durations.
        logp_d1_gm = fn.log_softmax(logp_d1_gm, dim=-1)
        K, _, C, D = logp_d1_gm.shape

        # probability mass for d=D
        logp_d1_D = torch.full([K, 1, C, D], self.p_short - self.log_D)
        logp_d1_D[..., -1] = 0
        logp_d1_D = fn.log_softmax(logp_d1_D, dim=-1)

        # combined duration probabilities for active states
        logp_d1 = torch.cat([logp_d1_gm, logp_d1_D], dim=1)
        logp_d1 = torch.logsumexp(fn.log_softmax(self.ratio, dim=1) + logp_d1, dim=1)

        # combined duration probabilities for inactive (n=0) and active (n=1) states
        logp_d = torch.stack([self.logp_d0, logp_d1], dim=2)

        return logp_d  # [K, C, N, D]
