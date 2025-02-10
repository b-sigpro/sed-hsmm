import torch
from torch import nn
from torch.nn import functional as fn

from einops.layers.torch import Rearrange

from sed_hsmm import GammaDuration
from sed_hsmm.nn import AttentionPoolHead


class HSM3Head(nn.Module):
    """
    Hidden Semi-Markov Model (HSMM) head for sequence modeling with neural network-based parameterization.

    This module implements the forward-backward algorithm for HSMMs, where emission probabilities,
    mixture ratios, and duration distributions are learned through neural networks. It calculates
    posterior probabilities at both the event and frame levels.

    Args:
        K (int): Number of components for HSMM mixtures (default is 8).
        L (int): Number of components for gamma distributions of durations (default is 1).
        C (int): Number of output classes (default is 10).
        D (int): Maximum duration (default is 156).
        F (int): Number of input feature channels (default is 256).
        a_00 (float): Self-transition probability from the inactive state (default: 0.99).
        a_10 (float): Transition probability from the active state to the inactive state (default: 0.99).

    Forward Args:
        h (torch.Tensor): Input feature tensor of shape (batch_size, F, sequence_length).

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - logp_event (torch.Tensor): Log posterior probabilities of events with shape (batch_size, C, N, D, T).
            - p_frame (torch.Tensor): Posterior frame-wise probabilities with shape (batch_size, C, T).
    """

    log_A: torch.Tensor

    def __init__(
        self,
        K: int = 8,
        L: int = 1,
        C: int = 10,
        D: int = 156,
        F: int = 256,
        a_00: float = 0.99,
        a_10: float = 0.99,
    ):
        super().__init__()

        # constants
        self.L, self.D, self.C, self.K = L, D, C, K
        self.N = 2  # 0=inactive, 1=active

        # transition probabilities
        A = torch.zeros([K, C, self.N, self.N])
        A[..., 0, 0], A[..., 0, 1] = a_00, 1 - a_00
        A[..., 1, 0], A[..., 1, 1] = a_10, 1 - a_10
        self.register_buffer("log_A", A.log())

        # initial probabilities
        self.logit_lo1 = torch.nn.Parameter(torch.randn(K, C, 1))

        # duration probabilities (as nn.Module)
        self.generate_duration = GammaDuration(D, K, L, C)

        # neural networks
        self.mixture_ratio_head = nn.Sequential(
            nn.Conv1d(F, F, 1),
            nn.ReLU(),
            AttentionPoolHead(F, F),
            nn.Linear(F, K),
            nn.LogSoftmax(dim=-1),
        )

        self.emission_prob_head = nn.Sequential(
            nn.BatchNorm1d(F),
            nn.ReLU(),
            nn.Conv1d(F, C * self.N * K, 1),
            Rearrange("b (k c n) t -> b k c n t", c=C, k=K),
        )

    @torch.compile
    def hsmm_forward(
        self,
        logp_x: torch.Tensor,  # [B, K, C, N, T]
        log_lo: torch.Tensor,  # [K, C, N, 1]
        logp_d: torch.Tensor,  # [K, C, N, D]
    ):
        B, K, C, N, T = logp_x.shape
        _, _, _, D = logp_d.shape

        # calculate log_b
        log_b = torch.full([B, K, C, N, D, T], -10.00, device=logp_x.device)
        for t in range(0, T):
            D_ = min(t + 1, D)
            log_b[:, :, :, :, :D_, t] = torch.cumsum(logp_x[..., t - D_ + 1 : t + 1].flip(dims=(-1,)), dim=-1)

        # calculate log_alpha
        d = torch.arange(min(D, T))

        log_alpha = torch.full_like(log_b, -1e9)
        log_alpha[..., d, d] = log_lo + logp_d[..., d] + log_b[..., d, d]

        log_c = logp_d[..., None] + log_b
        for t in range(1, T):
            D_ = min(t, self.D)

            log_al_A = torch.logsumexp(log_alpha[..., t - D_ : t].clone(), dim=4)[..., None, :] + self.log_A[..., None]
            log_alpha[..., :D_, t] = torch.logsumexp(log_al_A, dim=3).flip(dims=[-1]) + log_c[..., :D_, t]

        return log_alpha, log_b

    @torch.compile
    def hsmm_backward(
        self,
        log_b: torch.Tensor,  # [B, K, C, N, D, T]
        logp_d: torch.Tensor,  # [K, C, N, D]
    ):
        B, K, C, N, D, T = log_b.shape

        # calculate log_a
        log_a = self.log_A[..., None] + logp_d[:, :, None, :, :]  # [K, C, N, N, D]

        # calculate log_beta
        log_beta = torch.full([B, K, C, N, T], -10.00, device=log_b.device)
        log_beta[..., -1] = 0
        for t in range(T - 2, -1, -1):
            D_ = min(T - t, D)

            log_beta[..., t] = torch.logsumexp(
                torch.diagonal(log_b[:, :, :, None, :, : D_ - 1, t + 1 : t + D_], dim1=-2, dim2=-1)
                + log_beta[..., None, :, t + 1 : t + D_].clone()
                + log_a[..., : D_ - 1],
                dim=(4, 5),
            )

        return log_beta

    @torch.compile
    def calculate_logp_event(
        self,
        log_alpha: torch.Tensor,  # [B, K, C, N, D, T]
        log_beta: torch.Tensor,  # [B, K, C, N, T]
        log_pi: torch.Tensor,
    ):
        log_gamma = log_alpha + log_beta[..., None, :]

        logp_event = log_gamma - torch.logsumexp(log_alpha[..., -1], dim=(-2, -1), keepdim=True)[..., None]
        logp_event = torch.logsumexp(log_pi[:, :, None, None, None, None] + logp_event, dim=1)

        return logp_event  # [B, C, N, D, T]

    @torch.no_grad()
    @torch.compile
    def calculate_p_frame(self, logp_event: torch.Tensor):
        B, C, N, D, T = logp_event.shape

        mask: torch.Tensor = torch.ones((T, D, T), device=logp_event.device)
        mask = mask.triu().transpose(0, 1).triu().transpose(0, 1)

        for t in range(T):
            mask[t] = mask[t].tril(diagonal=t)

        neg_mask = mask.logical_not()

        logp_joint = torch.logsumexp(logp_event.unsqueeze(3).masked_fill(neg_mask, -1e9), dim=(-2, -1))

        return torch.softmax(logp_joint, dim=2)[:, :, 1]

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # calculate emission probabilities and mixture ratios from feature vectors h
        logp_x = self.emission_prob_head(h)
        logp_pi = self.mixture_ratio_head(h)

        # obtain model parameters log lo and log p(d)
        log_lo = fn.logsigmoid(torch.stack([self.logit_lo1, -self.logit_lo1], dim=2))  # [K, C, N, 1]
        logp_d = self.generate_duration()  # [K, C, N, D]

        # forward-backward algorithm
        log_alpha, log_b = self.hsmm_forward(logp_x, log_lo, logp_d)
        log_beta = self.hsmm_backward(log_b, logp_d)

        # calculate posterior probabilities
        logp_event = self.calculate_logp_event(log_alpha, log_beta, logp_pi)
        p_frame = self.calculate_p_frame(logp_event)

        return logp_event, p_frame
