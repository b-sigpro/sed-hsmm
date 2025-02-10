# Onset-and-Offset-Aware Sound Event Detection
This repository provides scripts for SED-HSMM and HSM3 in our paper: [Onset-and-Offset-Aware Sound Event Detection via Differentiable Frame-to-Event Mapping
](https://ieeexplore.ieee.org/document/10771642/).

## Features
- Onset-and-Offset-Aware Sound Event Detection based on a Hidden Semi-Markov Mixture Model (HSM3)
- Differentiable Frame-to-Event Mapping via the Forward-Backward Algorithm
- Easily combinable with standard frame-wise feature extractors (e.g., CRNNs, Conformers)


# Getting started
You can install this package by:
```bash
pip install git+https://github.com/b-sigpro/sed-hsmm.git
```

This package provides `sed_hsmm.HSM3Head` as a differentiable frame-to-event mapping layer for the standard frame-wise feature extractors.
It utilizes the forward-backward algorithm to compute posterior probabilities at both the event and frame levels.
Emission probabilities, mixture ratios, and duration distributions are learned as model parameters.


## Arguments
- **K** *(int)*: Number of components for HSMM mixtures (default: 8)
- **L** *(int)*: Number of components for gamma distributions of durations (default: 1)
- **C** *(int)*: Number of output classes (default: 10)
- **D** *(int)*: Maximum duration (default: 156)
- **F** *(int)*: Number of input feature channels (default: 256)
- **a_00** *(float)*: Self-transition probability from the inactive state (default: 0.99)
- **a_10** *(float)*: Transition probability from the active state to the inactive state (default: 0.99)

## Forward pass

### Input
- **h** *(torch.Tensor)*: Input feature tensor of shape `(batch_size, F, sequence_length)`

### Returns
- **logp_event** *(torch.Tensor)*: Log posterior probabilities of events with shape `(batch_size, C, N, D, T)`
- **p_frame** *(torch.Tensor)*: Posterior frame-wise probabilities with shape `(batch_size, C, T)`

## Event-wise  loss function
Let `y_eve` be a `torch.Tensor` representing event-level groundtruth labels with the shape `(B, C, N, D, T)`, where `B` and `N` are the batch size and the number of states (0=inactive, 1=active). The event-wise loss function can be calculated by:
```python
crnn = CRNN(...)  # please provide yourself
hsm3_head = HSM3Head(F=crnn.out_channels)  # initialize HSM3Head

...

h = crnn(log_mel)  # calculate frame-wise features with shape `(batch_size, F, sequence_length)`
logp_event, p_frame = hsm3_head(h)  # calculate event-level posterior probabilities

...

# calculate loss function
loss = -torch.mean(torch.sum(y_eve * logp_event, dim=(-3, -2, -1)) / torch.sum(y_eve, dim=(-3, -2, -1)), dim=-1)
```

# Limitations
- [ ] We are now preparing full recipes for building a CRNN-based SED system 

# Reference
Please cite as:
```bibtex
@article{yoshinaga2025onset,
  title={Onset-and-Offset-Aware Sound Event Detection via Differentiable Frame-to-Event Mapping}, 
  author={Yoshinaga, Tomoya and Tanaka, Keitaro and Bando, Yoshiaki and Imoto, Keisuke and Morishima, Shigeo},
  journal={IEEE Signal Processing Letters}, 
  volume={32},
  year={2025},
  pages={186-190},
  publisher={IEEE}
}
```