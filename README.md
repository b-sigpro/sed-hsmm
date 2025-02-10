# Onset-and-Offset-Aware Sound Event Detection
This repository provides scripts for SED-HSMM and HSM3 in our paper: [Onset-and-Offset-Aware Sound Event Detection via Differentiable Frame-to-Event Mapping
](https://ieeexplore.ieee.org/document/10771642/).

<div align="center"><img src="https://raw.githubusercontent.com/b-sigpro/sed-hsmm/main/docs/image/overview.png" width="600"/></div>

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
- **a_00** *(float)*: Self-transition probability for the inactive state (default: 0.99)
- **a_10** *(float)*: Transition probability from the active state to the inactive state (default: 0.99)

## Forward pass

### Input
- **h** *(torch.Tensor)*: Input feature tensor of shape `(batch_size, F, T)`

### Returns
- **logp_event** *(torch.Tensor)*: Log posterior probabilities of events with shape `(batch_size, C, N, D, T)`
- **p_frame** *(torch.Tensor)*: Posterior frame-wise probabilities with shape `(batch_size, C, T)`

## Event-wise  loss function
Ley `y_frame` be a `torch.Tensor` representing frame-level groundtruth labels whose shape is `(batch_size, C, T)` and contents is 0 (inactive) or 1 (active).
The event-wise loss function can be calculated by:
```python
from sed_hsmm import HSM3Head, EventProbabilityLoss, convert_labels

crnn = CRNN(...)  # please provide yourself
hsm3_head = HSM3Head(F=crnn.out_channels)  # initialize HSM3Head
calc_event_loss = EventProbabilityLoss()

...

h = crnn(log_mel)  # calculate frame-wise features with shape `(batch_size, F, sequence_length)`
logp_event, p_frame = hsm3_head(h)  # calculate event-level posterior probabilities

...

# calculate loss function
y_event = convert_labels(y_frame)
loss = calc_event_loss(logp_event, y_event)
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

# Acknowledgement
This study was supported in part by the JSPS KAKENHI under Grant No. 24K20807.
