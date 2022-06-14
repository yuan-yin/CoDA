# CoDA

Official code for [Generalizing to New Physical Systems via Context-Informed Dynamics Model](https://arxiv.org/pdf/2202.01889.pdf) (CoDA), accepted at ICML 2022.


## Abstract

> Data-driven approaches to modeling physical systems fail to generalize to unseen systems that share the same general dynamics with the learning domain, but correspond to different physical contexts. We propose a new framework for this key problem, context-informed dynamics adaptation (CoDA), which takes into account the distributional shift across systems for fast and efficient adaptation to new dynamics. CoDA leverages multiple environments, each associated to a different dynamic, and learns to condition the dynamics model on contextual parameters, specific to each environment. The conditioning is performed via a hypernetwork, learned jointly with a context vector from observed data. The proposed formulation constrains the search hypothesis space to foster fast adaptation and better generalization across environments. It extends the expressivity of existing methods. We theoretically motivate our approach and show state-ofthe-art generalization results on a set of nonlinear dynamics, representative of a variety of application domains. We also show, on these systems, that new system parameters can be inferred from context vectors with minimal supervision.

## Usage

```bash
python3 train.py -d lotka -h <EXP_HOME> -g <GPU_ID> -m 1e-6 -c 1e-4 -r l12m-l2c -x 2 -s 1
```

- `-d`: dataset name (`lokta`, `gray`, `navier`)
- `-h`: experiment home directory
- `-g`: CUDA GPU id
- `-m`: HyperNet regularization coeffcient
- `-c`: context regularization coeffcient
- `-r`: regularization norm option (`l12m`: L1-L2 norm for the HyperNet, `l2c`: L2 norm for the contexts)
- `-x`: context dimension
- `-s`: random seed

## Notes

This code contains an efficient parallel forward in all environments, implemented with group convolution. More explanation will be added in comments.
