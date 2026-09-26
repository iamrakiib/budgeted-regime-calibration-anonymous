# Budgeted Regime Calibration

Anonymous supplementary code for the ICLR 2027 submission:

**Budgeted Regime Calibration: Allocating Calibration Resolution Across Predictive Regimes**

This repository contains the implementation of Budgeted Regime Calibration
(BRC) and the primary matched BRC-versus-Uniform experiment runners reported
in the paper.

The complete experimental protocol, including data splits, reporting seeds,
support thresholds, score definitions, candidate resolutions, allocation
budgets, and evaluation settings, is specified in the paper and appendix.
The manuscript should be treated as the authoritative description of the
reported experimental protocol.

## Repository structure

```text
.
├── brc_core.py
├── dependency_paths.py
├── main/
│   ├── tabular/
│   ├── image/
│   │   ├── cifar100_vit/
│   │   ├── cifar100_resnet/
│   │   └── imagenet/
│   └── regression/
├── third_party/
│   └── README.md
├── THIRD_PARTY_DEPENDENCIES.md
├── requirements.txt
└── .gitignore