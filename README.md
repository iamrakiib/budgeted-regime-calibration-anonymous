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
budgeted-regime-calibration-anonymous/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── brc_core.py
├── dependency_paths.py
└── main/
    ├── tabular/
    │   ├── data_models.py
    │   └── run_final_tabular.py
    │
    ├── image/
    │   ├── cifar100_vit/
    │   │   └── run_cifar100_vit_brc.py
    │   │
    │   ├── cifar100_resnet/
    │   │   ├── data_models.py
    │   │   ├── validation_common.py
    │   │   └── run_cifar100_resnet_corrected.py
    │   │
    │   └── imagenet/
    │       └── run_imagenet_brc.py
    │
    └── regression/
        ├── matched_data_models.py
        └── run_remaining_literature_baselines.py
