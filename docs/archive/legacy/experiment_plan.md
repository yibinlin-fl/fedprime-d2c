# FedPRIME-D2C Experiment Plan

## Core Claim

RAHFL mainly asks which client is more reliable under corrupted data. FedPRIME-D2C asks which parts of public logits are contaminated by Non-IID predictive priors, then debiases and aggregates them in a class-balanced and complementary way.

## Stage 1: Fast MVP Battle

Goal: verify whether FedPRIME-D2C can beat original RAHFL under data heterogeneity.

Methods:

- Original RAHFL: AugMix + DCL + AsymHFL.
- FedPRIME-D2C MVP: PRIME local learning + D2C communication.

Settings:

- Private training: CIFAR-10-C.
- Public data: CIFAR-100 subset.
- Clients: 4.
- Models: ResNet10, ResNet12, ShuffleNet, MobileNetV2.
- Corruption rate: 1.0.
- Dirichlet alpha: 0.5 and 0.1.
- Report: average accuracy, worst-client accuracy, per-client accuracy.

## Stage 2: Strong Baseline

Add RAHFL + PRIME:

- Replace RAHFL's AugMix local augmentation with PRIME.
- Keep AsymHFL communication unchanged.

This blocks the criticism that FedPRIME-D2C only wins because PRIME is stronger than AugMix.

## Stage 3: Ablation

Run:

- FedPRIME-D2C.
- w/o PRIME.
- w/o prior debiasing.
- w/o class-balanced aggregation.
- w/o complementary KD.
- ordinary logit averaging.
- fixed beta vs adaptive beta.
- predicted prior vs oracle prior.

## Main Command

```bash
python scripts/run_experiment.py --config configs/fedprime_d2c_cifar10c.yaml
```

## Core Comparison Command

```bash
python scripts/run_grid.py \
  configs/cifar10c_rahfl.yaml \
  configs/cifar10c_rahfl_prime.yaml \
  configs/fedprime_d2c_cifar10c.yaml
```

## Ablation Command

```bash
python scripts/run_grid.py \
  configs/ablations/fedprime_d2c_no_prime.yaml \
  configs/ablations/fedprime_d2c_no_prior_debias.yaml \
  configs/ablations/fedprime_d2c_no_class_balanced.yaml \
  configs/ablations/fedprime_d2c_no_complementary_kd.yaml \
  configs/ablations/fedprime_d2c_oracle_prior.yaml \
  configs/ablations/fedprime_d2c_adaptive_ema_gate.yaml
```

For Kaggle or a school server, edit only:

- `data.private_root`
- `data.public_root`
- `output_root`
- `num_workers`
- optionally `device`
