# CLE-HFL held-out map2四臂40轮多训练种子正式结果

Updated: 2026-09-22

## 1. 正式结论

训练种子0/1/2的matched Formal全部完成。每个seed均为40轮、四臂各160条local traces，轨迹
匹配，输入审计PASS，prediction cache为`4 arms x 4 clients x 1000 sources x 15 operators x
10 classes`且无非有限值。三个seed的单seed冻结门槛全部通过，三seed聚合门槛全部通过：

```text
verdict = GO_MAP2_TRAINING_SEED_STABILITY
```

这把held-out map2结果从training-seed-0发现提升为固定CLE场景内的训练随机性稳定证据。

## 2. 三seed主表，mean +/- sample std

| Arm | DSA down | Operator-grid Acc up | Last-10 Avg up | Last-10 Worst up | WCCA up | CFG down |
|---|---:|---:|---:|---:|---:|---:|
| ERM | 0.250530 +/- 0.008670 | 20.4950 +/- 0.3718 | 19.9589 +/- 0.1212 | 16.3780 +/- 0.1259 | 0.0000 +/- 0.0000 | 32.7467 +/- 0.1708 |
| CVaR-DRO | 0.086654 +/- 0.005514 | 19.2206 +/- 1.1055 | 18.5788 +/- 0.3102 | 14.2636 +/- 0.2640 | 0.1167 +/- 0.0629 | 27.6750 +/- 0.1885 |
| PEW+GroupDRO | 0.205417 +/- 0.008045 | 21.4517 +/- 0.5783 | 20.9963 +/- 0.0924 | 17.5309 +/- 0.5148 | 0.0333 +/- 0.0144 | 31.8950 +/- 0.9747 |
| **PEW+BER** | **0.071470 +/- 0.011288** | **25.1772 +/- 1.2522** | **24.5466 +/- 0.1098** | **20.7516 +/- 0.6126** | **2.6500 +/- 0.0750** | **22.7825 +/- 0.5428** |

## 3. 冻结contrast

| Contrast | seed 0 | seed 1 | seed 2 | three-seed mean +/- std |
|---|---:|---:|---:|---:|
| DSA(ERM)-DSA(BER) | 0.182567 | 0.185345 | 0.169267 | 0.179060 +/- 0.008594 |
| DSA(GroupDRO)-DSA(BER) | 0.136834 | 0.143746 | 0.121260 | 0.133947 +/- 0.011518 |
| DSA(BER)-DSA(CVaR) | -0.011240 | -0.032184 | -0.002128 | -0.015184 +/- 0.015411 |
| BER-CVaR grid accuracy, pp | +3.2867 | +7.1767 | +7.4067 | +5.9567 +/- 2.3151 |
| BER-ERM grid accuracy, pp | +3.4367 | +5.1317 | +5.4783 | +4.6822 +/- 1.0925 |
| BER-ERM last-10 Avg, pp | +4.5940 | +4.4277 | +4.7413 | +4.5877 +/- 0.1569 |

相对三seed均值，BER相对ERM降低DSA约71.47%，相对共享同一PEW分组的GroupDRO降低约
65.21%。BER的mean pooled DSA也低于CVaR约17.52%，同时grid accuracy平均高5.9567个百分点。

## 4. 稳定性门槛

```text
mean_ber_vs_erm_dsa_positive                 PASS
mean_ber_vs_groupdro_dsa_positive            PASS
ber_vs_erm_positive_at_least_two_seeds       PASS (3/3)
ber_vs_groupdro_positive_at_least_two_seeds  PASS (3/3)
```

每个seed各自的I0/S0/P1/P2/P3/P4/P5也全部PASS，scientific verdict均为
`GO_FOUR_ARM_HELDOUT_MAP2`。

## 5. 精确解释

本结果支持：

1. held-out map2上的strong CLE shortcut不是seed-0偶然；
2. BER相对ERM的DSA抑制在3/3 training seeds复现；
3. 同一PEW side information下，BER相对GroupDRO在3/3 seeds更低DSA，故收益不能只归因于
   获得了PEW分组；
4. 在该固定场景中，BER相对CVaR取得更好的pooled DSA--utility组合。

本结果不支持：

1. 对每个客户端的DSA都支配CVaR；三seed均值下BER仅在c2/c3低于CVaR，c0/c1仍更高；
2. architecture causal attribution，因为客户端架构与non-IID数据切片绑定；
3. partition、binding-map population、第二数据集或真实部署稳定性；
4. PEW taxonomy覆盖未知、复合或连续corruption。

## 6. 运行与产物

```text
seed1 total: 9810.711 s (2 h 43 m 31 s)
seed2 total: 9757.306 s (2 h 42 m 37 s)
combined:   19568.017 s (5 h 26 m 08 s)
```

```text
seed1 archive bytes: 9325283
seed1 SHA256: DAA0C775E00589FA0A4CBE1AD077C7A53FBBB8879B132542B0C3C09F2AF01253
seed2 archive bytes: 9309538
seed2 SHA256: 465D87D4B1121AD640F365BC1AAB3C504FE62A7413F35E513CEEA81F3CF557BB
```

机器可读聚合结果：

```text
deliverables/cle_hfl_map2_multiseed_20260922/MULTISEED_SUMMARY.json
```
