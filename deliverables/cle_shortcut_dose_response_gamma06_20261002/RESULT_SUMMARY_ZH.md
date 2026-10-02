# CLE directional shortcut 三点剂量响应审计结果

Date: 2026-10-02

## 结论

历史CLE-v1、RAHFL-style、training seed 0的三档checkpoint在完全相同的1000个clean sources、
16个corruption operators、severity 3和历史binding map上呈现严格单调DSA：

| CLE强度 | pooled DSA | client 0 | client 1 | client 2 | client 3 |
|---:|---:|---:|---:|---:|---:|
| `gamma=0.0` | -0.000302 | 0.002314 | -0.003484 | 0.007391 | -0.007428 |
| `gamma=0.6` | 0.077541 | 0.096507 | 0.097012 | 0.047668 | 0.068977 |
| `gamma=0.9` | 0.201321 | 0.267356 | 0.228555 | 0.131077 | 0.178297 |

冻结的四项门槛全部通过，verdict为：

```text
MONOTONE_DOSE_RESPONSE_SUPPORTED
```

## 配对增量与binding特异性

```text
DSA(gamma0.6)-DSA(gamma0.0) = +0.077842890
source-bootstrap CI95          [0.075039905, 0.080635367]

DSA(gamma0.9)-DSA(gamma0.6) = +0.123780066
source-bootstrap CI95          [0.120450262, 0.127229973]

gamma0.6 observed DSA        = 0.077541000
gamma0.6 shuffled null p95   = 0.022842209
one-sided permutation p      = 0.000999001
```

中等强度下4/4客户端DSA均为正。结果支持：在该受控CLE协议中，类别--corruption绑定由无到中、
再到强时，模型沿真实训练binding方向移动概率质量的程度同步增加。

## 同一固定网格上的辅助指标

| gamma | Avg | Worst | WCCA | CFG | paired flip rate | family-bound top-1 bias |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 52.2422 | 43.4188 | 36.0000 | 2.6500 | 0.148325 | 0.000010 |
| 0.6 | 50.6578 | 41.9812 | 26.5625 | 5.4312 | 0.206817 | 0.088211 |
| 0.9 | 46.8250 | 37.9375 | 19.3125 | 11.8813 | 0.353404 | 0.230527 |

`gamma=0.6`辅助指标与7月原始test报告略有差异，是因为本审计使用Phase-A0重新冻结的同源、
固定severity=3 paired operator grid；不得混用两套评价口径。

## 复用与新增计算

- `gamma=0/0.9`直接复用封存的Phase-A0 softmax cache；其pooled DSA与正式结果逐位一致。
- 只对`gamma=0.6`四个历史最终checkpoint新增64,000次前向。
- 没有训练、梯度更新、调参、OpenI/Kaggle任务或付费算力。
- 三档resolved config除experiment name与gamma-specific private root外完全一致。

## 原始证据与哈希

```text
outputs/cle_shortcut_dose_response_gamma06_seed0/dose_response_summary.json
SHA256 67BD28E6728F61FE8ED73416B689F9EC2C3BB7734777639AC13A7B95F65F774F

outputs/cle_shortcut_dose_response_gamma06_seed0/gamma06_predictions.npz
SHA256 6198B94CF41733805126C54B7AB4FDD345B49C02CC3A7DA3109068F0535EB067

outputs/cle_shortcut_dose_response_gamma06_seed0/dose_response_per_client.csv
SHA256 427E47F4810D4C65D49B8F37C8BF2122FF8BDFD1D5F7EE5DDF3FD8859224BDA6
```

实现与冻结协议：

```text
scripts/audit_cle_shortcut_dose_response_gamma06.py
docs/experiments/current/CLE_SHORTCUT_DOSE_RESPONSE_GAMMA06_AUDIT_2026_10_02_ZH.md
```

## 论文可写主张与边界

可写：

> In the historical controlled CLE-v1 protocol, DSA increases monotonically from independent
> corruption to moderate and strong class--corruption entanglement, while the moderate condition
> already exceeds a shuffled-binding null.

不可写：

- 现实世界CLE发生率已经得到证明；
- 任意轻微CLE必然产生shortcut；
- 该单seed post-hoc结果等价于training-seed稳定性；
- 该结果证明PEW+BER有效；
- 原RAHFL官方40-pretrain+40-communication设置已经被复现。

本结果最适合进入问题刻画/诊断实验中的三点dose-response图或表，并在limitations中保留
controlled CLE、single historical seed与post-hoc扩展边界。
