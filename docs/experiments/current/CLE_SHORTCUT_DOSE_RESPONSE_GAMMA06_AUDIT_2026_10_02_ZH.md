# CLE shortcut 中等强度 gamma=0.6 零训练剂量响应审计

Updated: 2026-10-02

## 最终状态

```text
status: COMPLETE
verdict: MONOTONE_DOSE_RESPONSE_SUPPORTED
training: NONE
paid platform: NONE
```

完整1000-source审计中：

```text
gamma       0.0          0.6          0.9
pooled DSA -0.000302     0.077541     0.201321
```

`gamma0.6-gamma0`为`+0.077843`，source-bootstrap CI95为
`[0.075040, 0.080635]`；`gamma0.9-gamma0.6`为`+0.123780`，CI95为
`[0.120450, 0.127230]`。`gamma=0.6`的shuffled-binding null p95为`0.022842`，观测
DSA为`0.077541`，`p=0.000999`。D1--D4全部通过。

结果报告：

```text
deliverables/cle_shortcut_dose_response_gamma06_20261002/RESULT_SUMMARY_ZH.md
```

## 目的

历史Phase-A0已在完全相同的1000个clean sources、16个operators与severity 3网格上证明：
`gamma=0.0`的DSA接近零，而`gamma=0.9`的DSA显著升高。当前审计只补历史
`gamma=0.6`最终checkpoint，判断三个强度点是否呈现方向性shortcut的剂量响应。

## 冻结资产与信息边界

```text
checkpoint archive: outputs/cle_rahfl_diagnostic_outputs.tar.gz
endpoint cache:      Phase-A0 gamma0/gamma0.9 sealed probability cache
new inference:       gamma0.6 only, four clients
training:            none
sources/operators:   1000 x 16
severity/seed:       3 / 20260830
binding:             historical CLE-v1 client-class-family map
```

必须验证三档resolved config除experiment name与gamma-specific private root外完全一致；必须复用
Phase-A0相同labels、operator顺序、family映射、binding与severity。禁止重训、调参或改变网格。

## 运行前冻结的描述性门槛

```text
D1: DSA(gamma0) < DSA(gamma0.6) < DSA(gamma0.9)
D2: paired source-bootstrap CI95 of DSA(gamma0.6)-DSA(gamma0) has lower bound > 0
D3: paired source-bootstrap CI95 of DSA(gamma0.9)-DSA(gamma0.6) has lower bound > 0
D4: gamma0.6 observed DSA exceeds its shuffled-binding null p95
```

四项全过才记为`MONOTONE_DOSE_RESPONSE_SUPPORTED`。任何失败都必须如实报告，不得通过改变
gamma、source、operator、severity、bootstrap seed或门槛救回。

## 结论边界

这是使用一个历史training seed的post-hoc受控CLE-v1扩展。即使全部通过，也只支持“在该受控
协议中，directional shortcut随CLE强度增加”，不证明现实世界发生率、不覆盖未知/复合
corruption，也不构成PEW+BER有效性的证据。

## 入口

```text
scripts/audit_cle_shortcut_dose_response_gamma06.py
```
