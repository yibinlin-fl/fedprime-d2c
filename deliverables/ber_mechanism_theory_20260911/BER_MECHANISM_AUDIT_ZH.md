# BER 有效分布与 CLE 失衡压缩审计

本审计只读取固定 strict-fit 标签、冻结PEW标注和报告专用真实operator metadata；不训练、
不推理、不使用GPU。真实metadata不进入方法，只用于检验理论到真实family的连接。

## Pooled结果

- pseudo-environment TV: `0.487505 -> 0.199744` （相对下降 `59.03%`）。
- true-family TV: `0.634914 -> 0.433513` （相对下降 `31.72%`）。
- code/theory identity最大误差：`7.321e-16`。
- 冻结门槛：`{"T0_code_theory_identity_error_le_1e_10": true, "T1_pseudo_dependence_decreases_4_of_4": true, "T2_pooled_pseudo_tv_relative_drop_ge_0p50": true, "T3_true_family_dependence_decreases_4_of_4": true}`。
- verdict: `PASS`。

## 边界

该结果证明当前BER权重确实构造了理论定义的有效分布，并检验类别—环境依赖是否被压缩；
它不证明训练后的DSA必然下降，也不替代现有Formal A/B。
