# CLE-HFL V0.8交付说明

日期：2026-10-25。

本目录是由V0.7-writing派生的论文快照。旧V0.6、V0.7与V0.7-writing均保持不变。
本次只同步已经完成独立审计的moderate seed-0 Formal，不把待跑实验写成完成证据。

## 内容

```text
source/     可继续编辑的完整LaTeX源码
CLE_HFL_LATEX_V0_8_MODERATE_FORMAL_OVERLEAF_20261025.zip
            Overleaf兼容上传包；main.tex位于ZIP根目录
```

上传包：`917759` bytes；SHA256：
`D873E08C81E7FECF7F75E0F1503BC45CF6FB4EF06153DF016D5AD0FF3879ACBF`。
ZIP含49个条目，根目录存在`main.tex`，无`./`前缀和反斜杠条目；内置LaTeX编译已成功。

## 本次论文回填

- moderate `gamma=0/0.5/0.9` 的DSA剂量响应；
- `gamma=.9`下AsymHFL--ERM与AsymHFL+PEW+BER的匹配比较；
- PEW+BER的DSA绝对下降`0.094490`、相对下降`51.21%`；
- 5,000次source-paired bootstrap区间与4/4客户端同向下降；
- grid/corrupt改善与clean/mixed下降并存的效用边界；
- Axis II状态由3/8更新为5/8。

未修改问题定义、DSA/BER公式、理论命题、引用键或旧实验结论。FedDF、AugHFL、RAHFL、
local-first多training-seed、第二private dataset和bounded taxonomy stress仍保留为待证据。

正式数字来源：

```text
deliverables/cle_v2_moderate_axis2_dose_formal_seed0_20261025/RESULT_SUMMARY_ZH.md
```
