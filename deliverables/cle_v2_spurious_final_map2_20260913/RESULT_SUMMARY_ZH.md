# CLE-v2 Held-out Map2 40轮四臂Formal结果

日期：2026-09-13

## 正式结论

```text
verdict: GO_FOUR_ARM_HELDOUT_MAP2
gates: I0/S0/P1/P2/P3/P4/P5 all PASS
scientific_evidence: true
```

五臂12轮screen只在原binding map上选择晋级方法；本实验使用筛选未见的map2，固定partition、
evaluation seed、training seed、initial states、private batch轨迹、strict AsymHFL-val通信和
single-view训练，比较`ERM/CVaR-DRO/PEW+GroupDRO/PEW+BER`。训练40轮、每客户端每轮16个
local batches；AugMix/JSD/DCL/CDep全部禁用。

## Pooled主结果

| Arm | DSA↓ | Operator-grid Acc↑ | Last-10 Avg↑ | Last-10 Worst↑ | WCCA↑ | CFG↓ |
|---|---:|---:|---:|---:|---:|---:|
| ERM | 0.260308 | 20.3067 | 19.8578 | 16.3007 | 0.000 | 32.8325 |
| CVaR-DRO | 0.088982 | 20.4567 | 18.6700 | 14.0380 | 0.125 | 27.7775 |
| PEW+GroupDRO | 0.214575 | 20.7883 | 20.9562 | 17.8140 | 0.025 | 31.5275 |
| **PEW+BER** | **0.077741** | **23.7433** | **24.4518** | **20.4727** | **2.650** | **22.8025** |

PEW+BER在本次Formal的全部pooled报告指标上最优。相对ERM，DSA降低`0.182567`（70.13%），
operator-grid accuracy提高`3.4367pp`，last-10 Avg/Worst分别提高`4.5940/4.1720pp`，CFG降低
`10.0300pp`。

## 直接机制对照

5000次source-level paired bootstrap使用冻结seed `20260913`：

| Estimand | Point | Bootstrap mean | CI95 | 解释 |
|---|---:|---:|---:|---|
| DSA(ERM)-DSA(BER) | 0.182567 | 0.182527 | [0.180596,0.184407] | BER显著缓解CLE |
| DSA(PEW+GroupDRO)-DSA(BER) | 0.136834 | 0.136808 | [0.134952,0.138706] | 相同PEW信息下BER更优 |
| DSA(BER)-DSA(CVaR) | -0.011240 | -0.011226 | [-0.012355,-0.010092] | 本次Formal中BER pooled DSA更低 |

`BER-CVaR=-0.011240`表示BER的定向shortcut概率质量比CVaR低约`0.01124`，相对CVaR的DSA
低约12.63%。该区间只覆盖固定训练结果下1000个source的抽样不确定性，不覆盖training seed、
partition、binding map或数据集不确定性。

## 逐客户端结果

| Arm | c0 DSA | c1 DSA | c2 DSA | c3 DSA |
|---|---:|---:|---:|---:|
| ERM | 0.193430 | 0.265025 | 0.285601 | 0.297177 |
| CVaR-DRO | 0.034684 | 0.064317 | 0.148525 | 0.108401 |
| PEW+GroupDRO | 0.159634 | 0.169908 | 0.265515 | 0.263244 |
| PEW+BER | 0.050544 | 0.075389 | 0.145263 | 0.039770 |

BER相对ERM和PEW+GroupDRO在4/4客户端均降低DSA。相对CVaR，BER在c2/c3更低，在c0/c1
分别高`0.015860/0.011072`；因此只能宣称pooled BER优于CVaR，不能宣称所有架构DSA均优于
CVaR。

| Arm | c0 Grid Acc | c1 Grid Acc | c2 Grid Acc | c3 Grid Acc |
|---|---:|---:|---:|---:|
| ERM | 18.0467 | 20.2533 | 22.0267 | 20.9000 |
| CVaR-DRO | 15.1667 | 22.6933 | 23.4600 | 20.5067 |
| PEW+GroupDRO | 16.6867 | 20.5133 | 24.2533 | 21.7000 |
| PEW+BER | 19.5667 | 23.0867 | 26.4333 | 25.8867 |

BER的operator-grid accuracy在4/4客户端上均高于ERM、CVaR和PEW+GroupDRO。

## Shortcut真实性与完整性

```text
ERM observed DSA:       0.260308
shuffled-binding p95:   0.047625
p-value:                0.000999
private samples:        40000
paired sources:         1000
operators/source:       15
metrics rows/arm:       40
local trace rows/arm:   160
all traces matched:     true
input audit:            PASS
CDep used:              false
```

独立从`SPURIOUS_FINAL_MAP2_PREDICTIONS.npz`重算四臂DSA，与正式summary最大绝对误差为`0`。

## 与12轮Screen的一致性

```text
                         original-map screen     held-out map2 Formal
ERM-BER DSA reduction        0.176922                 0.182567
GroupDRO-BER gap             0.139850                 0.136834
BER grid gain vs ERM        +2.7083pp                +3.4367pp
```

BER相对ERM和matched GroupDRO的绝对DSA优势在不同binding map与训练长度下接近，形成较强的
机制一致性。CVaR与BER排序从screen的CVaR略优`0.002909`变为Formal的BER优`0.011240`，所以
总体主张应为“BER在DSA上与CVaR具有竞争力，并在held-out 40轮Formal中取得更低pooled DSA；
BER的任务效用在两次比较中均明显更高”，不得写成跨所有场景绝对支配CVaR。

## 理论含义与边界

该结果强化了BER有效分布理论的经验落点：PEW提供伪环境结构并不自动产生相同收益，因为共享
相同PEW的GroupDRO明显弱于BER；按类别内部环境支持量构造的BER目标才与CLE形成机制匹配。因此
PEW+BER可定位为“由CLE支持结构理论导出的类别内环境平衡方法”，而不只是经验重加权插件。

该结果仍不构成BER无条件性能定理，也不覆盖：

- 新training seeds或新partition；
- 第二private图像数据集；
- 新corruption库、severity机制或真实采集管线；
- 所有客户端架构上均优于CVaR的DSA；
- source-bootstrap之外的训练/场景/数据集不确定性。

## 运行与产物

```text
training: 9344.884 s
analysis:   93.444 s
total:    9438.328 s = 2 h 37 min 18 s

archive:
outputs/openi_downloads/cle_v2_spurious_final_map2_formal/cle_v2_spurious_final_map2_formal_outputs.tar.gz
bytes: 9321199
SHA256: A935094BA8C6DB37E20E0EB22AA54A5F2628C0B0A7B9039E6F7A783E78F7BA4C
```
