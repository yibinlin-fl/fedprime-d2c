# CLE-v2 Spurious 四臂 Held-out Map2 Formal

更新：2026-09-13

## 目的

五臂12轮screen已在原binding map上完成方法选择。最终实验删除未晋级JTT，只保留：

```text
ERM / CVaR-DRO / PEW+GroupDRO / PEW+BER
```

本协议使用筛选时未见的binding map2，训练40轮，回答：PEW+BER的shortcut抑制能否跨map
复现；它是否优于共享相同PEW分组的标准GroupDRO；相对taxonomy-free CVaR是否形成更好的
准确率—DSA折中。它不证明跨partition、跨数据集、跨corruption库或真实域泛化。

## 冻结协议

```text
scenario              cle_hfl_v2_cross_map2_seed0_split0
partition seed        0
binding-map seed      2
evaluation seed       20260909
training seed         0
rounds                40
local batches         16/client/round
communication         strict AsymHFL-val
augmentation          none
AugMix/JSD/DCL/CDep    all disabled
PEW                    frozen public family witness; only two PEW arms use it
```

四臂共享数据、initial states、私有batch轨迹、audit routing和评价协议。ERM为普通CE；CVaR-DRO
平均每个batch损失最高20%的样本；PEW+GroupDRO动态提高当前高风险类别×PEW组；PEW+BER按
类别内部环境支持量进行结构性平衡。

## 结果前冻结门槛

```text
I0  四臂local batch traces完全匹配
S0  ERM DSA>=0.08，超过shuffled-binding null p95，且p<=0.01
P1  ERM DSA - BER DSA>=0.05，source-bootstrap CI下界>0，4/4客户端均下降
P2  GroupDRO DSA - BER DSA>=0.02，source-bootstrap CI下界>0
P3  BER DSA不高于CVaR超过0.02
P4  BER operator-grid accuracy比CVaR至少高1.0个百分点
P5  BER operator-grid accuracy与last-10 Avg均不低于ERM
```

Formal仅在全部门槛通过时记为`GO_FOUR_ARM_HELDOUT_MAP2`。各项结果仍须逐项报告，不能用综合
verdict掩盖CVaR可能取得最低DSA或某个效用门失败。

## Formal结果：全部冻结门槛通过

```text
verdict: GO_FOUR_ARM_HELDOUT_MAP2
I0/S0/P1/P2/P3/P4/P5: all PASS
```

| Arm | DSA↓ | Grid Acc↑ | Last-10 Avg↑ | Last-10 Worst↑ | WCCA↑ | CFG↓ |
|---|---:|---:|---:|---:|---:|---:|
| ERM | 0.260308 | 20.3067 | 19.8578 | 16.3007 | 0.000 | 32.8325 |
| CVaR-DRO | 0.088982 | 20.4567 | 18.6700 | 14.0380 | 0.125 | 27.7775 |
| PEW+GroupDRO | 0.214575 | 20.7883 | 20.9562 | 17.8140 | 0.025 | 31.5275 |
| **PEW+BER** | **0.077741** | **23.7433** | **24.4518** | **20.4727** | **2.650** | **22.8025** |

```text
ERM-BER DSA:       0.182567 [0.180596,0.184407]
GroupDRO-BER DSA:  0.136834 [0.134952,0.138706]
BER-CVaR DSA:     -0.011240 [-0.012355,-0.010092]
```

BER相对ERM与GroupDRO在4/4客户端均降低DSA；相对CVaR的pooled DSA更低，但c0/c1的CVaR
DSA更低，故不作architecture-uniform DSA dominance主张。BER的grid accuracy在4/4客户端上
均高于其余三臂。完整报告：

```text
deliverables/cle_v2_spurious_final_map2_20260913/RESULT_SUMMARY_ZH.md
```

## 已完成验证

2026-09-13，本地真实CUDA一轮四臂训练、checkpoint、20-source DSA、三组source-bootstrap、
shuffled-binding null和4/4 paired batch trace均执行成功；8项相关单元测试通过。smoke数值不构成
科学证据。随后OpenI 40轮Formal、完整预测缓存分析与独立复算均通过。

## OpenI启动

```text
entry            scripts/openi_cle_v2_spurious_final_entry.py
mode             formal
confirm_formal   true
data_source      ""
skip_install     false
GPU              1 x V100 32GB
```

复用OpenI数据集`CLE_v2_CrossBindingMaps_Seed0_PEW_20260911`。唯一有效输入包：

```text
C:\Users\asus\Desktop\FedPRIME-D2C\local_runs\cle_v2_cross_scenario\cle_hfl_v2_cross_maps1_2_seed0_split0_with_pew.tar.gz
bytes  1385820059
SHA256 BEA8E98737BF881C701DCFFF05F4E04C3A1E6095B7CF7702A5177260C2F186F5
```

Formal实测训练`9344.884s`、分析`93.444s`、总计`9438.328s`（约2小时37分18秒）。

下载：

```text
cle_v2_spurious_final_map2_formal_outputs.tar.gz
RUN_TIMING.json
```

放入：

```text
C:\Users\asus\Desktop\FedPRIME-D2C\outputs\openi_downloads\cle_v2_spurious_final_map2_formal\
```

禁止误用694MB的原map筛选包；本次必须使用1,385,820,059-byte cross-maps包。也禁止使用早期
同时改变partition、PEW并含CDep的cross-scenario包。
