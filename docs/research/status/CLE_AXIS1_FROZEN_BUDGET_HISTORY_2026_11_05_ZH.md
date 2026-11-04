# 已完成Axis I的40×16：历史定义与预算解释

日期：2026-11-05。用途：解决“40轮是否40个epoch”和“当前runner是否代表旧实验”的混淆。
本轮只读取局部历史记忆、正式合同/配置和结果报告，没有修改实验代码、数据、论文或启动训练。

## 1. 历史事实

| 时间 | 已记录的执行/决策 |
|---|---|
| 2026-09-13之前 | 原binding map五臂12轮screen，16 local batches；用于方法选择 |
| 2026-09-13 | 筛选未见的map2，四目标40轮×16，training seed0完成；四臂全部冻结门槛通过 |
| 2026-09-22 | 同map2/partition/初始化，补training seeds1/2；仍为每臂40轮×16 |
| 2026-09-28 | 历史记忆明确指出该预算约4.8个本地epoch，不代表充分训练；提出full-epoch复验 |
| 2026-10-12 | Git提交3818a8e将通用runner从full epoch改为64-batch预算，以控制成本 |
| 2026-10-15 | 新优先级明确保留已完成M1作为Axis I固定预算正式证据，高预算复验降为可选 |
| 2026-11-04/05 | JTT审计核实：论文四臂原始配置仍是40×16，不能用今天runner的64默认值替代 |

9月28日的“旧数值不作为充分训练的accuracy终值”与10月15日的“保留固定预算机制证据”
针对不同主张。历史中确有决策调整，不能只摘某一天的状态作为当前永久结论。

## 2. 40×16准确含义

```text
每个方法、每个training seed：
  4个客户端模型
  40个联邦训练轮次
  每轮每客户端local_epochs=1，但max_local_batches=16提前截断
  每个batch 64张图
  每轮另有4个public batches，public batch size128
  pretrain_epochs=0
```

因此40是联邦循环次数，不是40个完整private-fit epoch。local_epochs=1不会绕过16-batch上限。
16指batch数，64指每个batch的样本数。一次联邦轮次包含既有公共通信/蒸馏与本地阶段，
不能将本地图次计算冒充全部计算量。

## 3. 当时数据规模与名义训练量

9月13日正式报告记录private samples=40000，即四客户端各10000张准备数据。
9月28日记忆记录map2的实际fit数量为8498/8500/8499/8497，总计33994；其余为持久化audit。
本轮引用这些历史计数，没有重新解压全部输入包或重新枚举训练样本。

按全部轮次跑满上限计算：

| 范围 | 名义local更新数 | 名义私有图次 |
|---|---:|---:|
| 一个client一轮 | 16 | 1024 |
| 一个client40轮 | 640 | 40960 |
| 四client，一个方法、一个seed | 2560 | 163840 |

总图次/总fit样本数=`163840/33994=4.819674...`，约4.82个等效fit遍历。
这是预算换算，不是独立新图片数量，也不是对每张图片曝光次数的实测。
训练loader会shuffle，每轮不固定只用同一1024张；但没有逐样本全程计数，不能保证所有fit图片
都已出现或出现次数一致。旧160条local trace为每轮四client的首batch对齐记录，不是只有160个
训练batch，也不足以独立证明每一个optimizer step均已执行。

## 4. 四目标与数据身份

ERM / CVaR-DRO / PEW+GroupDRO / PEW+BER；CIFAR-10 private、CIFAR-100 public；
strict AsymHFL-val；同map2、partition0、evaluation seed20260909、fit/audit split和初始权重。
训练随机性用seeds0/1/2。standard单视图保留基础ColorJitter/裁剪/翻转，未加入AugMix/JSD/DCL/CDep。
alpha=.5、rho1、gamma.9；它不同于后来alpha1/rho.5且40×64的主评价协议。

## 5. 当时要回答什么与已有结果

从12轮screen延长到40轮并换map2，用相同预算比较：
BER收益能否延续到更长训练和新binding map；是否超出共享PEW分组的GroupDRO；
与taxonomy-free CVaR形成何种DSA--utility关系。之后三seed检验训练随机性。
原文档没有证明“16已足够收敛”，不能把早期低成本设计事后改说成充分训练方案。

三seed正式mean±sample SD：

| 方法 | DSA | Operator-grid accuracy (%) |
|---|---:|---:|
| ERM | .250530±.008670 | 20.4950±.3718 |
| CVaR-DRO | .086654±.005514 | 19.2206±1.1055 |
| PEW+GroupDRO | .205417±.008045 | 21.4517±.5783 |
| PEW+BER | .071470±.011288 | 25.1772±1.2522 |

支持固定受控场景、固定预算下的方法比较和稳定性，不单独保证充分训练后的最终排序。
低accuracy不能仅归因于corruption难度；预算、数据与模型/通信均可能影响。

## 6. 时间记录与来源

每项时间是一个seed的四臂任务整体，包含分析，不是单臂时间：seed0约2h37m18s；
seed1约2h43m31s；seed2约2h42m37s。不能套用到后来的40×64主评价或full-epoch任务。

- `docs/project/CURRENT_PROJECT_MEMORY.md`：2026-09-13 screen/Formal、2026-09-22多seed、
  2026-09-28预算修复三个局部段落。
- `docs/experiments/current/CLE_HFL_PENDING_EXPERIMENT_PRIORITY_2026_10_15_ZH.md`：P0-1保留M1及高预算复验定位。
- `outputs/openi_downloads/cle_v2_spurious_final_map2_formal/extracted/configs/SPURIOUS_FINAL_MAP2_CONTRACT.json`及对应实际configs。
- `deliverables/cle_v2_spurious_final_map2_20260913/RESULT_SUMMARY_ZH.md`。
- `deliverables/cle_hfl_map2_multiseed_20260922/RESULT_SUMMARY_ZH.md`。

以后复跑旧四臂或补入JTT主表，以原40×16合同为准；高预算实验单独命名/比较。
原四臂结果保留，不能把runner后来的参数变化追溯写入历史结果。
