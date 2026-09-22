# CLE-HFL 投稿前实验：执行就绪清单

Updated: 2026-09-22

> 2026-09-22 result update：M1 seeds 1/2已完成并与seed 0聚合为
> `GO_MAP2_TRAINING_SEED_STABILITY`。M2可按本文参数直接Formal；M3、S1、
> S2和JTT必须benchmark-first。核心matched实验从共同`initial_states`开始，任务模型
> `pretrain_epochs=0`，不得临时加入RAHFL式40轮本地预训练破坏归因。PEW是独立公共预训练对象：
> M1/M2复用冻结PEW，M3在CIFAR-10公共图像上训练一次并冻结，S1训练排除motion_blur的新PEW。
> 当前S2是40轮protocol-matched context table，不是RAHFL官方40+40完整recipe；若论文需要
> official-budget参考，必须另立S2b协议、先benchmark并单独报告，不能混入M1主表。

## 1. 状态边界

本轮只完成实验工程、数据准备、审计、分析和OpenI入口；没有启动benchmark或Formal训练。
Smoke/benchmark不构成科学证据，Formal仍需用户明确授权。所有主实验禁用CDep，不修改冻结
PEW/BER超参数，不使用final-test标签进行路由、选择、早停或调参。

定向测试结果：Python静态编译通过，38项相关测试通过；M1、M2、S2、JTT已用真实输入包完成
`prepare-only`契约检查。新CIFAR-100数据包完成独立数据审计：100类、1000个source、15个
operator、固定severity=3的完整paired grid，状态`PASS`。

## 2. 推荐执行顺序

```text
M2 local-first training seeds 1/2
-> M3 CIFAR-100 benchmark
-> 经成本确认后 M3 Formal seeds 0/1/2
-> S1 bounded taxonomy benchmark/Formal
-> S2-v2 ten-arm HFL context benchmark/Formal
-> 仅在正文需要正式比较时运行 O1 JTT 或 O3 KT/FCCL插件实验
```

M1已经完成；M2已有同协议历史成本信息，可在用户明确授权后直接Formal；M3/S1/S2/JTT是新的长协议，
必须先benchmark。不要一次性提交所有Formal。

## 3. M1：held-out map2 四臂40轮多training-seed（完成）

2026-09-22 training seeds 1/2返回并通过审计，与seed 0聚合后四项稳定性门槛全部PASS：

```text
verdict: GO_MAP2_TRAINING_SEED_STABILITY
report: deliverables/cle_hfl_map2_multiseed_20260922/RESULT_SUMMARY_ZH.md
```

回答：seed-0最强结果是否可跨训练随机性复现；BER降低DSA是否仍优于ERM和共享同一PEW分组的
GroupDRO；相对CVaR报告shortcut--utility trade-off，不要求每客户端DSA全胜。

```text
dataset display: CLE_v2_CrossBindingMaps_Seed0_PEW_20260911
upload: reuse existing dataset
local source: C:\Users\asus\Desktop\FedPRIME-D2C\local_runs\cle_v2_cross_scenario\cle_hfl_v2_cross_maps1_2_seed0_split0_with_pew.tar.gz
bytes: 1385820059
sha256: BEA8E98737BF881C701DCFFF05F4E04C3A1E6095B7CF7702A5177260C2F186F5
entry: scripts/openi_cle_v2_spurious_final_entry.py
mode = formal
train_seed = all
confirm_formal = true
data_source = ""
skip_install = false
GPU: 1 x V100 32GB
type: Formal, seeds 1/2; seed0 already complete
outputs: cle_v2_spurious_final_map2_trainseed1_formal_outputs.tar.gz
         cle_v2_spurious_final_map2_trainseed2_formal_outputs.tar.gz
download folder: C:\Users\asus\Desktop\FedPRIME-D2C\outputs\openi_downloads\m1_map2_multiseed
```

四臂：ERM、CVaR-DRO、PEW+GroupDRO、PEW+BER。冻结40轮、每客户端每轮16个local batches、
map2、partition seed0和评价source。聚合器：

```text
scripts/analyze_cle_submission_multiseed.py --kind map2
```

可证明：训练随机性下的方向稳定性。不能证明：partition、dataset或现实部署稳定性。

## 4. M2：HFL-vs-Local四臂多training-seed

回答：`delta_Local`是否在seeds 1/2仍解释`delta_HFL`的主要部分，从而把local-first由seed-0
发现提升为训练随机性下的复现。四臂只含HFL/Local × gamma=0/0.9，不再误跑插件四臂。

```text
dataset display: CLE_v2_Factorial_Seed0_PEW_20260909
upload: reuse existing dataset
local source: C:\Users\asus\Desktop\FedPRIME-D2C\local_runs\cle_v2_factorial\cle_hfl_v2_paired_factorial_seed0_split0_with_pew.tar.gz
bytes: 694118746
sha256: B6C802DA0BC2A95183DB91D2855E35D9A807B39441F2A24CF88514EB1CAD69AA
entry: scripts/openi_cle_v2_factorial_entry.py
mode = formal
factorial_scope = mechanism
train_seed = all
confirm_formal = true
data_source = ""
skip_install = false
GPU: 1 x V100 32GB
type: Formal, seeds 1/2; 12 rounds
outputs: cle_v2_factorial_mechanism_trainseed1_formal_outputs.tar.gz
         cle_v2_factorial_mechanism_trainseed2_formal_outputs.tar.gz
download folder: C:\Users\asus\Desktop\FedPRIME-D2C\outputs\openi_downloads\m2_local_first_multiseed
```

主要contrast：`delta_HFL`、`delta_Local`、`delta_Comm=delta_HFL-delta_Local`。比例只作描述，
不解释为样本级因果中介比例。聚合器使用`--kind local-first`。

## 5. M3：第二private dataset（CIFAR-100）

回答：CLE directional shortcut、paired DSA和PEW+BER缓解是否只存在于CIFAR-10 private task。
新协议包含gamma=0 ERM零基准、gamma=.9 ERM、CVaR-DRO和PEW+BER，三组matched training seeds。
为避免CIFAR-100公共/私有source重叠，PEW不复用主实验中由CIFAR-100训练的checkpoint；OpenI
只用包内CIFAR-10公共图像训练一次PEW并冻结，随后为三个matched training seeds共享同一组
checkpoint和私有annotations。PEW仍不读取CIFAR-10类别标签或CIFAR-100任务标签。

```text
dataset display: CLE_v2_CIFAR100_Submission_Seed0_20260917
upload: new dataset required
local upload: C:\Users\asus\Desktop\FedPRIME-D2C\local_runs\cle_cifar100_submission_clean\cle_hfl_v2_cifar100_factorial_seed0_split0_input.tar.gz
bytes: 681493067
sha256: 650C3363B708554EB164BDEB94A3566B98CADBFF697FC093B9DE7F061C7DA708
entry: scripts/openi_cle_cifar100_submission_entry.py
benchmark parameters:
  mode = benchmark
  train_seed = 0
  confirm_formal = false
  data_source = ""
  skip_install = false
formal parameters after benchmark approval:
  mode = formal
  train_seed = all
  confirm_formal = true
  data_source = ""
  skip_install = false
GPU: 1 x V100 32GB
outputs: cle_cifar100_submission_trainseed{0,1,2}_{benchmark|formal}_outputs.tar.gz
download folder: C:\Users\asus\Desktop\FedPRIME-D2C\outputs\openi_downloads\m3_cifar100_submission
```

Formal固定40轮、16 local batches。主要指标为gamma0 DSA、gamma.9 ERM DSA、ERM-BER DSA、
CVaR-BER DSA、operator-grid accuracy和last-10 utility。聚合器使用`--kind cifar100`。
只能支持“第二个受控图像任务复现”，不能支持医院/汽车真实部署。

禁止使用旧包：

```text
local_runs/cle_hfl_v2_second_dataset/cle_hfl_v2_prepared_cifar100_alpha05_gamma09_seed0_split0.tar.gz
local_runs/cle_cifar100_submission/cle_hfl_v2_cifar100_factorial_seed0_split0_input.tar.gz
```

第一个旧包缺少当前gamma0、source-paired operator grid和三臂Formal协议；第二个废弃包携带由
CIFAR-100公共图像训练的PEW reuse source，存在公共/私有source重叠风险。唯一有效包是上述
`681493067`-byte clean输入包。

## 6. S1：bounded taxonomy stress

回答：当PEW公共训练没有见过`motion_blur`，但该operator仍出现在private CLE时，BER的shortcut
抑制如何退化。OpenI从主数据包现场训练一个排除`motion_blur`的PEW，再比较ERM、CVaR-DRO、
PEW+BER。报告PEW在该operator上的family accuracy/unknown rate、整体DSA、operator-specific DSA
和utility。

```text
dataset display: CLE_v2_Factorial_Seed0_PEW_20260909
dataset/upload/hash: 与M2相同，复用
entry: scripts/openi_cle_taxonomy_stress_entry.py
benchmark: mode=benchmark, confirm_formal=false, data_source="", skip_install=false
formal:    mode=formal,    confirm_formal=true,  data_source="", skip_install=false
GPU: 1 x V100 32GB
output: cle_taxonomy_stress_motion_blur_{benchmark|formal}_outputs.tar.gz
download folder: C:\Users\asus\Desktop\FedPRIME-D2C\outputs\openi_downloads\s1_taxonomy_stress
```

无论成功或失败都只确定一个预注册operator的适用边界，不证明未知、复合、连续corruption的
开放世界鲁棒性。

## 7. S2-v2：protocol-matched HFL context table

回答：CLE不是只在一个自定义通信实现上出现，并将论文放回HFL文献坐标。十个独立方法为
Local/ERM、FedMD adapter、FedProto adapter、FedTGP adapter、FedDF-fidelity、KT-pFL-fidelity、
FCCL protocol-matched adapter、RHFL adapter、AugHFL-fidelity、RAHFL anchor。该表不安装
PEW+BER，也不承担BER归因。FedTGP只能称protocol-matched core adapter；完整审计见
`docs/research/baselines/CLE_HFL_S2_V2_AND_FEDTGP_KILL_TEST_2026_09_22_ZH.md`。

```text
dataset display/upload/hash: 与M2相同，复用
entry: scripts/openi_cle_hfl_context_entry.py
benchmark: mode=benchmark, confirm_formal=false, data_source="", skip_install=false
formal:    mode=formal,    confirm_formal=true,  data_source="", skip_install=false
GPU: 1 x V100 32GB
output: cle_hfl_context_{benchmark|formal}_outputs.tar.gz
download folder: C:\Users\asus\Desktop\FedPRIME-D2C\outputs\openi_downloads\s2_hfl_context
```

Formal固定40轮。FedDF/KT-pFL/AugHFL标为fidelity adapter；FedMD/FedProto/FedTGP/FCCL/RHFL
只能写protocol-matched adapter或core adapter，不得写成各论文完整官方recipe复现。当前不运行
单独的AugHFL/RAHFL官方40+40表。

## 8. 条件性O1：JTT Formal

仅当正文需要正式声称优于JTT时运行。JTT先跑完整ERM、从fit错误生成冻结mask，再从共同初始
状态重训，因此总训练成本约为普通单阶段方法的两倍，必须披露。

```text
dataset display/upload/hash: 与M2相同，复用
entry: scripts/openi_cle_jtt_formal_entry.py
benchmark: mode=benchmark, confirm_formal=false, data_source="", skip_install=false
formal:    mode=formal,    confirm_formal=true,  data_source="", skip_install=false
GPU: 1 x V100 32GB
output: cle_jtt_formal_{benchmark|formal}_outputs.tar.gz
download folder: C:\Users\asus\Desktop\FedPRIME-D2C\outputs\openi_downloads\o1_jtt_formal
```

## 9. 已有O3：KT-pFL/FCCL插件验证

原四臂runner和本地smoke已存在；本轮没有把它混入S2。它回答的是在固定通信底座上添加
PEW+BER后的matched A/B，而S2回答standalone HFL背景表现。只有论文保留“跨多个底座的插件”
强主张时才运行Formal。

```text
entry: scripts/openi_cle_v2_kt_fccl_plugin_entry.py
guide: docs/experiments/current/CLE_V2_KT_FCCL_PEW_BER_PLUGIN_ZH.md
download folder: C:\Users\asus\Desktop\FedPRIME-D2C\outputs\openi_downloads\o3_kt_fccl_plugin
```

## 10. 代码入口

```text
scripts/analyze_cle_submission_multiseed.py
scripts/analyze_cle_v2_local_first.py
scripts/prepare_cle_cifar100_submission_data.py
scripts/audit_cle_cifar100_submission.py
scripts/run_cle_cifar100_submission.py
scripts/analyze_cle_cifar100_submission.py
scripts/openi_cle_cifar100_submission_entry.py
scripts/run_cle_taxonomy_stress.py
scripts/analyze_cle_taxonomy_stress.py
scripts/openi_cle_taxonomy_stress_entry.py
scripts/run_cle_hfl_context.py
scripts/analyze_cle_hfl_context.py
scripts/openi_cle_hfl_context_entry.py
scripts/run_cle_jtt_formal.py
scripts/analyze_cle_jtt_formal.py
scripts/openi_cle_jtt_formal_entry.py
```
