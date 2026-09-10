# CLE-v2 Pure PEW+BER Stage-2 OpenI 运行说明

状态：seed-0、12-round Formal已完成；I0/L0/P1通过、P2失败，冻结判定为
`NO_GO_PEW_BER_STAGE2_SEED0`。后续零训练架构归因已完成，未启动新训练。

## Formal结果（2026-09-10）

结果包为`cle_v2_plugin_stage2_seed0_formal_outputs.tar.gz`，4,454,630 bytes，SHA256：

```text
1719D2C3801986FCA7BAAEF9CC89AFCCD5D7818470C94AA67C4139FFC993CD79
```

输入审计PASS，两臂48条local batch/AugMix trace完全匹配，CDep未使用。独立复算的主要结果：

```text
pooled DSA: base 0.119644, plugin 0.041252
DSA reduction: 0.078392 (65.52%)
source-bootstrap CI95: [0.077191, 0.079601]
client DSA reductions: [0.111312, 0.011317, 0.115762, 0.075178]

last-5 plugin-minus-base:
Avg +1.7323, Worst -0.3760, WCCA +0.3500, CFG -9.2600
```

`Worst>=+1.0`是P2唯一失败项，故不得事后把正式总判定改成GO。按论文核心目标可另外记录
`CLE mitigation efficacy: GO`，但跨架构效用一致性尚未建立。零训练归因显示client2/ShuffleNet
发生类别选择性预测重分配；由于架构与非IID客户端划分固定绑定，不能归因为模型容量。完整归因：

```text
scripts/analyze_cle_v2_plugin_architecture_attribution.py
deliverables/cle_v2_plugin_stage2_architecture_attribution_20260910/
```

Formal训练8,074.75秒、分析41.77秒，总计8,116.51秒（约2小时15分）。source bootstrap不覆盖
训练seed不确定性。当前不允许通过调BER权重、改门槛或补seed翻案。

## 目标与边界

Stage-2只回答一个问题：在已证明存在强directional shortcut的固定CLE-v2 HFL场景中，纯
PEW+BER本地插件是否在降低真实operator-level DSA的同时改善任务效用。

```text
h9_b = gamma0.9 + AugMix/JSD/DCL + strict AsymHFL-val
h9_p = h9_b + frozen public PEW annotations + hard BER
```

候选保留原DCL，不改变服务器通信、模型、数据、初始化、fit/audit角色或测试协议。CDep被显式
禁止，配置中出现`cdep`会立即失败。真实operator/binding只在最终报告用paired DSA中读取，不进入
PEW、BER、本地梯度、路由、选择或调参。粒度消融、其他底座、多seed、40轮和跨scenario均不在
本阶段范围。

## 数学对象与最弱假设

主要estimand为同一强CLE场景下的匹配插件效应：

```text
Plugin DSA reduction = DSA(h9_b) - DSA(h9_p)
Utility delta        = metric(h9_p) - metric(h9_b)
```

它识别的是目标场景中的插件效用，不是完整的gamma difference-in-differences；因此不能单凭本实验
宣称插件只作用于CLE而不影响gamma0。最弱解释假设是：两臂除预注册PEW+BER字段外完全匹配；
公共PEW与私有最终标签/真实operator metadata隔离；paired DSA对同一source只改变operator；两臂均
越过学习门。可观测的真正有害routing由evaluation-only真实binding下的operator-level DSA识别，
PEW预测准确率或BER loss下降本身不能替代DSA。

## 冻结协议

```text
scenario: fixed CLE-v2 seed0_split0, gamma=0.9
training seed: 0
rounds: 12
local batches/client/round: 16
batch size: 64
optimizer: Adam, lr=0.001
public batches/round: 4
models: ResNet10, ResNet12, ShuffleNet, MobileNetV2
routing: client-private audit only
training gradients: private fit only
reporting metrics: complete test split each round; labels never affect training/routing
final paired DSA: 1,000 sources × 15 operators × 4 clients, severity 3
source-paired bootstrap: 2,000
num_workers: 0
```

PEW/BER冻结参数沿用已审计公共PEW包：hard assignment、unknown threshold `0.27`、
`support_gamma=0.5`、`count_cap=32`、`min_group_count=2`。不得在结果后修改阈值、BER权重、门槛或
窗口救结果。

## 冻结门槛

- `I0`：两臂local batch/AugMix trace完全匹配。
- `L0`：两臂最终operator-grid pooled accuracy均`>=20%`。
- `P1`：`DSA(h9_b)-DSA(h9_p)>=0.02`，source-bootstrap CI95下界`>0`，4/4客户端方向为正。
- `P2`：candidate-minus-base last-5 `Avg>=+1.5`、`Worst>=+1.0`、`WCCA>=0`、`CFG<=-1.0`。

Formal中四门全部通过才记为`GO_PEW_BER_STAGE2_SEED0`，否则为
`NO_GO_PEW_BER_STAGE2_SEED0`。不得用补seed、改窗口或改参数翻案。即使GO，也只证明固定场景、
training seed 0、12轮下纯PEW+BER有效；40轮耐久性、训练seed稳定性、跨scenario和插件跨底座仍需
单独证据。

## 已完成验证

- `tests/test_cle_v2_plugin_stage2.py + tests/test_cle_v2_mechanism_stage1.py`：9/9 PASS。
- 正式输入审计PASS：40,000私有样本、1,000×15 DSA grid、公共PEW checkpoint/hash与annotations
  完整，PEW checkpoint SHA256为
  `BC9FF7523B8474774B36E02507A544FEBD76363772E9B865191FD3119960DCBB`。
- 真实RTX 3050 CUDA一批次两臂smoke完成；两臂各保存4个最终checkpoint。
- PEW annotations加载成功；BER实际触发，四客户端有效组数分别为36/49/44/54；CDep未启用。
- 两臂4条local trace完全匹配；20-source × 15-operator最终DSA/bootstrap分析链完成。
- Smoke数值是执行验证，不是科学证据。

Windows本地必须使用完整Conda启动：

```powershell
& "D:\anaconda3\Scripts\conda.exe" run --no-capture-output -n pytorch python ...
```

不要直接调用`D:\anaconda3\envs\pytorch\python.exe`。Codex进程PATH优先包含另一套native DLL，
直接调用曾以Windows原生退出码`0xC06D007F`在Matplotlib渲染时崩溃并弹窗；完整Conda启动下
Matplotlib保存图与RTX 3050 CUDA运算均通过。

## OpenI Formal 启动卡

```text
commit: 使用最终启动卡所列、已push到origin/main的Stage-2提交
dataset: 复用 CLE_v2_Factorial_Seed0_PEW_20260909
upload: 不需要重新上传
input archive: cle_hfl_v2_paired_factorial_seed0_split0_with_pew.tar.gz
input bytes: 694118746
input sha256: B6C802DA0BC2A95183DB91D2855E35D9A807B39441F2A24CF88514EB1CAD69AA
startup: scripts/openi_cle_v2_plugin_stage2_entry.py
parameter mode: formal
parameter confirm_formal: true
parameter data_source: 留空
parameter skip_install: false
GPU: V100 32GB × 1
run type: Formal
expected time: about 2.5--3.5 V100 GPU-hours
output: cle_v2_plugin_stage2_seed0_formal_outputs.tar.gz
download folder: C:\Users\asus\Desktop\FedPRIME-D2C\outputs\openi_downloads\cle_v2_plugin_stage2_seed0_formal\
```

本次唯一有效入口是`scripts/openi_cle_v2_plugin_stage2_entry.py`。禁止误用旧八臂入口
`scripts/openi_cle_v2_factorial_entry.py`、Stage-1入口
`scripts/openi_cle_v2_mechanism_stage1_entry.py`，或旧八臂输出
`cle_v2_factorial_seed0_formal_outputs.tar.gz`。
