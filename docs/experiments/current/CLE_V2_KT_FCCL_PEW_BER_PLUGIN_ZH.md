# CLE-HFL v2 KT-pFL/FCCL × PEW+BER 插件验证

Updated: 2026-09-12

状态：实现与本地真实CUDA smoke已完成；OpenI benchmark与Formal均未授权。

## 基座状态与解耦原则

KT-pFL和FCCL不是新实现。仓库已分别存在：

```text
fedprime/communication/baselines.py
  KTPFLFidelityCommunicationStrategy
  FCCLCommunicationStrategy
```

本实验不修改两者的通信算法。只在独立实验组合层把同一通信基座分别连接到普通CE和冻结的
PEW+BER本地目标。PEW/BER逻辑不得写入KT-pFL或FCCL类；实验合同显式记录
`baseline_communication_implementations_modified=false`。

## 四臂

```text
kt_b = standard CE + identical kt_pfl_fidelity communication
kt_p = PEW-grouped hard BER-weighted CE + identical kt_pfl_fidelity communication

fc_b = standard CE + identical FCCL cross-correlation communication
fc_p = PEW-grouped hard BER-weighted CE + identical FCCL cross-correlation communication
```

所有臂禁用AugMix/JSD/DCL/CDep；两臂复用相同公开通信参数、初始化、私有batch轨迹、公共batch
预算和strict fit/audit/test角色。真实operator与binding只用于最终paired DSA评价。

KT-pFL只称`equation-oriented KT-pFL fidelity adapter`；FCCL只称`public cross-correlation core
adapter`，均不宣称原论文完整端到端recipe复现。

## 预算

```text
smoke:     1 round x 1 local batch/client
benchmark: 1 round x 8 local batches/client
formal:    12 rounds x 16 local batches/client/round
```

Smoke/benchmark只验证执行与成本，不能作为论文证据。Formal有双锁且仍未授权。

2026-09-12本地真实CUDA smoke完成四臂训练、checkpoint、paired DSA推理和分析。KT与FCCL
两组的Base/Plugin本地batch轨迹均逐项匹配；输入审计和回归测试通过。由于只有1轮、每客户端
1个本地batch，模型尚未形成可识别的directional shortcut，所有smoke数值仅用于执行验证，
不得进入论文表格或决定插件有效性。

## Formal冻结门槛草案

每个通信基座独立判定：

```text
I0: Base/Plugin paired local trace完全匹配
L0: 两臂operator-grid pooled accuracy均 >= 20%
C1: Base DSA >= 0.05，且高于shuffled-binding null p95，p <= 0.01
C2: DSA reduction >= 0.02，CI95 lower > 0，4/4客户端同向
U0: last-5 Avg delta >= 0，且plugin grid accuracy不低于base 1pp以上
```

C2与U0分开报告，防止用准确率覆盖shortcut抑制，也防止把DSA下降误写为通用无损插件。

## 入口

```text
scripts/run_cle_v2_kt_fccl_plugin.py
scripts/analyze_cle_v2_kt_fccl_plugin.py
scripts/openi_cle_v2_kt_fccl_plugin_entry.py
tests/test_cle_v2_kt_fccl_plugin.py
```

复用既有`cle_hfl_v2_paired_factorial_seed0_split0`正式输入包和冻结PEW，不生成或上传新数据集。

## 已完成验证

```text
22 passed
四臂1-round CUDA smoke: PASS
KT Base/Plugin paired trace: MATCHED
FCCL Base/Plugin paired trace: MATCHED
分析器与shuffled-binding/null/bootstrap链路: PASS
baseline_communication_implementations_modified=false
scientific verdict=SMOKE_ONLY_NO_SCIENTIFIC_DECISION
```

下一步只有在用户确认上述Formal门槛后才冻结协议；随后如需估算成本，只启动OpenI benchmark。
不得从本次smoke直接跳到Formal。
