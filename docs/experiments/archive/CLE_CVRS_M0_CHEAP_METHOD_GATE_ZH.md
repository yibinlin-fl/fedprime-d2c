# CLE-HFL CVRS M0 低成本方法生死实验

状态：`COMPLETED_NO_GO_CVRS`。OpenI V100S Formal 已完成；不得调参、补种子或进入完整 HFL。

## 研究问题

CVRS 不识别损坏类型，也不预测具体的类别绑定。它只压制无标签公共图片经过通用 nuisance probe 后，跨不同 carrier 持续朝同一类别方向移动的 logit evidence：

```text
delta_q(u) = P_C[z(A_q(u)) - z(u)]
mu_q       = mean_u delta_q(u)
E_q        = mean_u ||delta_q(u)||^2
L_CVRS     = mean_q ||mu_q||^2 / (stopgrad(E_q) + eps)
```

该比值位于 `[0,1]`。carrier 间互相抵消的普通敏感性几乎不罚；跨 carrier 保持同向的类别路由会被强罚。

## M0 三臂

两个独立上下文分别从 Phase-B0 封存的 H9 round-40 权重开始：

```text
client 0 / ResNet10
client 3 / MobileNetV2
```

三臂为：

```text
baseline：只继续原 AugMix + JSD + DCL 私有训练
jsd：     baseline + 无标签公共原图/PRIME view 成对 JSD
cvrs：    baseline + CVRS
```

历史 artifact 没有 Adam state，因此三臂统一新建 Adam；这是 matched short adaptation，不称为 round-41 无缝续训。

正式训练量固定为 3 local epochs、private batch 64。每 4 个 private optimizer steps 后，JSD/CVRS 臂增加一个独立 public optimizer step。public batch 64，每步 4 个 Bank-A probe；seed `20260905` 对 64 probes 做无放回洗牌，因此每 16 次 public update 完整覆盖一轮 bank。公共步骤使用 eval mode 冻结 BatchNorm running statistics，但保留参数梯度。

JSD 与 CVRS 的 lambda 各自在初始模型的首个匹配 private/public batch 上设置一次：

```text
lambda = 0.1 * ||grad L_task|| / (||grad L_reg|| + eps)
```

随后冻结，不做搜索。

## 公共数据与盲测

复用 K0-B seed `20260901` 封存的 1000 张 CIFAR-100 train images，永不读取标签：

```text
Ua positions 0:500     public regularization only
Ub positions 500:756  held-out routing evaluation only
```

held-out routing 使用 Bank-B 固定 probes `0..15`，共 `256 x 16`，报告同一 CVRS 比值 `R_route`。Bank-B 与 Ub 不参与训练或 lambda 初始化。

训练期间禁止读取 CIFAR-100 label、private corruption type/family/severity/binding。所有训练输出封存后，Formal 才可打开既有 Phase-A0 的 1000-source x 16-operator paired oracle，计算 DSA 与任务指标。

本 M0 中每个架构的 `Avg` 定义为全部 source/operator pair 的平均 top-1 accuracy；`Worst` 定义为 16 个 operator 中最低的 top-1 accuracy。

## Formal gate

ResNet10 与 MobileNetV2 必须分别全部满足：

```text
(DSA_baseline - DSA_cvrs) / |DSA_baseline| >= 20%
DSA_jsd - DSA_cvrs >= 0.02
Avg_baseline - Avg_cvrs <= 1 pp
Worst_baseline - Worst_cvrs <= 1 pp
```

全部通过才是 `GO_TO_FULL_HFL_INTEGRATION`；否则为 `NO_GO_CVRS`，不做 CVRS-v2。`R_route` 用于验证 blind routing 是否同步下降，但不是额外调参入口。

## 入口

OpenI 当前只运行平台 benchmark：

```text
启动文件：scripts/openi_cle_cvrs_m0_entry.py
运行参数：mode = benchmark
数据集文件：cle_cvrs_m0_seed0_inputs.tar.gz
```

不要添加 `confirm-formal`。最终输入包固定为 `109142359` bytes，SHA256 为
`E9427A55DBE2545AF9D5A1EBD8BEA5B18C41C84D7FE89D06674165F4109E3818`。包内只保留
client 0/3 的 private image/label；不包含 private corruption id/type/severity/source metadata。

本地入口：

```bash
python scripts/run_cle_cvrs_m0.py --mode=smoke --device=cuda
python scripts/run_cle_cvrs_m0.py --mode=benchmark --device=cuda
```

Formal 被显式锁住，只有成本获用户确认后才允许：

```bash
python scripts/run_cle_cvrs_m0.py --mode=formal --confirm-formal --device=cuda
```

Smoke 只验证执行，benchmark 只估算成本；二者都不是科学证据。

## OpenI V100S benchmark 结果（2026-09-05）

输入包大小与 SHA256、两个起始 checkpoint、两套 PRIME bank 和所有冻结输入均核验通过。
ResNet10 与 MobileNetV2 各自三臂的 private batch/AugMix trace hash 完全一致；benchmark
没有解压 evaluation 资产，也没有修改通信或解锁 Formal。

```text
hardware: Tesla V100S-PCIE-32GB
mean private step: 0.117390 s
mean public step:  0.124062 s
projected six-arm 3-epoch training: 328.047 s
projected training GPU-hours:       0.091124
archive SHA256: 570BC2B57E8012750DCCD575E617A675E3C492A3E429F6BD25FA202700E6AE7D
verdict: BENCHMARK_ONLY_NO_SCIENTIFIC_DECISION
```

该投影不含 held-out routing 与最终 DSA/accuracy evaluation，因此不能把 328 秒当作整项
Formal 的承诺耗时；但训练部分的成本已明显低于本地 RTX 3050 的 2.163 GPU-hour 投影。
该 benchmark 后续用于授权一次 seed-0 Formal；正式结果如下。

## OpenI Formal 结果（2026-09-05）

输入、checkpoint、Bank-A/B 与 pre-oracle seal 均通过哈希核验，两个架构内三臂的
private batch/AugMix trace 完全一致。训练完成并封存 taxonomy-free 输出后才打开 oracle。

```text
ResNet10:
  DSA baseline / JSD / CVRS = 0.293632 / 0.255995 / 0.219023
  CVRS relative reduction   = 25.409%                         PASS
  JSD - CVRS                = +0.036972 (threshold +0.02)     PASS
  Avg CVRS - baseline       = +0.500 pp                       PASS
  Worst CVRS - baseline     = +2.800 pp                       PASS

MobileNetV2:
  DSA baseline / JSD / CVRS = 0.172066 / 0.116098 / 0.129098
  CVRS relative reduction   = 24.972%                         PASS
  JSD - CVRS                = -0.013000 (threshold +0.02)     FAIL
  Avg CVRS - baseline       = +1.4375 pp                      PASS
  Worst CVRS - baseline     = +6.100 pp                       PASS

verdict: NO_GO_CVRS
full_hfl_training_authorized: false
archive SHA256: 5916858CC71A7AFF1F18B4EEB90F7D3D0C9A13E0B695BFD2F7F7B278861DAB27
```

CVRS 在两个架构上都相对 baseline 降低约 25% DSA，并保住/提高了准确率；但它没有跨架构
稳定优于普通 Public-JSD。MobileNetV2 上 CVRS 的 taxonomy-free routing strength 比 JSD 更低，
真实 DSA 却更高，说明该 proxy 不能稳定标识真正有害的 CLE 路由。因此按预注册门槛冻结为
方法 NO-GO，不允许用 pooled 均值、改阈值或调 lambda 覆盖该失败。
