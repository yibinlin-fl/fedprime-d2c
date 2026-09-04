# CLE-HFL CVRS M0 低成本方法生死实验

状态：`FROZEN_BEFORE_FORMAL`。当前只允许实现、单元测试、tiny smoke 与 wall-clock benchmark；未经用户确认不得运行 Formal。

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
