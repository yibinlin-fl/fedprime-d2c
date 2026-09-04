# K1-C-Minimal Formal 独立审计

Date: 2026-09-04

## 最终结论

```text
verdict: NO_GO_CRSF_INTERVENTION
stage1 taxonomy-free: FAIL
stage2 causal DSA:     FAIL
full training:         NOT AUTHORIZED
```

本次失败是可信的科学结果，不是执行故障。CRSF 在 H9/L9、两个预注册架构上均产生方向一致但
很弱的效果：未见 Bank-B 谱集中度约下降 `5.4%`，真实 DSA 约下降 `0.0052`（相对 `2.3%`）。
它们远低于冻结的 `15%` 与 `0.05 or 25%` 门槛，也没有达到相对 RawSpec 的必要优势。因此停止
CRSF，不运行 B-to-A、不补 ResNet12/ShuffleNet、不调学习率/步数/probe，也不进入完整训练。

## 原始产物

```text
archive: outputs/openi_downloads/cle_k1_c_minimal_formal_seed0/cle_k1_c_minimal_seed0_formal_outputs.tar.gz
bytes:   58,124,491
sha256:  E07B9E75E2AEDDE0C1B3A4FF018CE0B4FD90EAA6CB88144D6E0D98588E43D4CA
```

## Stage 1：完整未见 Bank-B

| System | CRSF chi reduction | RawSpec reduction | CRSF-RawSpec | CRSF energy retention | Positive clients |
|---|---:|---:|---:|---:|---:|
| H9 | 5.369% | 1.532% | 3.838 pp | 93.851% | 2/2 |
| L9 | 5.452% | 1.493% | 3.959 pp | 95.096% | 2/2 |

冻结门槛要求 CRSF reduction `>=15%` 且 CRSF-RawSpec `>=10 pp`，两个 system 均失败。

架构分解揭示明显的不均衡：

```text
H9 ResNet10:      8.936% reduction
H9 MobileNetV2:   2.199% reduction
L9 ResNet10:      8.936% reduction
L9 MobileNetV2:   2.412% reduction
```

因此效果主要来自 ResNet10，在 MobileNetV2 上接近微弱扰动，未能证明 model-heterogeneous
通用干预。

## Stage 2：真实 CLE DSA

| System | Frozen DSA | CRSF DSA | Absolute reduction | Relative reduction | CRSF-RawSpec advantage |
|---|---:|---:|---:|---:|---:|
| H9 | 0.225502 | 0.220265 | 0.005237 | 2.322% | 0.005487 |
| L9 | 0.224956 | 0.219764 | 0.005193 | 2.308% | 0.005903 |

冻结门槛要求 absolute `>=0.05` 或 relative `>=25%`，并要求相对 RawSpec 优势 `>=0.02`；均明显
失败。两个 client 虽然同向，但 ResNet10 的 DSA 改善约 `0.009445`，MobileNetV2 仅约
`0.00094--0.00103`，再次说明干预缺乏跨架构强度。

## 报告性任务指标

CRSF 相对 Frozen：

```text
H9: Avg +0.1469, Worst +0.2313, CFG -0.6125, Clean Avg -0.10, Clean Worst -0.30
L9: Avg +0.1531, Worst +0.2313, CFG -0.6500, Clean Avg -0.15, Clean Worst -0.30
```

这些变化很小且按协议不参与 gate，不能用于覆盖 DSA/chi 的主失败。

## 独立完整性审计

- archive 内31项 artifact hash 全部重算一致。
- primary taxonomy-free seal 内18项 hash 全部一致，且不含 oracle、task、result、stage2 文件。
- 72组 unseen mean/energy/Gram 从 NPZ 重算；Gram、chi、energy 最大误差均为 `0`。
- 6组 oracle predictions 独立重算 DSA 与全部任务指标；最大误差均为 `0`。
- 8条 optimization traces 均有5个 accepted steps，accepted objective 单调不增，无 contract
  failure，最大最终 anchor KL 为 `0.0196457 <= 0.02`。
- OpenI 使用的6个 source hash 与本地对应源码逐项一致。
- Formal 使用4个冻结 round-40 checkpoint；未训练模型、未修改通信、未写完整 checkpoint。
- taxonomy-free primary 明确在 oracle/evaluation 打开前封存。

## 科学解释边界

K1-C0 的观察性发现仍成立：CLE checkpoint 存在显著响应谱集中。Minimal Formal 进一步说明：
在当前 late-block、KL约束和预注册低成本 correction 下，压低该谱只能带来很弱、架构不均衡的
DSA 改善。由此不能声称该集中度是 CLE failure 的充分因果杠杆，也不能把 CRSF 提升为训练方法。
