# CLE-HFL K0-A：Public-Carrier Transfer Oracle

Updated: 2026-09-01

## 1. 唯一问题

K0-A 只判断：已有 CIFAR-10 clean-carrier PIDR 所读出的 CLE 定向捷径，能否迁移到
**任务标签空间不同**的 CIFAR-100 公共载体。CIFAR-100 与 CIFAR-10 仍有相近视觉语义，
不得写成“完全无关语义”。

本阶段是零训练 oracle mechanism audit：不更新模型、不修改通信、不训练 DME、不进入 K0-B。

## 2. 冻结资产

```text
arms:       H0/H9/L0/L9
clients:    4 heterogeneous frozen round-40 classifiers per arm
models:     ResNet10 / ResNet12 / ShuffleNet / MobileNetV2
carriers:   CIFAR-100 train, fixed 1,000 indices, labels unused
public RNG: 20260901
probes:     existing 16 CLE operators, severity=3
probe RNG:  20260830
```

复用 Phase-B0 输入包，不重新上传数据：

```text
archive: cle_public_canonicalization_phase_b0_seed0_inputs.tar.gz
bytes:   535256689
sha256:  DFB766F6494A5F61AA16F45666EC250A30501066AB54D89C984CD2324293B9BC
```

Phase-B0 已返回的概率缓存使用 CIFAR-10 carriers，不能代替本次 CIFAR-100 推理。

## 3. Blind response

对 client/model `i`、class `c`：

\[
m_{i,c}(x)=z_{i,c}(x)-\log\sum_{k\ne c}\exp z_{i,k}(x).
\]

对 public carrier `U_j` 与 operator `o`：

\[
\Delta_{i,j,o}=m_i(T_o(U_j))-m_i(U_j),\qquad
r_{i,j,o}=\Delta_{i,j,o}-\frac1C\mathbf1\mathbf1^\top\Delta_{i,j,o}.
\]

主方向矩与 coherence：

\[
\mu_{i,o}=\frac1M\sum_j r_{i,j,o},\qquad
\kappa_{i,o}=\frac{\|\mu_{i,o}\|_2^2}
{M^{-1}\sum_j\|r_{i,j,o}\|_2^2+\epsilon}.
\]

主判定只用 centered class-vs-rest logit response。Centered raw-logit 与 probability response
仅作 secondary robustness check。新旧 PIDR 的 mAP/AUC/hit 可以比较；其 pooled magnitude
不可直接比较，因为旧 PIDR 使用 task label 排除 target-class sources，而 K0-A 不使用
CIFAR-100 labels。

## 4. 真值隔离

response 生成阶段只允许读取 operator identity，不允许读取 operator-family 或 CLE binding。
16个 classifier 的 response 文件全部保存并写入 SHA256 manifest 后，评分阶段才局部导入真值。

真实结构是：

```text
operator -> family
client x class -> bound family
```

不是单一的 `operator -> class`。Oracle truth 必须复用旧 PIDR：

```python
binding[:, None, :] == operator_family_ids[None, :, None]
```

## 5. 冻结正式门槛

HFL 与 Local 分别全部满足：

```text
gamma9 mAP >= 0.65
gamma9 - gamma0 mAP >= 0.20
positive mAP clients >= 3/4
gamma9 class-to-family hit >= 0.70
class-map and probe-identity shuffled null p <= 0.01
paired-carrier bootstrap CI95 lower bound of directional-strength delta > 0
paired-carrier bootstrap CI95 lower bound of coherence delta > 0
gamma9 fixed 500/500 split-carrier cosine > 0
gamma9 - gamma0 split-carrier cosine > 0
positive split-cosine clients >= 3/4
```

Positive mAP client 沿用旧 PIDR：对应 client 的 `mAP(gamma9)-mAP(gamma0)>0`。
Bootstrap 固定1,000次，binding permutation 固定1,000次。不得在结果后修改门槛。

```text
all pass: GO_TO_K0_B
otherwise: NO_GO_PUBLIC_CARRIER_ROUTE
```

通过只允许设计 K0-B taxonomy-free generic probes；禁止自动训练或进入 K1。

## 6. 实现与运行

```text
engine:   fedprime/engine/cle_public_carrier_moment.py
analyzer: scripts/analyze_cle_public_carrier_k0a.py
OpenI:    scripts/openi_cle_public_carrier_k0a_entry.py
tests:    tests/test_cle_public_carrier_k0a.py
```

OpenI 默认 smoke：

```bash
python scripts/openi_cle_public_carrier_k0a_entry.py --mode=smoke
```

Smoke 固定为8 carriers、前2个operators、client0、H0/H9，只验证 checkpoint strict-load、
deterministic probes、logit/response形状、有限值、NPZ round-trip、盲化评分接口与输出打包。
必须输出 `SMOKE_ONLY_NO_SCIENTIFIC_DECISION`。

Smoke 经审计并由用户批准后，正式运行：

```bash
python scripts/openi_cle_public_carrier_k0a_entry.py --mode=formal
```

正式模式固定1,000 carriers、16 operators、16 classifiers、1,000 bootstrap与1,000
permutations。它只有前向推理，没有梯度、优化器或模型写回。

## 7. 产物

```text
outputs/cle_public_carrier_k0a_seed0_<mode>/
  config.json
  selected_public_indices.npy
  blind_response_manifest.json
  responses/<arm>_client<id>.npz
  metrics/arm_metrics.csv
  metrics/per_client_metrics.csv
  metrics/bootstrap.json
  metrics/permutation_test.json
  result.json
  final_report.md
```

原始产物保留在 `outputs/` 且不跟踪；正式结果返回后再将独立复算报告写入
`deliverables/`。
