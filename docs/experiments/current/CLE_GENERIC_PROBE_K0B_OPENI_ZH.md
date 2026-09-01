# CLE K0-B v2：Taxonomy-Free Generic Probe Gate

Updated: 2026-09-02

## 1. 目的与边界

K0-A 已正式 `GO_TO_K0_B`，但同时发现 H0/L0 的普通 split cosine 也很高。因此 K0-B
不再把“方向可复现”本身当作 CLE 信号，而是检测：

```text
carrier-stable + class-selective directional response
```

本阶段只复用 K0-A 完全相同的16个 H0/H9/L0/L9 round-40 checkpoint 和同一批1,000张
CIFAR-100公共图片。禁止训练、fine-tune、checkpoint写回、DME/K1、PNCB、PEW/BER和通信修改。

Primary inference/scoring 禁止读取 CIFAR-C operator、corruption type、family、severity、
CLE binding 或 private corruption metadata。CIFAR-100标签不加载到分析逻辑。

## 2. 公共载体与冻结模型

复用的 OpenI 输入：

```text
archive: cle_public_canonicalization_phase_b0_seed0_inputs.tar.gz
bytes:   535256689
sha256:  DFB766F6494A5F61AA16F45666EC250A30501066AB54D89C984CD2324293B9BC
```

公共载体仍以 `seed=20260901` 从 CIFAR-100 train 无放回选择1,000张，正式索引 hash 必须为：

```text
731B8CFFDCBD241474D33B261E323F9EC11C2EA59BC7705261140A3B8572F6CA
```

按冻结顺序拆为：

```text
Ua = indices[0:500]
Ub = indices[500:1000]
```

两半互斥，禁止结果后重排。

## 3. Frozen PRIME generic banks

PRIME 只作为 model-agnostic nuisance generator。使用与项目 PRIME 相同的三类数学 primitive：

```text
diffeomorphic spatial field
smooth color transform
random convolutional filter
```

每个 recipe 固定：3条 mixture chains、每条1--3层、逐层 primitive identity、全部 primitive
参数与数组、Dirichlet mixture weights 和 Beta mixing coefficient。同一个 recipe state 对所有
carrier、arm、client完全复用；推理阶段不允许采样任何 transform state。

分布冻结为：

```text
mixture weights: Dirichlet(1,1,1)
mixing:         Beta(1,1)
depth:          Uniform{1,2,3}
primitive:      Uniform{diffeo,color,filter}
diffeo:         PRIME beta cut/temperature + Gaussian spectral coefficients
color:          cut Uniform{1,...,100}, T Uniform[0,0.01], Gaussian coefficients
filter:         kernel size Uniform{3,5}, sigma Uniform[0,4], Gaussian kernel + identity impulse
```

两套独立 bank：

```text
bank A: 64 recipes, seed 20260902
canonical bank SHA256: 6CAE529D4240715162B19B3968D47FA037A940B4D52D688FF52B859C5523DC01

bank B: 64 recipes, seed 20260903
canonical bank SHA256: 4A53497EC5DB6EC05C312E6166109FA4B52A5CC402CCE74E6EDB1253D913BF4E
```

每个 bank 保存 `states.npz + manifest.json`。Manifest 记录每个 recipe 的 scalar state、
每个谱系数/位移场/color coefficient/filter kernel数组的key、shape、dtype、SHA256，以及
per-recipe和whole-bank canonical hash。

## 4. Primary response 与 cross-fit risk

沿用 K0-A centered class-vs-rest logit response：

\[
m_{i,c}(x)=z_{i,c}(x)-\log\sum_{k\ne c}\exp z_{i,k}(x),
\]

\[
r_{i,q}(u)=P_C\{m_i(A_q(u))-m_i(u)\}.
\]

对每个 client/probe 分别在 `Ua/Ub` 上计算：

\[
\mu_a=\mathbb E_{u\in U_a}r(u),\quad
\mu_b=\mathbb E_{u\in U_b}r(u),
\]

\[
E_a=\mathbb E_{u\in U_a}\|r(u)\|_2^2,\quad
E_b=\mathbb E_{u\in U_b}\|r(u)\|_2^2,
\]

\[
\kappa_{cf}=\frac{[\langle\mu_a,\mu_b\rangle]_+}
{\sqrt{E_aE_b}+\epsilon}.
\]

令 `mu_bar=(mu_a+mu_b)/2`，`mu_(1),mu_(2)` 为最大与第二大类别分量：

\[
\mathrm{sel}=\frac{\mu_{(1)}-\mu_{(2)}}{\|\bar\mu\|_2+\epsilon},\qquad
\rho=\kappa_{cf}[\mathrm{sel}]_+.
\]

每个 client 内独立定义：

\[
e_q=(E_a+E_b)/2,qquad
q\text{ active}\iff e_q\ge\mathrm{median}_{q'}e_{q'}.
\]

Active 只排除近 identity/无响应 probe，不使用任何类别或退化元数据。Client指标：

\[
S_i=\operatorname{mean}_q e_q,
\quad D^{cf}_i=\operatorname{mean}_q[\langle\mu_a,\mu_b\rangle]_+,
\]

\[
K_i=\operatorname{mean}_q\kappa_{cf},
\quad R_i=\operatorname{CVaR}_{top20\%}\{\rho_q:q\ active\}.
\]

`combined` 在128 probes上重新计算全部统计量；不是简单平均 bank A/B 的R。

## 5. Bootstrap 与冻结门槛

HFL和Local分别进行1,000次paired-carrier bootstrap。每次在Ua和Ub内部独立有放回抽样，
但gamma0/gamma0.9使用相同抽样计数；active median和CVaR在每个bootstrap replicate内重算。

Combined bank，HFL必须全部满足：

```text
CI95 lower of Dcf(H9)-Dcf(H0) > 0
K(H9)-K(H0) >= 0.03
CI95 lower of K delta > 0
R(H9) >= 1.20 * R(H0)
CI95 lower of R delta > 0
positive R clients >= 3/4
```

Local 对 L9/L0 使用完全相同门槛。

独立 bank replication：

```text
bank A: R_H9/R_H0 >= 1.10 and R_L9/R_L0 >= 1.10
bank B: R_H9/R_H0 >= 1.10 and R_L9/R_L0 >= 1.10
```

若 `S9-S0` bootstrap CI95下界大于0，但K或R门槛失败，必须标记
`generic_fragility_kill=true`，最终仍为：

```text
NO_GO_GENERIC_DIRECTIONAL_SIGNAL
```

全部门槛通过才是：

```text
GO_TO_K1_CHECKPOINT_SURGERY
```

通过只表示可以另行设计K1；K0-B本身不执行任何surgery或训练。

## 6. Binding isolation 与 optional coverage

生成全部response后先写 `blind_response_manifest.json`。Primary统计、bootstrap、
`result.json`、CSV和response tensors全部写完并由 `primary_artifact_manifest.json` 封存。
这一阶段不导入K0-A operator/family/binding实现。

只有primary封存后，若显式传入 `--oracle-k0a-root`，才允许读取K0-A已保存的16个oracle
mean directions，输出generic-to-oracle cosine coverage。Coverage不选probe、不调threshold、
不进入primary result，也不能改变verdict。OpenI默认不运行该可选分析。

## 7. 实现入口与产物

```text
frozen PRIME: fedprime/augmentations/frozen_prime.py
statistics:  fedprime/engine/cle_generic_probe_gate.py
analyzer:    scripts/analyze_cle_generic_probe_k0b.py
OpenI:       scripts/openi_cle_generic_probe_k0b_entry.py
tests:       tests/test_cle_generic_probe_k0b.py
```

Smoke：

```bash
python scripts/openi_cle_generic_probe_k0b_entry.py --mode=smoke
```

Smoke仍生成并哈希完整64+64 recipe states，但只推理K0-A固定公共列表的前8张、每个bank
前2个recipe、H0/H9 client0。只验证strict checkpoint load、recipe确定性、完整state保存、
response shape/finite、primary isolation、metric/bootstrap接口和打包；必须输出
`SMOKE_ONLY_NO_SCIENTIFIC_DECISION`。

Smoke审计通过后才运行：

```bash
python scripts/openi_cle_generic_probe_k0b_entry.py --mode=formal
```

Formal固定1,000 carriers、64+64 recipes、16 checkpoints、1,000 bootstrap。不得改bank、
seed、active/CVaR定义或门槛。

## 8. 2026-09-02 Formal结果

正式OpenI产物已完成完整性核验，并从16份原始response tensors独立重算。报告与重算均为：

```text
GO_TO_K1_CHECKPOINT_SURGERY
```

HFL与Local各自8/8冻结gates通过，generic-fragility kill未触发。关键结果：

```text
HFL:  K delta=+0.252727, R ratio=4.901569, R positive clients=4/4
Local: K delta=+0.232752, R ratio=4.385780, R positive clients=4/4
Bank A/B R ratios: HFL 5.739226/4.317300, Local 5.166668/4.094945
```

完整独立审计：

```text
deliverables/cle_generic_probe_k0b_20260902/RESULT_SUMMARY_ZH.md
```

该GO只授权另行设计K1 checkpoint surgery。K0-B没有训练纠正模块，也没有证明最终CLE性能收益。
