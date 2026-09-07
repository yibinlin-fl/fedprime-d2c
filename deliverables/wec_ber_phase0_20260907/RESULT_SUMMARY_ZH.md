# WEC-BER Phase-0 结果摘要

日期：2026-09-07

## 最终判定

```text
NO_GO_WEC_BER_ERROR_TRANSFER
TRAINING_NOT_STARTED
```

Primary channel 是 `6 x 5` 的 observed-by-latent 矩阵：保留 PEW `unknown` 作为可观测输出，
但不把 composite unknown 提升为第五个基础 corruption family。每个 operator cross-fit 只比较
该 operator 所属真实 family 的通道列，避免虚构无法观测的完整 held-out 矩阵。

## Public Q

| observed \ true | clean | noise | blur | weather | digital |
|---|---:|---:|---:|---:|---:|
| clean | 0.610778 | 0.011976 | 0.179641 | 0.119760 | 0.271084 |
| noise | 0.053892 | 0.808383 | 0.000000 | 0.071856 | 0.012048 |
| blur | 0.119760 | 0.000000 | 0.682635 | 0.035928 | 0.277108 |
| weather | 0.173653 | 0.017964 | 0.065868 | 0.592814 | 0.078313 |
| digital | 0.029940 | 0.000000 | 0.059880 | 0.017964 | 0.325301 |
| unknown | 0.011976 | 0.161677 | 0.011976 | 0.161677 | 0.036145 |

奇异值：`0.987042, 0.824265, 0.593659, 0.444066, 0.258820`。

## 四层 Gate

- G1 `PASS`：rank `5`，最小奇异值
  `0.258820`，condition number
  `3.813622`。
- G2 `FAIL`：operator column L1 median/p75 为
  `0.309333/0.526667`，超过冻结上限
  `0.25/0.35`；dominant confusion preservation 为
  `68.75%`，单项超过 60%。
- G3 `PASS`：相对 hard pseudo prevalence 的 median recovery improvement
  `66.00%`，改善 operator 比例
  `81.25%`。
- G4 support/risk：`NOT_OPENED`，因为 G2 失败后 private oracle 必须保持 sealed。
- Public-only 冻结 lambda：`0.0`。

G3 通过不能覆盖 G2：这说明某些 fold 上矩阵反解能把纯 family prevalence 拉回正确方向，
但 error channel 本身跨具体 operator 不稳定，因此没有资格迁移到 private CLE cells。

G2 最不稳定的 operator：

- `jpeg_compression` (digital): column L1 `0.797333`
- `spatter` (weather): column L1 `0.730667`
- `contrast` (digital): column L1 `0.602667`
- `snow` (weather): column L1 `0.546667`
- `brightness` (digital): column L1 `0.520000`

## Private sealed 状态

由于 G2 已失败，脚本按 sealed protocol 没有打开 private oracle。

exact PEW+BER 12-round archive 还设置了 `save_final=false`，没有分类模型 checkpoint。即使 G2
通过，private risk recovery 也不能由现有资产严格构造，必须另行获得 fixed exact-method
checkpoint 后才能评估，不能借用含 CDep 或其他方法的 checkpoint。

## Severity secondary diagnostic

- `SEVERITY_DEPENDENCE_HIGH`; max Frobenius `0.649774`;
  最大 diagonal-recall range `57.50 pp`。

severity dependence 很高，与 G2 的 operator instability 方向一致；它只作诊断，不改变 primary
NO-GO，也不授权 severity-aware WEC-BER。

## 科学解释

公共混合 confusion matrix 本身满秩且条件良好，但这只是 pooled 可逆性。按具体 operator
拆开后，PEW 的错误规律变化过大，尤其集中在 digital/weather family。因而一个从 pooled
public validation 学到的单一 Q 不能被当成稳定的 private noise channel。这正是 Phase-0 要杀掉
的假设：`pooled invertible != operator-transferable`。

没有训练模型，没有更新参数，没有使用 GPU 或 DSA evaluator，没有打开 private oracle，也没有
启动 12-round HFL。不得通过删 operator、按 family 选择性保留、调 gate、调 threshold、改 solver
或补 seed 复活 WEC-BER。
