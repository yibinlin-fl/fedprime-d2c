# CRSR 类条件预测残差谱风险 Audit 0

日期：2026-08-11

## 1. 研究问题

当前 hard PEW 将公共损坏监督为 `clean/noise/blur/weather/digital/unknown`，再由 hard BER
平衡本地 `class x predicted-environment` 风险。A0--A6 消融已经确认 BER 是主要正增益来源，
但五类损坏 taxonomy 是论文的首要方法风险。

本审计只回答：不预测环境、不学习环境 embedding、不排序高损失尾部时，同一类别预测残差
的主谱方向能否稳定对应潜在弱 `class x operator` 单元，并在 matched 一步更新中产生正归因。

本阶段不接正式 runner，不启动 12/40 轮，不修改 AsymHFL。

## 2. 冻结数学对象

模型概率为 `p(x)`，真实类别 `c` 的残差为：

```text
r(x,c) = p(x) - one_hot(c)
mu_c   = E[r | y=c]
Sigma_c = E[(r-mu_c)(r-mu_c)^T | y=c]
S_c    = sqrt(lambda_max(Sigma_c))
```

候选分类目标：

```text
L_CRSR = mean_c [ mean(CE | y=c) + 2.0 * S_c ]
```

完整隔离一步目标仍保留 `12 * JSD + DCL`。`spectral_weight=2.0` 对应理论质量下界
`rho=1/(1+2^2)=0.2`；Audit 0 不搜索该权重。

对类别内任意未知子群 `G`，若 `pi=P(G|y=c)`，协方差分解给出：

```text
||mu_c,G - mu_c|| <= sqrt((1-pi)/pi * lambda_max(Sigma_c))
```

结合错误分类时 `p_c<=1/2` 及 `1-p_c<=-log(p_c)`：

```text
Err(c,G) <= 2 * [ CE_c + sqrt((1-pi)/pi) * S_c ]
```

该界只覆盖训练支持内具有正质量的潜在子群，不声称覆盖零质量的全新 operator。

## 3. 严格数据边界

```text
scenario: CLE-HFL v2 alpha=0.5 gamma=0.9 seed0_split0
clients/models: client 1 / ResNet12, client 3 / MobileNetV2
base training: 3 epochs AugMix/JSD/DCL, no PEW, no BER
probe: only from persisted fit indices, class-stratified and disjoint from base train
signal fitting: two disjoint probe halves, evaluated in both directions
private audit: forbidden
final test: forbidden
operator ID: post-hoc cell evaluation only
```

## 4. 对照信号

CRSR 的 held-out spectral energy 必须同时与以下信号比较：

```text
CE
Brier residual norm
fixed random simplex-tangent direction
```

若 CRSR 与 CE/Brier 几乎完全相关，或不能优于随机方向和普通误差信号，则视为谱包装而非新机制。

## 5. 冻结 GO/NO-GO 门槛

```text
G0 validity:
   每个 directed probe 的 base accuracy >= 20%
   每个 directed probe 至少 6 个有效类别

G1 spectral activity:
   source lambda_max / trace 的中位数 >= 0.18

G2 cross-split stability:
   主方向绝对 cosine 中位数 >= 0.35
   held-out variance share 相对固定随机方向优势中位数 >= 0.03

G3 non-redundancy:
   类别平衡的 |Spearman(spectral, CE)| 中位数 <= 0.95
   类别平衡的 |Spearman(spectral, Brier)| 中位数 <= 0.95

G4 cell relevance:
   合计至少 20 个有效 class x operator cells
   spectral cell correlation 中位数 >= 0.25
   相对 CE/Brier/random 中较强者的优势中位数 >= 0.02

G5 matched one-step mean non-inferiority:
   每个客户端 candidate-control mean probe CE <= +0.0001

G6 matched one-step weak-cell attribution:
   每个客户端 worst-cell CE 严格下降至少 0.00001
   每个客户端 mean class-wise cell-gap CE 不增加
```

`G0` 失败为 `INVALID_PROBE`；其他任一门槛失败均为 `NO-GO`。失败后冻结 CRSR，不得通过调整
谱权重、probe 大小、支持阈值、幂迭代次数或门槛复活。

## 6. 入口

```text
python scripts/audit_class_residual_spectral_risk.py --smoke
python scripts/audit_class_residual_spectral_risk.py
```

输出：

```text
outputs/class_residual_spectral_risk_audit0/result.json
outputs/class_residual_spectral_risk_audit0/signals.npz
outputs/class_residual_spectral_risk_audit0/RESULT_SUMMARY_ZH.md
```

## 7. 2026-08-11 正式结果

正式 Audit 0 在冻结配置上完整结束，stderr 为空。结果文件：

```text
outputs/class_residual_spectral_risk_audit0/result.json          14,886 bytes
outputs/class_residual_spectral_risk_audit0/signals.npz          83,296 bytes
outputs/class_residual_spectral_risk_audit0/RESULT_SUMMARY_ZH.md  1,142 bytes
```

独立从 `result.json` 重算得到：

| Gate | 结果 | 关键数值 |
|---|---:|---|
| G0 validity | PASS | 所有 directed probe accuracy 均超过 20%，有效类别数为 8/9 |
| G1 spectral activity | PASS | top eigenvalue share 中位数 `0.752411` |
| G2 cross-split stability | PASS | direction cosine `0.975226`；transfer advantage `0.639379` |
| G3 non-redundancy | PASS | 与 CE/Brier 的绝对相关中位数 `0.107906/0.120375` |
| G4 cell relevance | **FAIL** | spectral cell correlation `0.069658 < 0.25`；相对最强基线优势 `-0.903889 < 0.02` |
| G5 mean non-inferiority | **FAIL** | client 1/3 mean CE delta `+0.006485/+0.000360`，均超过 `+0.0001` |
| G6 weak-cell attribution | **FAIL** | client 1 worst-cell CE `+0.101343`；client 3 为 `-0.045068`，未跨客户端成立 |

由于本机 LAPACK 不可用，正式脚本使用确定性 power iteration。归档前又对 `signals.npz`
中的全部类条件协方差进行了 10 个坐标起点、128 次迭代的独立复核；原方向与复核方向的
最小绝对 cosine 为 `0.9999999747`，最大相对特征值误差为 `6.21e-9`。因此 G1/G2/G4
失败不是单一起点的数值伪影。

双架构的谱方向都高度稳定，但它对弱 `class x operator` cell 的解释不稳定：ResNet12
两个方向的 cell correlation 为 `-0.4524/-0.1317`，MobileNetV2 为
`+0.3830/+0.2711`。这说明 CRSR 找到了可复现的预测残差几何结构，却没有找到可跨架构
替代 BER 环境分组的风险对象。普通 CE/Brier 对 cell error 的相关性明显更强。

正式判定：`NO-GO`。冻结 CRSR，不调 `spectral_weight`、支持阈值、probe 大小、power
iteration 次数或门槛，不实现正式训练路径，不连接 runner，不启动 12/40 轮。该结果不否定
“taxonomy-free 本地模块”这一研究目标，只否定“类内残差最大特征值是 BER 替代风险”这一具体对象。
