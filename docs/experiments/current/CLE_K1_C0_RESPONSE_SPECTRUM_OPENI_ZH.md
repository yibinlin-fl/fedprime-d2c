# CLE K1-C0 Response Spectrum Gate（OpenI）

## 当前问题

K1-B0 的正式结论保持为 `NO_GO_SHARED_NUISANCE_ROUTING`，不得降低其
1.20 specificity 门槛或重新挑 probe。K1-C0 只检验由该实验暴露出的另一条
机制假设：强 CLE 是否会让大量通用 nuisance intervention 的跨 carrier 表征
响应复用少数共享方向，从而形成更集中的响应谱。

K1-C0 是零训练机制门控。它不训练或微调模型，不修改 RAHFL、AsymHFL、
backbone、classifier 或 BN，也不实现 CRSF。

## 冻结输入

- Phase-B0 输入包：`cle_public_canonicalization_phase_b0_seed0_inputs.tar.gz`
- 文件大小：`535256689` bytes
- SHA256：`DFB766F6494A5F61AA16F45666EC250A30501066AB54D89C984CD2324293B9BC`
- checkpoint：H0/H9/L0/L9，各 4 个异构客户端，round 40
- 公共数据：冻结的 D_rep 2000 张 CIFAR-100 train image，只读取图像
- D_rep index SHA256：`321C0910E8AA376B10D04D1319F24917EE91EABD25BCC8C31A0BDE66F8E240EE`
- U1/U2：D_rep 的前/后 1000 张，互斥
- PRIME Bank A/B：各 64 个固定 recipe，全部等权使用

虽然旧输入包还带有 `evaluation/`，K1-C0 的 OpenI 入口在解包时明确跳过该目录，
只校验和保留 `public/` 与 `checkpoints/`；分析器因此无法读取标签、CLE binding、
真实 corruption taxonomy、DSA、WCCA 或 CFG。

## 数学对象

对公共图像 `u` 和固定 PRIME probe `q`，在分类头前取表征 `h`：

```text
delta_h_q(u) = h(A_q(u)) - h(u)
mu_q         = mean_u delta_h_q(u)
E_q          = mean_u ||delta_h_q(u)||^2
r_q          = mu_q / (sqrt(E_q) + 1e-12)
S            = [r_1, ..., r_64]
K            = S^T S
chi          = ||K||_F^2 / (trace(K)^2 + 1e-12)
```

U1、U2 分开计算 `chi` 后再平均。`chi` 越高表示响应越集中、有效响应秩越低。
同时在同一公共图像上计算居中的普通干净特征协方差谱 `chi_clean`，用于排除
一般性的 feature collapse。

系统级量均采用 ratio of means：

```text
R_resp = mean(chi_response_gamma9) / mean(chi_response_gamma0)
R_clean = mean(chi_clean_gamma9) / mean(chi_clean_gamma0)
D_spec = log(R_resp) - log(R_clean)
```

## 正式门控

HFL（H9 对 H0）和 Local（L9 对 L0）分别通过五项门控：

1. Bank A、B 的 mean chi 均为 gamma9 > gamma0；
2. combined `R_resp >= 1.25`；
3. 至少 3/4 客户端的两 bank 平均 chi 增加；
4. carrier bootstrap 的 Delta-chi CI95 lower > 0；
5. `D_spec > 0` 且其 bootstrap CI95 lower > 0。

bootstrap 固定为 2000 次、seed `20260912`，只重采样 public carrier；U1/U2
独立重采样，同一组索引在全部 arm、bank、client 间复用。每次都从重采样后的
carrier 重新计算 mu、E、S、K、chi 及 clean covariance chi。

十项全部通过才是：

```text
GO_TO_K1_C_CRSF_SURGERY
```

任一失败即：

```text
NO_GO_RESPONSE_SPECTRAL_MECHANISM
```

## OpenI 运行

入口：

```bash
python scripts/openi_cle_k1_c0_response_spectrum_entry.py --mode=formal
```

建议先在本地完成 smoke；OpenI 可直接运行 formal。若希望在 OpenI 再做一次
执行检查，可将 `formal` 改为 `smoke`，但 smoke 数值不允许用于科研判断。

正式结果包：

```text
cle_k1_c0_response_spectrum_seed0_formal_outputs.tar.gz
```

结果包不复制 checkpoint，只保存 Gram、eigenvalue、bootstrap、门控、hash 与
审计文件，因此应保持紧凑。下载后放到：

```text
outputs/openi_downloads/cle_k1_c0_response_spectrum_seed0/
```

## 停止规则

Formal 完成并独立重算 saved artifacts 后立即停止。即使 GO，也不能自动进入
CRSF surgery 或 40-round 训练；下一步必须先冻结可微估计器、mini-batch 偏差、
independent-bank 设计、优化预算、语义保持和对照组。
