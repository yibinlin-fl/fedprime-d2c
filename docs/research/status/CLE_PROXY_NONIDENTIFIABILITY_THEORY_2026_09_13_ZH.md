# CLE代理环境不可识别性与DSA必要性

更新：2026-09-13

## 1. 本文补上的逻辑缺口

BER直接平衡的是PEW给出的伪环境`Z=E_hat`，CLE真正关心的是潜在真实corruption环境`E`。
因此必须回答：仅凭`P(Y,Z)`的改善，能否保证`P(Y,E)`中的类别—环境依赖同步下降？答案是：
没有额外识别条件时不能。本文件给出构造性不可识别定理、已有条件传递界、两个数值反例及
本文评价协议中的DSA必要性推论。

该结论不是把BER写成无效方法，而是区分三个对象：

```text
BER可直接控制：       proxy support dependence P(Y,Z)
附加假设下可传递：    true environment dependence P(Y,E)
最终必须直接评价：     binding-directed model behavior (DSA)
```

## 2. 依赖度量

对任意离散变量`A,B`，定义偏离独立性的总变差：

\[
\mathcal D(A,B)
=TV(P_{A,B},P_AP_B)
=\frac12\sum_{a,b}|P(a,b)-P(a)P(b)|.
\]

`D(A,B)=0`当且仅当二者独立。BER有效分布理论直接压缩的是
`D_Q(Y,Z)`，其中`Q`是BER诱导的有效训练分布。

## 3. 定理：Proxy-only non-identifiability

**定理1（代理环境不足以识别真实环境依赖）。** 给定任意可观测联合分布`P(Y,Z)`，若`Y`
至少有两个正概率取值且没有对未知通道`P(Z|E,Y)`施加约束，则存在两个完整数据生成分布
`P_A(Y,Z,E)`和`P_B(Y,Z,E)`，满足：

\[
P_A(Y,Z)=P_B(Y,Z)=P(Y,Z),
\]

但：

\[
\mathcal D_A(Y,E)=0,
\qquad
\mathcal D_B(Y,E)=1-\sum_eP_B(E=e)^2>0.
\]

**构造性证明。**

1. 先固定非恒定映射`g`，并令`nu(e)=P(g(Y)=e)`。在世界A中取`E~nu`且独立于`(Y,Z)`；
   当`Z`与`Y`独立且边缘恰为`nu`时也可直接令`E=Z`。于是两个世界甚至具有相同的`P_E`，
   且`P_A(Y,E)=P_A(Y)P_A(E)`，故`D_A(Y,E)=0`。
2. 在世界B中，选择一个非恒定映射`g`并令`E=g(Y)`；同时保持条件分布`P(Z|Y)`与给定
   可观测分布完全相同。因此边缘`P_B(Y,Z)`没有任何变化。
3. 因为世界B中的`E`由`Y`确定，逐项计算TV得到：

\[
\begin{aligned}
\mathcal D_B(Y,E)
&=\frac12\sum_y\left[
P(y)(1-P(g(y)))+P(y)\sum_{e\ne g(y)}P(e)
\right]\\
&=\sum_yP(y)(1-P(g(y)))\\
&=1-\sum_eP(E=e)^2.
\end{aligned}
\]

两世界具有完全相同的所有`(Y,Z)`可观测量，却具有不同的真实环境依赖，故仅凭`P(Y,Z)`不可
识别`D(Y,E)`。证毕。

**推论1（不存在紧的无条件proxy-only证书）。** 任何只以`P(Y,Z)`为输入、同时对所有未约束
误差通道成立的上界，在给定可观测分布上都至少要覆盖
`max_g[1-sum_e P(g(Y)=e)^2]`这一潜在最坏值。对本文10类、5个均衡环境的构造，该值为`0.8`，
故proxy-only观测不可能给出“真实依赖很小”的证书。任何更紧保证至少需要以下之一：

- 已知且足够小的目标分布误差；
- 稳定、可迁移且条件良好的误差通道；
- 独立的真实环境审计信息；
- 直接面向目标shortcut行为的评价量。

## 4. 与PEW条件传递界统一

现有BER理论已证明，在同一样本耦合下：

\[
\mathcal D_Q(Y,E)
\le
\mathcal D_Q(Y,Z)+2\epsilon_Q,
\qquad
\epsilon_Q=P_Q(Z\ne E).
\]

它不是错误的界，而是定理1所要求的“附加识别条件”的一种形式。只有当：

\[
\mathcal D_Q(Y,Z)+2\epsilon_Q<1
\]

时，它才能给出非平凡真实环境证书。当前BER有效分布上的PEW family误差为`0.507--0.575`，
导致上界截断为`1.0`。因此论文必须保留条件性表述。

WEC-BER Phase-0进一步验证了不能简单换用混淆矩阵反演：尽管公共矩阵最小奇异值为
`0.258820`、条件数为`3.813622`，operator cross-fit column-L1 median/p75达到
`0.309333/0.526667`，超过冻结门槛，且severity依赖很强。数值可逆不等于误差通道可迁移；
不得通过调门槛、删除operator或换名复活该路线。

## 5. 与CVRS反例统一

CVRS M0 Formal在MobileNetV2上给出真实经验反例：

\[
R_{proxy}: 0.244061\;(Public\text{-}JSD)
\rightarrow0.155208\;(CVRS),
\]

代理下降`0.088854`；但：

\[
DSA:0.116098\rightarrow0.129098,
\]

真实CLE directional harm反而上升`0.013000`。这不是定理1的证明前提，而是其在项目真实模型
中的经验对应：generic public-response proxy的下降不能替代目标对齐的shortcut评价。

Stage-1冻结概率缓存还给出另一反例：令同一operator的三个增强视图预测完全相同，则
within-operator JSD严格为`0`，而同一缓存的cross-operator DSA仍为`0.119644`。因此：

\[
\text{within-operator consistency}
\not\Rightarrow
\text{cross-operator directional invariance}.
\]

## 6. 推论：DSA为什么在本文协议中是必要终点

**推论2（目标对齐验证必要性）。** 若训练方法只优化一个不读取真实binding的proxy目标`R_Z`，
则在没有额外识别条件时，`R_Z`下降不能推出CLE directional harm下降。要提出“方法缓解了
CLE routing”这一行为主张，必须额外报告一个读取冻结evaluation binding并比较同源跨operator
响应的目标对齐estimand。本文选择paired DSA。

这里的“必要”限定于证据类型，不声称DSA是所有shortcut问题唯一可能的指标。其他指标若能
直接读取同一目标行为也可能成立；但PEW accuracy、Public-JSD、CVRS routing proxy和普通鲁棒
accuracy均不能在本协议中替代DSA。

DSA不进入训练、路由、模型选择或调参。它只在结果封存后读取预先固定的binding，并用同一
source跨operator的概率差消除operator-invariant语义基线。因此形成以下完整逻辑：

```text
CLE support imbalance
  -> BER theorem: compress P_Q(Y,Z)
  -> conditional bridge: transfer requires controlled PEW error
  -> impossibility theorem: no such condition means no proxy-only guarantee
  -> paired DSA: directly test the frozen target behavior
  -> Formal evidence: mitigation replicated across maps/bases and against strong controls
```

## 7. 缓存数值验证

零训练、零推理、CPU脚本：

```text
scripts/validate_proxy_nonidentifiability.py
```

冻结输入：

```text
outputs/openi_downloads/cle_v2_mechanism_stage1_seed0_benchmark/
  extracted_formal/analysis/STAGE1_PREDICTIONS.npz
outputs/openi_downloads/cle_cvrs_m0_seed0_formal/extracted/outputs/
  cle_cvrs_m0_seed0_formal/result.json
```

结果：

```text
same observed P(Y,Z):                         true
observable D(Y,Z):                            0.000000
latent world A D(Y,E):                        0.000000
latent world B D(Y,E):                        0.800000
identical-view JSD / cached DSA:               0 / 0.119644
CVRS-vs-JSD proxy change / DSA change:        -0.088854 / +0.013000
N0/N1/N2/N3/N4:                               all PASS
```

在1000个均衡CIFAR source上，构造五个均衡proxy环境；世界A令真实环境等于该均衡proxy，世界B
令真实环境为`Y mod 5`。两世界的`P(Y,Z)`逐样本完全相同，但真实依赖TV分别为0和0.8。该构造
验证的是不可识别性，不声称模拟自然corruption生成过程。

产物：

```text
deliverables/cle_proxy_nonidentifiability_20260913/
  PROXY_NONIDENTIFIABILITY_VALIDATION.json
  PROXY_NONIDENTIFIABILITY_THEORY.png
  PROXY_NONIDENTIFIABILITY_THEORY.pdf
  RESULT_SUMMARY_ZH.md
```

## 8. 完成后允许与禁止的论文表述

允许：BER对observable pseudo-environment support具有精确分布解释；真实环境传递需要额外识别
条件；proxy-only无条件保证在一般情形下不可识别；因此paired DSA是本文CLE缓解主张所需的
target-aligned评价终点。已有Formal显示BER实际降低了DSA。

禁止：BER无条件保证真实环境与类别独立；DSA是全世界唯一shortcut指标；不可识别定理本身证明
BER有效；用构造性反例替代多seed、第二数据集或真实场景实验。

## 9. 理论闭环状态

本文件完成的是“逻辑闭环”：明确BER直接保证什么、传递到真实环境需要什么、缺少条件时为何
原则上不可能、以及为什么必须用DSA验真。它没有也不可能在当前观测条件下变成无条件性能定理。
方法有效性仍由Formal实验支撑，外部有效性仍需training-seed稳定性和第二数据集。
