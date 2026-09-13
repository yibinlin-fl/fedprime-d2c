# CLE-HFL 中 DSA 的识别对象、成立假设与统计推断
Updated: 2026-09-13

## 1. 结论先行

DSA 不是普通鲁棒准确率，也不是声称能从任意观测数据恢复“模型内部因果机制”的万能统计量。
它识别的是一个更窄但可检验的对象：在保持 source 语义不变时，更换 corruption operator 后，
模型分配给该 operator 在训练中绑定类别的概率质量，是否相对其他 operator 定向增加。

这一识别成立依赖四个最弱条件：

1. 同一 source 的 paired operator 变换不改变任务标签；
2. 类别—operator binding 在查看评价预测前预先固定；
3. 每个 source 的多个 operator 视图被当作一个配对簇，而不是独立样本；
4. 最终评价标签、真实 operator 与 binding 不进入训练、路由、选择或调参。

因此，本文可以把贡献写成“受控 CLE-HFL 问题 + paired directional diagnostic + matched
mechanism attribution + 可行动缓解”，但不能把 DSA 写成无条件的因果充分统计量。

## 2. 数学对象

令客户端为 `k`，source 为 `z=(x,y)`，operator 为 `o`，语义保持变换为 `T_o(x)`。训练数据
预先给定客户端特定 binding `b_k(c)`。对 operator `o`，其绑定类别集合为：

\[
B_{k,o}=\{c:b_k(c)=o\}.
\]

为避免“真实类别本来就属于该集合”产生平凡质量，评价只保留 `y` 不属于 `B_{k,o}` 的 source。
定义绑定质量：

\[
m_{k,o}(z,o')=\sum_{c\in B_{k,o}}p_k(c\mid T_{o'}(x)).
\]

source-level directional contrast 为：

\[
d_{k,o}(z)=m_{k,o}(z,o)-
\frac{1}{|\mathcal O|-1}\sum_{o'\ne o}m_{k,o}(z,o').
\]

`DSA_{k,o}=E_z[d_{k,o}(z)]`，pooled DSA 再对有效 operator 与客户端等权平均。DSA 位于概率
尺度；例如 `0.12` 表示平均约 12 个百分点的绑定方向概率质量差，而不是 0.12 个百分点准确率。

## 3. 识别定理：paired contrast 消除 operator-invariant 语义项

假设绑定质量响应可写为：

\[
m_{k,o}(z,o')=s_{k,o}(z)+r_{k,o}(z,o'),
\]

其中 `s` 是同一 source 上不随 operator 改变的语义/类别基线，`r` 是 operator-dependent response。
代入 paired contrast 后：

\[
d_{k,o}(z)=r_{k,o}(z,o)-
\frac{1}{|\mathcal O|-1}\sum_{o'\ne o}r_{k,o}(z,o').
\]

因此 `s` 精确相消。DSA 识别的是绑定方向的 operator response contrast。若所有 operator 条件
交换，则 `r(z,o)=r(z,o')`，故 DSA 为零。反之，正 DSA 表示响应沿预先固定 binding 定向移动；
配合 shuffled-binding null，能够排除“任意 corruption 普遍推高某些类别”的主要替代解释。

该结论不自动证明：模型只使用了 corruption、该依赖在现实分布中一定有害，或不存在未控制的
operator-specific语义泄漏。这些均由语义保持设计、gamma0对照、shuffled null和数据角色隔离
共同支撑，而不是由一个公式单独保证。

## 4. DSA 的代数性质

### 4.1 零基准

若同一 source 的预测对 operator 条件交换，则 DSA 严格为零。固定缓存的 exchangeable
projection 为 `-2.50e-20`；gamma0 的经验最大绝对 DSA 为 `0.001914`。

### 4.2 对预测概率的仿射性

对 `p_lambda=(1-lambda)p_0+lambda p_1`：

\[
DSA(p_\lambda)=(1-\lambda)DSA(p_0)+\lambda DSA(p_1).
\]

当 `p_1` 相对 `p_0` 增加绑定方向响应时，DSA 随混合强度单调增加。缓存 11 点曲线的最大数值
误差为 `2.78e-17`。

### 4.3 JSD 不充分反例

同一 operator 周围的增强预测可完全一致，使 AugMix-JSD 为零；不同 operator 的预测仍可分别
向其绑定类别移动，使 DSA 为正。因此 within-operator augmentation consistency 不推出
cross-operator directional invariance。固定缓存反例为 `JSD=0`、`DSA=0.119644`。

## 5. 受控注入验证与 binding specificity

在合法概率单纯形中构造 binding-aligned response，并以强度 `lambda` 注入。11 个强度点上，
DSA 对已知目标强度的最大恢复误差为 `6.66e-16`。再加入同一 source 上对所有 operator 相同的
概率位移，DSA 改变量仅 `8.33e-17`，数值验证了 paired cancellation。

同一构造下，真实 binding 的 DSA 为 `0.241071`，shuffled-binding null p95 为 `0.026786`，
置换 `p=0.000999`。这验证实现能区分“binding-aligned”与“同样大小但方向被打乱”的响应。

## 6. 估计量与不确定性

统计独立单位是 source，不是单张 corrupted image。每个 source 的 operator grid 先压缩成一个
source-level contrast，再在 source 间重采样。若 source 独立且 `d(z)` 位于 `[-1,1]`，Hoeffding
给出双侧半径：

\[
\epsilon(n,\delta)=\sqrt{\frac{2\log(2/\delta)}{n}}.
\]

在 `n=1000, delta=0.05` 时，保守半径为 `0.085894`。论文主结果使用 source-clustered
bootstrap；该区间只覆盖评价 source 抽样不确定性，不覆盖训练 seed、binding map、数据划分或
模型选择不确定性。把 15 个 operator 图像当成 15 个独立观测会产生伪重复并低估方差。

## 7. local-first 归因的准确含义

定义 matched contrasts：

\[
\Delta_{HFL}=DSA(HFL,\gamma=.9)-DSA(HFL,\gamma=0),
\]

\[
\Delta_{Local}=DSA(Local,\gamma=.9)-DSA(Local,\gamma=0).
\]

固定场景中二者分别为 `0.119889` 与 `0.107960`，描述性比值为 `90.05%`，通信附加 contrast
为 `0.011929`。这支持 local-first，但不是逐样本中介分析，也不意味着联邦设置多余。HFL仍然
决定多客户端、非IID、模型异构与公共知识交换的研究边界。

## 8. 当前可写主张与待补证据

可写：DSA 在明确语义保持与角色隔离假设下识别 binding-specific directional response；它有
零基准、仿射性、paired cancellation、binding specificity及正确的source-level推断单位。

不可写：DSA 是所有 spurious correlation 的充分统计量；一次固定 mapping 能证明跨场景泛化；
DSA 降低必然带来准确率提升；PEW+BER是taxonomy-free或通用无损插件。

## 9. 2026-09-13：代理不可识别性为何使DSA成为必要终点

最新构造性定理证明：不约束代理误差通道时，相同`P(Y,E_hat)`可对应真实环境独立或高度绑定
的潜在世界。因此PEW accuracy、Public-JSD、CVRS routing proxy或任何只读取代理环境的统计量
下降，都不能单独证明真实CLE directional harm下降。CVRS MobileNetV2 Formal进一步给出
`proxy -0.088854`但`DSA +0.013000`的经验反例。

所以本文所称“DSA必要”是证据协议意义上的：提出CLE缓解主张时，必须额外报告一个直接读取
冻结binding和同源跨operator响应的target-aligned estimand；本文采用paired DSA，并不声称它是
所有shortcut问题唯一可能的指标。完整证明与缓存复算见：

```text
docs/research/status/CLE_PROXY_NONIDENTIFIABILITY_THEORY_2026_09_13_ZH.md
deliverables/cle_proxy_nonidentifiability_20260913/RESULT_SUMMARY_ZH.md
```

Cross-binding map1 Formal与held-out map2 40轮四臂Formal均已完成。前者DSA降低56.46%；后者
PEW+BER DSA为`0.077741`，低于CVaR的`0.088982`和matched PEW+GroupDRO的`0.214575`，全部
冻结门槛通过。它们增强外部复现与机制对照，但不覆盖新partition、training seed或第二数据集。
