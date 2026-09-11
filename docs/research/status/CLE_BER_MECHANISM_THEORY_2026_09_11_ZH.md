# BER 作为 CLE 有效分布压缩算子的机制理论

Updated: 2026-09-11

## 1. 论文中的准确定位

BER不是任意困难样本加权。它在每个任务类别内部重新分配伪环境组的总风险质量，等价于在一个
显式有效分布`Q_gamma`上优化。其直接作用对象是CLE的数据来源`P(E|Y)`；DSA则在训练完成后
检查模型行为是否仍沿预先固定binding方向移动概率质量。

```text
CLE: P(E|Y)随类别改变
  -> BER: 压缩有效Q(E_hat|Y)的组支持优势
  -> 理想条件: Y与E_hat独立，纯环境分类优势消失
  -> 现实条件: PEW误差、缺失支持与优化误差留下残余
  -> paired DSA: 最终行为验证
```

## 2. 有效分布等价定理

对客户端`k`、类别`c`、PEW伪环境`e`，strict-fit组数与组风险为：

\[
n_{k,c,e}=\#\{i:y_i=c,\hat e_i=e\},\qquad
R_{k,c,e}=\frac1{n_{k,c,e}}\sum_{i\in(c,e)}\ell_i.
\]

当前hard BER定义：

\[
a^{(\gamma)}_{k,c,e}=
\frac{\mathbf 1[n_{k,c,e}\ge m]\min(n_{k,c,e},K)^\gamma}
{\sum_{e'}\mathbf 1[n_{k,c,e'}\ge m]\min(n_{k,c,e'},K)^\gamma},
\]

\[
R_{BER,k}=\frac1{|\mathcal C_k^{valid}|}
\sum_{c\in\mathcal C_k^{valid}}\sum_e a^{(\gamma)}_{k,c,e}R_{k,c,e}.
\]

因此BER精确等价于在样本质量：

\[
q_i=\frac{a^{(\gamma)}_{k,y_i,\hat e_i}}
{|\mathcal C_k^{valid}|n_{k,y_i,\hat e_i}}
\]

定义的有效经验分布上最小化风险。当前Formal配置为`gamma=0.5, K=32, m=2`。CPU审计中，
该公式聚合出的class-environment joint与训练实现的目标质量最大误差为`7.32e-16`；生产PyTorch
loss另有直接单元测试与NumPy等价公式对照。

## 3. 类内环境优势压缩定理

对两个有效环境组：

\[
\frac{Q_\gamma(e_1\mid c)}{Q_\gamma(e_2\mid c)}=
\left(\frac{\min(n_{c,e_1},K)}{\min(n_{c,e_2},K)}\right)^\gamma.
\]

无cap区域的对数支持优势满足：

\[
\log\frac{Q_\gamma(e_1\mid c)}{Q_\gamma(e_2\mid c)}
=\gamma\log\frac{n_{c,e_1}}{n_{c,e_2}}.
\]

故`0<=gamma<1`将类内环境log-odds压缩为原来的`gamma`倍。当前`gamma=.5`执行平方根压缩；
结合`K=32,m=2`，任何有效伪环境组间的最大理论质量比不超过`sqrt(32/2)=4`。四客户端审计
中的BER后伪环境最大支持比均精确为4，而重加权前为`116.5--205.67`。

## 4. 完全均衡的独立性推论

当`gamma=0`且所有类别共享同一有效环境支持集合时：

\[
Q_0(\hat E=e\mid Y=c)=1/|\mathcal E|=Q_0(\hat E=e),
\]

所以：

\[
Y\perp\hat E,\qquad I_{Q_0}(Y;\hat E)=0.
\]

于是只看伪环境的最优分类器不能超过类别先验多数类准确率。这一结论要求共同支持；BER不能
凭空生成缺失的类别—环境组。当前选择`gamma=.5`不是完全独立化，而是在纠偏强度、少数组方差
和PEW噪声之间折中。

## 5. PEW误差下的条件边界

令BER有效分布下PEW family误差为：

\[
\epsilon_Q=\Pr_Q(\hat E\ne E).
\]

通过同一样本上的`E`与`E_hat`耦合及三角不等式：

\[
TV(Q_{Y,E},Q_YQ_E)
\le TV(Q_{Y,\hat E},Q_YQ_{\hat E})+2\epsilon_Q.
\]

任意只使用真实环境的分类器相对类别先验的Bayes准确率优势，也不超过左侧TV。因此BER对真实
环境去相关的保证由“可观测伪环境依赖”与“目标分布PEW误差”共同控制。

该边界在当前数据上是诚实但无信息的：BER加权后各客户端PEW family误差为`0.507--0.575`，
导致上界截断为`1.0`。这与普通PEW准确率不矛盾：BER主动提高稀有伪组质量，恰好会放大PEW
易错区域。项目此前WEC-BER失败也禁止假设一个稳定可逆的PEW误差通道。因此论文不得声称已
获得非平凡的真实环境理论保证。

## 6. 固定数据CPU Kill Test

输入为CLE-v2 `seed0_split0/gamma09` strict-fit样本、冻结PEW标注和只用于离线审计的真实operator
family。参考分布先做类别均衡但不做环境重加权，以隔离BER的环境作用。冻结门槛与结果：

```text
T0 code/theory identity <= 1e-10:               PASS, max 7.32e-16
T1 pseudo-environment TV decreases on 4/4:      PASS
T2 pooled pseudo TV relative reduction >= 50%:  PASS, 59.03%
T3 true-family TV decreases on 4/4:              PASS, 31.72% pooled
verdict: PASS
```

Pooled equal-client结果：

| Object | Before | BER effective | Relative drop |
|---|---:|---:|---:|
| `TV(Y,E_hat)` | 0.487505 | 0.199744 | 59.03% |
| `TV(Y,E_true-family)` | 0.634914 | 0.433513 | 31.72% |
| true-family environment-only Bayes advantage | 0.313840 | 0.240863 | 23.25% |

client2的伪环境TV下降`73.35%`、真实family TV下降`38.66%`，所以它在既有Formal中的任务效用
下降不能解释为“BER没有压缩该客户端的CLE分布”。更可能的剩余问题是PEW噪声、类别选择性
梯度重分配或模型—数据对固定混杂；该CPU审计不能判定具体原因。

## 7. 能证明与不能证明什么

允许：当前BER实现精确对应有效分布公式；它在固定strict-fit数据上大幅压缩伪环境依赖，并且
真实family依赖在四客户端同方向下降；BER具有直接针对CLE数据来源的机制解释。

不允许：由分布审计直接推出训练后DSA必然下降或准确率必然提高；声称PEW误差边界非平凡；
声称gamma=.5已经实现`Y independent of E`；用真实operator metadata进入实际训练。

训练后的shortcut缓解仍由已有AsymHFL与FedDF Formal DSA结果支撑。该审计补的是“为什么BER
有理由起作用”，不是第三份训练效果证据。

## 8. 证据位置

```text
fedprime/engine/ber_theory.py
scripts/audit_ber_mechanism.py
tests/test_ber_theory.py
deliverables/ber_mechanism_theory_20260911/BER_MECHANISM_AUDIT.json
deliverables/ber_mechanism_theory_20260911/BER_MECHANISM_AUDIT_ZH.md
```
