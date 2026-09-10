# DSA 理论命题与缓存验证协议

Updated: 2026-09-11

状态：协议已冻结并完成验证；全部预注册门槛通过。只读取了 Stage-1 正式预测缓存，未训练、
未重新推理，也未调整门槛。

## 2026-09-11 验证结果

```text
命题1：exchangeable projection DSA = -2.50e-20；gamma0经验最大绝对DSA = 0.001914
命题2：HFL/Local 11点混合曲线严格单调；最大线性误差 = 2.78e-17
命题3：三个相同预测视图的最大JSD = 0；缓存h9 HFL DSA = 0.119644
verdict: ALL_FROZEN_GATES_PASS
```

这说明DSA的零点、概率混合线性和“JSD为零不推出DSA为零”均在固定缓存上得到数值验证；
它不是跨数据集、跨场景或因果充分性的证明。

## 数学对象

operator-level DSA 对预测概率是线性泛函。对客户端 `k`、operator `o`及其绑定类别集合
`B_{k,o}`，排除真实标签属于该集合的source后：

```text
DSA_k,o(p) = E_x [ sum_{c in B_k,o} p(c|T_o(x))
                 - mean_{o' != o} sum_{c in B_k,o} p(c|T_o'(x)) ]
```

pooled DSA 再对有效operator与客户端取均值。

## 三个命题与冻结验证

1. **零基准**：若同一source的预测对operator条件交换，即所有`o`下概率相同，则DSA严格为0。
   构造exchangeable projection并要求`abs(DSA)<=1e-12`；固定gamma0的HFL/Local经验DSA均要求
   `abs(DSA)<=0.01`。
2. **混合线性与单调性**：对`p_lambda=(1-lambda)p_0+lambda p_1`，要求11个lambda点上的DSA
   与端点线性插值最大误差`<=1e-12`；Stage-1 HFL和Local的gamma0到gamma0.9曲线均严格递增。
3. **JSD不充分性**：复制同一operator预测为三个完全相同增强视图，要求最大JSD`<=1e-12`，
   同时使用缓存h9 HFL预测要求operator-level DSA`>=0.10`。

所有门槛均在运行前冻结。数值PASS只验证代数实现、固定缓存经验零点及一个反例，不把它外推为
跨场景定理；理论成立仍依赖同一source跨operator保持任务语义、binding预先固定且评价标签不进入训练。

## 输入与输出

```text
input:
outputs/openi_downloads/cle_v2_mechanism_stage1_seed0_benchmark/extracted_formal/analysis/STAGE1_PREDICTIONS.npz

entry:
scripts/validate_cle_dsa_theory.py

output:
deliverables/cle_dsa_theory_validation_20260911/
```
