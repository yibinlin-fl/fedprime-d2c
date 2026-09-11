# DSA 识别理论五项命题与缓存验证

本验证只读取冻结的 Stage-1 概率缓存，不训练、不推理、不使用 GPU。

## 结论

- 命题1：exchangeable projection DSA = `-2.502e-20`；gamma0最大绝对DSA = `0.001914`。
- 命题2：11点混合曲线最大线性误差 = `2.776e-17`；HFL/Local均严格单调 = `True`。
- 命题3：三个完全一致视图的最大JSD = `0.000e+00`，同时跨operator DSA = `0.119644`。
- 命题4：已知shortcut强度恢复最大误差 = `6.661e-16`；operator-invariant偏移不变性误差 = `8.327e-17`；shuffled-binding p = `0.000999`。
- 命题5：n=1000、delta=0.05、source effect位于[-1,1]时，保守Hoeffding半径 = `0.085894`。
- 冻结门槛总判定：`PASS`。

## 证据边界

- 命题1和命题2来自DSA作为预测概率线性泛函的代数性质；缓存验证是数值审计。
- 命题3是反例：同一operator附近增强一致，不约束不同operator之间的binding方向。
- gamma0接近0是固定场景经验支持，不是任意数据分布下的无条件保证。
- 命题4使用人工注入的已知binding-aligned概率响应，验证识别方向而非现实生成充分性。
- 命题5要求独立source；同一source的operator图像必须作为配对簇共同重采样。
