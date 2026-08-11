# Taxonomy-free BER 替代的可识别性边界

日期：2026-08-11
状态：理论与协议审计；未提出可进入 runner 的新方法

## 1. 为什么要先检查可识别性

hard BER 的收益来自平衡同一类别内的环境风险。删除 PEW 后，如果训练过程既不观察环境，也不
获得能区分环境的额外信号，仅靠普通 `(x,y)` 风险不可能知道两个视觉上不同的潜在分解中哪个才是
真实环境分解。PIE、C3R、CRSR 的负结果已经分别表明：公共干预表征、退化遗憾和预测残差谱在
当前协议中都没有稳定提供该额外信息。

一个自然设想是利用不同客户端具有不同 `P_k(E|Y)`：把客户端身份当作多个未知环境混合视图，
不推断单样本环境，只约束同类别风险的跨客户端对比。下面给出其能力上限。

## 2. Mixture Contrast Identifiability（MCI）

固定类别 `c`。设潜在环境数为 `M`，共享决策对象 `h` 在各环境上的风险向量为

\[
\rho_c(h)=[R_{c,1}(h),\ldots,R_{c,M}(h)]^\top.
\]

客户端条件混合矩阵为 `Pi_c`，第 `k` 行是未知的 `P_k(E|Y=c)`。训练能观察到的客户端-类别
风险仅为

\[
r_c(h)=\Pi_c\rho_c(h).
\]

令 `H_K=I-\mathbf 1\mathbf 1^\top/K` 为客户端中心化矩阵，定义混合对比

\[
\mathcal C_c(h)=\|H_Kr_c(h)\|_2
=\|H_K\Pi_c\rho_c(h)\|_2.
\]

它不需要知道 `Pi_c` 或环境标签，只需要同一个决策对象在各客户端上的类条件风险。

### 命题 1：完全覆盖条件

因为 `Pi_c 1_M=1_K`，环境常数风险必定位于 `H_K Pi_c` 的零空间。若

\[
\operatorname{null}(H_K\Pi_c)=\operatorname{span}(1_M),
\]

则 `C_c(h)=0` 当且仅当所有环境风险相等。这要求

\[
\operatorname{rank}(H_K\Pi_c)=M-1\le K-1.
\]

因此 `K` 个客户端最多识别 `K-1` 个环境风险对比。它不能凭通信凭空补出更多独立环境方向。

### 命题 2：部分覆盖界

对中心化环境风险 `rho_c^perp` 在 `row(H_K Pi_c)` 内的分量，有

\[
\|P_{\rm id}\rho_c^\perp\|_2
\le \frac{\mathcal C_c(h)}{\sigma_{\min}^+(H_K\Pi_c)}.
\]

零空间中的环境风险差异完全不可见。换言之，混合对比只能控制由客户端混合差异真正激发的环境
方向，不能给所有隐藏环境提供 BER 式保证。

## 3. 当前 CLE-HFL v2 的协议审计

对固定 `seed0_split0`，4 个客户端意味着理论最大中心化秩为 3。先按协议元数据中的四个真实
corruption family（noise/blur/weather/digital）检查每个类别的主导 family，得到：

| 类别 | 四客户端主导 family | 最大有效秩 |
|---:|---|---:|
| 0 | blur, blur, weather, weather | 1 |
| 1 | weather, digital, blur, weather | 2 |
| 2 | digital, blur, digital, blur | 1 |
| 3 | blur, weather, noise, digital | 3 |
| 4 | noise, noise, digital, digital | 1 |
| 5 | blur, digital, blur, blur | 1 |
| 6 | weather, blur, noise, blur | 2 |
| 7 | digital, digital, weather, noise | 2 |
| 8 | digital, digital, digital, digital | 0 |
| 9 | digital, weather, digital, noise | 2 |

只有类别 3 激发了四个真实 family 的全部 3 个对比；类别 8 的跨客户端 family 混合完全没有
变化。PEW 还可能产生 `clean/unknown` 预测组，但客户端真实混合并未因此获得新的独立激发；把
它们计入只会增加待控制方向，不会修复当前秩缺失。这不是模型训练失败，而是由冻结数据映射直接
决定的信息缺失。

若直接按 11 个 seen operator 计算，则完整覆盖需要秩 10，而 4 个客户端的上限仍为 3；十个
类别的实际 operator-level 秩只有 2 或 3，没有任何类别能够完整覆盖。

此外，真实 HFL 中每个客户端使用不同 backbone，严格说客户端风险是 `R_{k,c,e}(h_k)`，而不是
同一个 `R_{c,e}(h)`。架构能力、标签支持与环境混合会共同改变跨客户端风险。因此即使某一类别
的混合矩阵满秩，直接惩罚 client-class 风险方差也不能干净归因于环境不变性。

## 4. 对下一方法的硬约束

仅做以下任一项都不足以形成下一主方法：

1. 对 client-class 风险做 GroupDRO/方差惩罚；它既受异构架构混淆，也没有覆盖所有类别。
2. 把 client identity 重新命名成 environment；它不能看到客户端内部的少数环境。
3. 用梯度聚类/梯度图补足秩；这会回到已否决的 LCC，并与 GRASP/GoG 碰撞。
4. 用高损失尾部补足秩；这会退化为 CVaR/JTT/EVaLS 类路线。
5. 继续调整 PIE/C3R/CRSR；这些信号已经由冻结门槛否定。

若仍要无五类标签地接近 BER，方法必须明确新增一种可识别信息来源：例如真正能替换而非叠加
原 nuisance 的干预、经独立门槛验证的外部 nuisance 表征，或更多独立且可比较的混合视图。
这项新增假设必须写进问题设定，不能藏在模块内部。

## 5. 当前优先级

| 候选 | 新颖性 | 理论合理性 | 成本 | 归因 | 论文价值 | 决策 |
|---|---:|---:|---:|---:|---:|---|
| LCC 梯度聚类/图 + 公共下降 | 低 | 中 | 中高 | 差 | 低 | `NO-GO`，外部碰撞 |
| 纯 client-class 混合对比 | 中 | 低到中 | 低 | 差 | 低 | 不实现；覆盖秩不足 |
| Fourier/MixStyle nuisance 交换 | 低 | 中 | 低 | 中 | 低 | 不作为原创；已有直接方法 |
| 经验证的额外 nuisance side information | 待定 | 取决于可识别性 | 中高 | 可设计 | 潜在中高 | 仅保留为下一理论搜索方向 |
| hard PEW + hard BER | 低（五类假设脆弱） | 高 | 已完成 | 强 | 仅作性能参照 | 保留 reference，不冒充最终创新 |

当前没有候选同时通过“新颖性、可识别性、冻结负结果边界、清晰归因”四个门槛。因此正确的下一步
不是启动另一个训练实验，而是先定义额外信息来源及其最小假设，再做不训练或一步更新的可证伪审计。

复核入口（只读取协议 metadata，不加载数据或模型）：

```text
python scripts/audit_mixture_contrast_identifiability.py --level=family
python scripts/audit_mixture_contrast_identifiability.py --level=operator
```
