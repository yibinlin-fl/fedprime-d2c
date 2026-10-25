# CLE-HFL论断—证据审计（V0.8）

审计日期：2026-10-14。

## 状态定义

- **已证实**：现有Formal、缓存复算或数学证明足以支撑限定后的表述。
- **条件性**：在一个或少量受控协议中成立，不能外推为普遍结论。
- **待证据**：论文骨架可以保留位置，但正文不能写成已经完成。
- **禁止**：与现有证据冲突或超过方法信息边界。

## 主论断台账

| 论断 | 状态 | 当前证据 | 正文允许写法 | 禁止写法 |
|---|---|---|---|---|
| strong CLE会诱发沿binding方向的预测偏移 | 已证实 | paired counterfactual、DSA、shuffled-binding null；seed-0机制实验 | strong controlled CLE induces a directional shortcut in our protocol | corruption普遍导致该现象；真实部署已被证明 |
| DSA能够识别binding-directed概率质量移动 | 已证实但有假设 | paired cancellation、交换性零点、混合线性、source-level inference | under the stated semantic-preservation and registered-binding assumptions | DSA无条件等价于因果shortcut强度 |
| CLE主要local-first形成 | 条件性 | matched HFL-vs-Local seed 0；通信add-on较小 | predominantly local-first in the controlled setting | 联邦通信无关；90.05%是严格因果分解；多seed稳定性已完成 |
| PEW+BER能够抑制CLE | 条件性已证实 | original/map1、固定预算map2三seed、FedDF-fidelity | suppresses DSA across controlled maps and two communication settings | 对任意HFL都无损有效；统一提高accuracy |
| BER区别于通用tail/worst-group重加权 | 条件性已证实 | matched ERM/CVaR/共享PEW-GroupDRO固定预算比较 | better matched to class-conditional support imbalance in the completed protocol | 已在所有预算、数据集和通信方法上战胜CVaR/GroupDRO |
| PEW是taxonomy-free环境发现 | 禁止 | PEW由公共合成corruption标签监督 | taxonomy-assisted public environment witness | taxonomy-free、自动发现任意真实环境 |
| proxy下降可证明真实CLE下降 | 禁止 | proxy non-identifiability定理与CVRS反例 | target-aligned paired DSA remains necessary | PEW准确率/JSD/任意proxy loss足以认证真实harm |
| FedDF上utility统一提升 | 禁止 | FedDF DSA下降但冻结utility gate未过 | cross-base shortcut suppression with a utility boundary | architecture-uniform utility improvement |
| moderate dose response | 已完成（seed 0） | gamma 0/.5/.9 matched Formal与shuffled null/source bootstrap | DSA increases monotonically with controlled binding strength in this seed | 现实发生率、跨seed线性规律 |
| moderate AsymHFL方法对照 | 已完成（seed 0） | batch-matched ERM/PEW+BER Formal，4/4客户端DSA下降 | PEW+BER substantially suppresses CLE with a nonuniform utility trade-off | 无损、完全消除、通用插件 |
| moderate Axis-II八行领域排名 | 部分待证据 | Local/FedMD/FedProto/AsymHFL-ERM/AsymHFL+PEW+BER已完成，其余三行待跑 | five completed rows are reported descriptively | 八方法排名或SOTA结论 |
| 第二private dataset外部复现 | 待证据 | 尚未完成 | explicit limitation / planned evidence | 已跨数据集泛化 |

## 已锁定的关键数字

以下数字只能按其原协议使用，不能跨预算混合：

- HFL CLE effect：`0.119889`；Local CLE effect：`0.107960`；communication add-on：`0.011929`。
- held-out map2固定预算三seed：ERM DSA `0.2505±0.0087`，CVaR `0.0867±0.0055`，PEW+GroupDRO `0.2054±0.0080`，PEW+BER `0.0715±0.0113`。
- moderate Axis II seed 0：Local DSA/Grid/Clean/Corr/Mixed为`0.168374/39.830/51.425/39.508/45.467`；FedMD为`0.167835/38.732/50.400/38.378/44.389`；FedProto为`0.160788/39.250/50.800/38.937/44.868`。
- moderate dose seed 0：ERM gamma 0/.5/.9 DSA为`-0.000730/0.039679/0.184514`；相邻增量source-bootstrap CI95均严格大于零。
- moderate AsymHFL seed 0：ERM与PEW+BER DSA为`0.184514/0.090024`，相对下降51.21%，4/4客户端同向；Grid/Corr为`+0.967/+0.765`pp，Clean/Mixed为`-4.350/-1.793`pp。

## 写作传播规则

1. Abstract只概括已经完成的证据；最终投稿数字必须在Axis I和Axis II冻结后统一替换。
2. Introduction可以陈述研究闭环，但不能把pending实验写成贡献结果。
3. Methods必须明确PEW使用公共合成taxonomy监督，不接触私有真实operator标签。
4. Experiments必须区分strong stress与moderate utility协议，不能把两者当成两个独立数据集。
5. Discussion必须保留single private dataset、固定partition、taxonomy-assisted和训练随机性覆盖不足等边界。
6. Conclusion不能出现“universal plug-in”“state of the art”或“real-world effectiveness”。

## 本轮结论

当前稿已经足以进入持续写作，但还不能冻结投稿版。缺口主要是证据覆盖而不是再增加一个新模块：moderate Axis II剩余三行（可按预算收缩）、local-first多seed、第二private dataset和有界taxonomy stress。现有40轮三seed Axis I继续作为正式主证据，不再默认要求昂贵复跑。
