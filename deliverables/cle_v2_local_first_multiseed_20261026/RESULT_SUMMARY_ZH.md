# CLE-v2 Local-first 多训练种子 Formal 结果

日期：2026-10-26

## 协议与信息边界

- 协议：`cle_v2_local_first_multiseed_12r16b_v1`
- 固定数据场景：CLE-v2 `seed0_split0`
- 训练种子：0、1、2
- 四个匹配臂：HFL/Local × `gamma={0,0.9}`
- 预算：12轮，每客户端每轮16个本地batch，batch size 64
- paired grid：1000个semantic sources × 15个corruption operators，severity 3
- PEW/BER：未使用
- operator ID与binding map：仅用于封存的受控构造与评价，不进入训练

## 三种子结果

| Seed | HFL control DSA | HFL strong DSA | Delta HFL | Local control DSA | Local strong DSA | Delta Local | Delta Comm | Local/HFL | 最低grid acc. | 冻结verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | -0.000245 | 0.119644 | 0.119889 | -0.001914 | 0.106046 | 0.107960 | 0.011929 | 90.05% | 21.4367 | GO |
| 1 | -0.001693 | 0.122025 | 0.123717 | -0.001430 | 0.125857 | 0.127287 | -0.003570 | 102.89% | 21.2683 | GO |
| 2 | -0.001273 | 0.123364 | 0.124638 | -0.001046 | 0.130165 | 0.131210 | -0.006573 | 105.27% | 19.0683 | INCONCLUSIVE_UNDERTRAINED |
| mean +/- sample SD | -- | -- | 0.122748 +/- 0.002518 | -- | -- | 0.122152 +/- 0.012446 | 0.000596 +/- 0.009929 | 99.40% +/- 8.19% | -- | 2 GO; 1 undertrained |

`Delta Comm = Delta HFL - Delta Local`。`Local/HFL`是两个匹配contrast的描述性比值，
不是逐样本中介比例，也不应截断到0--100%。

## 冻结门槛审计

三个训练种子均满足：

- HFL CLE effect不低于0.10，source-bootstrap下界大于0，且4/4客户端方向为正；
- observed binding DSA超过shuffled-binding 95% null，置换检验显著；
- Local CLE effect不低于0.05，source-bootstrap下界大于0，且4/4客户端方向为正；
- Local/HFL描述性比值不低于0.50；
- paired local batch traces匹配。

Seed 2唯一失败项是预注册`L0_learning_floor`：四臂中HFL strong-CLE的operator-grid accuracy为
`19.0683%`，低于`20%`下限`0.9317`个百分点。因此不能写成“三个seed全部正式GO”。正确表述是：

> 方向性shortcut与local-first机制门槛在3/3种子中复现；完整预注册verdict为2/3 GO，
> 另一个种子因学习下限失败而判为undertrained sensitivity result。

Seed 1的`Delta HFL/Delta Local/Delta Comm` source-bootstrap 95%区间分别为
`[0.121899,0.125612]`、`[0.125381,0.129198]`、`[-0.004291,-0.002856]`；
Seed 2分别为`[0.122854,0.126534]`、`[0.129653,0.132937]`、
`[-0.007736,-0.005435]`。Seed 1/2 shuffled-binding检验均为`p=0.000999`。

## 科学解释

`Delta HFL`在三个种子中非常稳定（`0.122748 +/- 0.002518`），说明strong CLE产生约12.3个
概率百分点的binding-directed shift。`Delta Local`同样在三个种子中保持较大且为正；
`Delta Comm`则在seed 0为正、seed 1/2为负，三种子均值接近零且符号不稳定。这支持：

1. strong CLE directional shortcut稳定存在；
2. 该机制在没有通信的本地训练中已经形成；
3. 通信不是跨训练种子稳定的主要放大来源；
4. 不能把通信增量写成普遍为正，也不能把Local/HFL比值解释为严格因果分解。

## 耗时与成本

| Seed | Training | Analysis | Total | V100积分（6分/小时） |
|---:|---:|---:|---:|---:|
| 1 | 6.883 h | 1.867 h | 8.750 h | 约52.50 |
| 2 | 7.018 h | 1.760 h | 8.778 h | 约52.67 |

新增两种子合计约17.53 V100小时、105.17积分，显著超过原先预算。主要成本来自四臂串行训练、
逐轮评价和最终4臂×4客户端×1000 sources×15 operators的完整paired推理。

## 论文使用边界

- 可以正式填入三个seed的原始结果、均值/样本标准差和每seed gate状态。
- 可以把“predominantly local-first”从seed-0观察升级为跨三个训练种子的方向稳定结论。
- 必须同时披露Seed 2学习下限失败；不得写“三seed全部通过完整预注册门槛”。
- source-bootstrap区间与training-seed样本标准差必须分开报告。
- 本实验不评价PEW+BER效果，也不证明自然场景或第二数据集外部有效性。
