# TF-SACI 可行性验证

分支 `exp/tf-saci` 从 `81d7a78` 创建。只新增 SSA 诊断和损失单测。主分支的训练、权重、数据、实验结果及 V0.11 是参考底本。新方法的训练开关全部关闭。

## 代码结构与审查

| 现有入口 | 复用边界 |
| --- | --- |
| `scripts/prepare_cle_v2_factorial_data.py` 与主评价输入包 | 严格复用同一 partition、fit/audit 索引、源图与初始化。主评价用 map2、`alpha=1.0`、`rho=0.5`。 |
| `scripts/run_cle_v2_moderate.py`、`fedprime/methods/rahfl_asymhfl.py` | 主评价两条 AsymHFL 锚点按同一训练预算运行。潜在 Gate 1 插入点是本地任务损失和稳定局部行 ID。原路由、fit mini-batch、模型纯 logits、公共蒸馏和最终评价需要做逐项配对。 |
| `fedprime/data/strict_fit_audit.py`、`fedprime/methods/fedease.py` | AsymHFL 的路由 audit 与梯度 fit 各有角色。PEW+BER 依赖公共合成的粗粒度损坏族监督与 frozen annotation。SSA 只拟合私有 fit 任务标签，各客户端分开，缓存留在各客户端。 |
| `scripts/analyze_cle_v2_moderate.py`、`fedprime/engine/cle_v2_factorial.py` | DSA 在同源的 operator 网格上用预先定义的训练绑定计算方向性变化。标签、绑定与最终预测在冻结后用于正式分析。Gate 0 无需读取 sealed 网格或模型结果。 |

## Gate 0 执行约定

1. `gate0.json` 固定设置。每端、每个 gamma、每套描述符各拟合一个五折 OOF 方案。分折以源图为单位，按源图任务类别分层，并用 frozen fit 分区内原始局部行号进行缓存索引。稀有类别的 2–4 张源图放入不同折。仅一张源图时无法折外拟合该类，入口停止。每一折的标准化和经验类别先验只用该折训练数据。线性探针以 `CE(g + log(pi), y)` 拟合，正则作用在权重，OOF 用未加 prior 的 `g`。只在该折活跃类别上先中心化再截断到 ±3，截断后的均值允许偏离零。
2. `phi_tex` 预先指定为主审计。灰度为 BT.601 权重，RGB 缩放到 [0,1]，Laplacian 只用内部像素，峰度采用四阶标准矩。块方差为非重叠 4×4。JPEG 边界比在 8px 网格合并横竖边界，分母最小 1e−6。色度对每个通道合并横竖梯度。 `phi_full` 额外含五个强度统计，只作诊断对照。
3. `fit` 对完整文件作完整性校验，只提取 fit 图像和类别用于学习。audit 的源图 ID 仅用于排除与 fit 同源。拟合命令不读取私有真实损坏元数据、路由 audit 图像或任务标签、最终测试、sealed DSA、PEW 注释。先固定缓存和哈希。`audit` 只读取已固定缓存，随后用 fit 的真实损坏元数据分层。operator 切片按该切片实际有支持的类宏平均，保留支持数，同时报告训练折众数基线。不同 operator 的标签支持可大不相同。
4. 针对 `phi_tex`，预先记录 clean balanced accuracy 高出同切片 fold-prior 基线至少 5 pp 为描述性语义警示。另看同源、同折的 clean/损坏交集上 `gamma09 − gamma00`，以区分现象。该描述阈值不构成统计显著性检验。clean 预测性可以来自图像纹理、形状或色度与类相关。gamma0 不要求随机水平。OOF 证据仅反映固定数据和拆分下的标签可预测性。

在 worktree 根目录依次运行，输入传现有展开的主评价包绝对路径，输出传全新的未跟踪目录。CPU 运行。

```powershell
python -m pytest -q tests/test_tf_saci_ssa.py
python -m experiments.tf_saci.gate0 fit --package-root $package --output $run
python -m experiments.tf_saci.gate0 audit --package-root $package --output $run
```

`FIT_MANIFEST.json` 记录版本、环境、初始权重、训练分区、输入和缓存哈希及耗时。缓存含训练源图 ID、标签及 OOF logits，应留在本地。`GATE0_AUDIT.json` 和 `GATE0_SLICES.csv` 记录聚合结果。重复运行请使用新目录。

## Gate 1 及后续推进条件

Gate 1 只考虑 SSA-PoE 及同一 SSA 的 CE-anchor 两臂。两条冻结锚点的 DSA / Clean / Mixed 为 AsymHFL–ERM 的 `0.184514 / 49.700 / 43.354` 与 AsymHFL+PEW+BER 的 `0.090024 / 45.350 / 41.562`。40 轮，每端每轮最多 64 个私有 mini-batch，每 batch 64 张，公开 batch size 128，每轮 4 个公开 batch，无预训练，seed 0。Axis I 的 40×16 是另一协议。

在有独立、与 fit/audit/final 按源图隔离的开发集之前，Gate 1 不启动。开发集来源与参数选择规则需先冻结。候选 `lambda` 取 0.5 或 1.0，CE-anchor 的 `eta=0.5` 为拟议初值。开发选择后才进行一次统一的最终评价。复用同一初始化、数据增广、私有 batch 次序、公开批次和 audit 路由。用同一 evaluator 报告 DSA、Grid、Clean、Corrupt、Mixed 及逐端结果、计算成本，WCCA/CFG 从已核验的原口径独立汇报。历史锚点的结果校验和恢复包只做哈希核验，最终模型不进入 SSA 拟合。

双赢目标沿用附件中的方向，DSA 达到冻结 PEW+BER 水平，Clean 和 Mixed 的容忍界限在开发试验前写进 Gate 1 注册并以 ERM 为参照。Gate 1 只有一个训练种子，来源图 bootstrap 只刻画冻结模型下的评价变异。Gate 1 通过后才设计 Gate 1b 的 CC-Mix only 与 SSA+CC-Mix 消融，再考虑 Gate 2 的 PS-PID。当前两者均未接入。

训练期相加偏置属于已有的 PoE 去偏思路。带先验的交叉熵也只有在模型正确设定、先验正确及总体风险最优等条件下，才能把最优 logit 解释为与类条件密度相差一个样本依赖常数。当前有限样本正则化线性探针可能混入语义，具有预测性无法确认其识别了纯损坏捷径，不能取代 paired DSA。[Clark et al., EMNLP-IJCNLP 2019](https://aclanthology.org/D19-1418/)
