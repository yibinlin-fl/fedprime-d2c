# FedPRIME-D2C 实验配置与运行指南

本文档用于长期记录 FedPRIME-D2C 项目需要运行的实验、配置文件路径、
实验目的、关键参数、运行顺序和结果位置。

恢复项目时建议依次阅读：

```text
ARCHITECTURE.md
PROJECT_STATE.md
EXPERIMENT_GUIDE_ZH.md
TODO_NEXT.md
```

## 一、当前实验目标

项目目标是在以下三个条件同时存在时，与 RAHFL 正面对比：

```text
模型异构：4 个客户端使用不同模型
数据异构：Dirichlet Non-IID 类别划分
数据损坏：RAHFL-style random_corrupt_1
```

当前主要比较：

```text
RAHFL = AugMix + DCL + AsymHFL
FedPRIME-D2C = PRIME + D2C
```

FedPRIME-D2C 当前不默认加入 DCL，以便首先验证 PRIME + D2C 本身能否击败
RAHFL。FedPRIME-D2C + DCL 作为后续严格控制实验。

## 二、所有主实验共享的数据与公平性设置

当前 alpha=0.5 主实验统一使用：

```yaml
data:
  private_corrupt_rate: 1
  test_corrupt_rate: 1
  partition: dirichlet
  dirichlet_alpha: 0.5
  private_samples_per_client: 10000
  partition_indices_path: outputs/partitions/cifar10c_alpha05_seed0_clients4_samples10000.npz
```

含义：

```text
训练数据：RAHFL-style 100% 随机损坏 CIFAR-10
测试数据：RAHFL-style 100% 随机损坏 CIFAR-10
公共数据：CIFAR-100
客户端数量：4
每个客户端：10000 个训练样本
```

四个异构客户端模型：

```text
ResNet10
ResNet12
ShuffleNet
Mobilenetv2
```

RAHFL 与 FedPRIME-D2C 读取同一个固定 partition `.npz` 文件，因此客户端
看到的样本索引完全一致。

## 三、今天正在运行的实验

### 3.1 Kaggle T4 核心对比：RAHFL vs FedPRIME-D2C warmup=3

这是当前优先级最高的实验。

启动脚本：

```text
scripts/run_kaggle.sh
```

默认运行配置：

```text
configs/kaggle_t4_rahfl.yaml
configs/kaggle_t4_fedprime_d2c_warmup3.yaml
```

#### RAHFL 配置

路径：

```text
configs/kaggle_t4_rahfl.yaml
```

算法：

```text
AugMix + DCL + AsymHFL
```

关键设置：

```yaml
experiment_name: rahfl_cifar10c_alpha05_cr1_t4
method_name: rahfl
train:
  rounds: 40
  local_epochs: 1
  batch_size: 64
  public_batch_size: 128
method:
  use_prime: false
  augmix_module: jsd
  cl_module: dcl
  lambda_jsd: 12.0
```

目的：

```text
得到当前 RAHFL 最强基线的完整 40 轮结果。
```

结果目录：

```text
outputs/rahfl_cifar10c_alpha05_cr1_t4/
```

#### FedPRIME-D2C warmup=3 配置

路径：

```text
configs/kaggle_t4_fedprime_d2c_warmup3.yaml
```

算法：

```text
PRIME + D2C
```

关键设置：

```yaml
experiment_name: fedprime_d2c_cifar10c_alpha05_cr1_t4_warmup3
method_name: fedprime_d2c
train:
  rounds: 40
  local_epochs: 1
  batch_size: 64
  public_batch_size: 128
method:
  use_prime: true
  communication: d2c
  d2c_warmup_rounds: 3
  lambda_jsd: 12.0
  lambda_d2c: 1.0
```

Warmup 行为：

```text
round 0、1、2：只运行本地 PRIME，d2c_loss=0
round 3 开始：运行本地 PRIME + 公共数据 D2C
```

目的：

```text
避免客户端模型刚初始化时使用不可靠 public logits 相互蒸馏。
验证模型先完成少量本地学习，再启动 D2C 是否能够击败 RAHFL。
```

结果目录：

```text
outputs/fedprime_d2c_cifar10c_alpha05_cr1_t4_warmup3/
```

#### Kaggle 运行命令

挂载数据集：

```text
/kaggle/input/fedprime-data
```

克隆并导入数据：

```bash
git clone --depth 1 https://github.com/yibinlin-fl/fedprime-d2c.git
cd /kaggle/working/fedprime-d2c

python scripts/import_prepared_data.py \
  --source /kaggle/input/fedprime-data \
  --destination .
```

正式运行：

```bash
RUN_INSTALL=0 RUN_PREPARE_DATA=0 RUN_DEBUG=0 bash scripts/run_kaggle.sh
```

汇总结果：

```text
outputs/summary.csv
outputs/summary.md
```

### 3.2 当前实验需要观察的指标

#### avg_acc：平均客户端准确率

定义：

```text
avg_acc = 4 个异构客户端测试准确率的平均值
```

意义：

```text
衡量整个联邦系统的总体分类性能。
这是与 RAHFL 比较时最直接的主指标。
```

健康现象：

```text
随轮次总体上升，允许个别轮次小幅回落。
FedPRIME-D2C 与 RAHFL 的差距逐渐缩小，或最终超过 RAHFL。
```

危险信号：

```text
长期停留在约 10% 的随机猜测水平。
连续多轮明显下降。
FedPRIME-D2C 比 RAHFL 落后 8 至 10 个点以上且差距继续扩大。
```

#### worst_acc：最弱客户端准确率

定义：

```text
worst_acc = 4 个客户端测试准确率中的最小值
```

意义：

```text
衡量最弱客户端是否受益。
对于 Non-IID 场景尤其重要，因为 D2C 的目标之一是帮助本地稀缺类别较多的客户端。
```

健康现象：

```text
总体上升。
FedPRIME-D2C 的 worst_acc 高于 RAHFL，或差距比 avg_acc 更有优势。
```

危险信号：

```text
avg_acc 上升但 worst_acc 长期不升或明显下降。
说明系统只改善强客户端，弱客户端可能被错误 teacher 伤害。
```

#### local_loss：本地训练损失

RAHFL 的 `local_loss` 不是单独的交叉熵，而是：

```text
local_loss_RAHFL = CE + lambda_jsd × JSD + DCL
lambda_jsd = 12
```

其中：

```text
CE：干净视图分类损失
JSD：干净视图与两个 AugMix 强增强视图的一致性损失
DCL：original、weak、strong 特征之间的对比与分布匹配损失
```

因此 RAHFL 的 `local_loss` 数值显著大于普通 CE 是正常的。例如当前实验中：

```text
round 0：15.1687
round 5：14.1863
```

该趋势是健康的，因为损失有限并且持续下降。

FedPRIME-D2C 无 DCL 主配置的本地损失为：

```text
local_loss_FedPRIME-D2C = CE + lambda_jsd × JSD
```

重要注意事项：

```text
不能直接用 RAHFL local_loss 与 FedPRIME-D2C local_loss 的绝对值判断谁更好。
两者包含的损失项不同，RAHFL 额外包含 DCL。
应该关注各自损失是否有限、稳定，以及在自身训练过程中是否总体下降。
```

健康现象：

```text
有限、无 NaN/Inf、总体下降或稳定。
```

危险信号：

```text
突然成倍增大并持续上涨。
出现 NaN 或 Inf。
准确率长期不升，同时 local_loss 也没有下降趋势。
```

#### col_loss：RAHFL AsymHFL 通信损失

定义：

```text
col_loss = RAHFL 在公共 CIFAR-100 数据上的 AsymHFL KL 蒸馏损失
```

RAHFL 会让较弱客户端向当前准确率不低于自己的客户端学习：

```text
KL(student public prediction || selected teacher public prediction)
```

当前实验观测：

```text
round 0：0.1735
round 1 至 5：约 2.1 至 3.0
```

这是正常现象。第 0 轮所有随机初始化模型的输出都接近均匀分布，彼此较相似，
因此 KL 较小。随着客户端在不同 Non-IID 数据上学习，预测逐渐分化，通信 KL
上升到非零值，然后在一定范围内波动。

健康现象：

```text
保持有限，在一定范围内波动。
通信后 avg_acc 和 worst_acc 总体改善。
```

危险信号：

```text
出现 NaN/Inf。
连续多轮快速增大到远高于此前量级，同时准确率下降。
长期为 0，可能意味着没有客户端发生有效协作。
```

注意：

```text
col_loss 并不是越低越好。
较低可能表示客户端预测一致，也可能表示没有有效学习信号。
必须结合准确率趋势解释。
```

#### d2c_loss：FedPRIME-D2C 公共数据蒸馏损失

定义：

```text
d2c_loss = 客户端在公共数据上向 D2C teacher 学习的 complementary KD 损失
```

它由 D2C teacher、客户端 prior 和 complementary class 权重共同决定，并包含
知识蒸馏温度的 `T²` 缩放。

Warmup 是否生效的检查：

```text
FedPRIME-D2C round 0、1、2 的 d2c_loss 应为 0
round 3 开始 d2c_loss 应变为非零有限值
```

健康现象：

```text
warmup 后为非零有限值。
可以波动，但不应持续爆炸。
启用 D2C 后 avg_acc 或 worst_acc 的增长速度改善。
```

危险信号：

```text
warmup 后仍长期为 0，可能表示 D2C 没有执行。
出现 NaN/Inf。
数值持续快速增大，同时准确率下降，可能是 teacher 或 prior 不稳定。
```

注意：

```text
d2c_loss 与 RAHFL col_loss 的公式不同，绝对值不能直接横向比较。
```

#### head_acc、tail_acc、missing_acc：类别覆盖诊断指标

这些指标由以下脚本在训练完成后生成：

```text
scripts/diagnose_underrepresented.py
```

含义：

```text
head_acc：客户端本地多数类别的准确率
tail_acc：客户端本地少数类别的准确率
missing_acc：客户端本地完全缺失类别的准确率
```

意义：

```text
如果 D2C 的设计有效，tail_acc 和 missing_acc 应比普通 LogitAvg 或 RAHFL 更有优势。
这是证明 D2C 确实缓解 Non-IID 类别知识缺失的重要机制指标。
```

#### corruption group accuracy：细分损坏鲁棒性

由以下脚本评估：

```text
scripts/evaluate_corruptions.py
```

关注：

```text
noise、blur、weather、digital 等损坏组准确率
```

意义：

```text
判断 PRIME 带来的鲁棒性提升是否覆盖不同损坏类型，而不只是当前随机损坏缓存。
```

#### 多 seed 均值与标准差

最终论文结果应至少报告：

```text
seed 0、1、2 的 mean ± std
```

意义：

```text
mean 衡量平均性能。
std 衡量训练稳定性。
只有单 seed 的优势可能是偶然结果，不能作为最终论文结论。
```

### 3.3 如何判断当前核心对比是否成功

初步正向信号：

```text
FedPRIME-D2C avg_acc 上升趋势不弱于 RAHFL
FedPRIME-D2C worst_acc 更高或差距逐渐缩小
d2c_loss 在 warmup 后为非零有限值并保持稳定
local_loss 有限且总体下降
```

强论文结果：

```text
FedPRIME-D2C 最终 avg_acc 超过 RAHFL
FedPRIME-D2C worst_acc 明显超过 RAHFL
tail_acc / missing_acc 提升，证明 D2C 帮助本地稀缺类别
alpha=0.1 下优势比 alpha=0.5 更明显
多 seed 下仍保持提升且标准差可控
```

## 四、下一批必须运行的实验

### 4.1 Warmup 消融：warmup=0 vs warmup=3

配置：

```text
configs/kaggle_t4_fedprime_d2c.yaml
configs/kaggle_t4_fedprime_d2c_warmup3.yaml
```

差异：

```yaml
d2c_warmup_rounds: 0
d2c_warmup_rounds: 3
```

目的：

```text
判断早期随机 public logits 是否伤害 D2C。
证明 warmup 是否真实贡献准确率与稳定性。
```

运行：

```bash
python scripts/run_grid.py \
  configs/kaggle_t4_fedprime_d2c.yaml \
  configs/kaggle_t4_fedprime_d2c_warmup3.yaml
```

注意：

```text
当前 RAHFL 从 round 0 就执行通信。
后续若要最严格比较，应为 RAHFL 增加同样的 communication warmup。
```

### 4.2 LogitAvg + PRIME 基线

配置：

```text
configs/logitavg_prime_cifar10c.yaml
configs/logitavg_prime_cifar10c_alpha01.yaml
```

算法：

```text
PRIME + 普通 public logits 平均蒸馏
```

关键设置：

```yaml
method:
  communication: logit_avg
```

目的：

```text
证明提升来自 D2C 的 prior debias、class-balanced aggregation 和
complementary KD，而不是只要使用公共数据蒸馏就能提升。
```

注意：

```text
当前 LogitAvg 正式配置 batch_size=256，不建议直接在单张 T4 上运行。
后续应创建 kaggle_t4_logitavg_prime.yaml。
```

### 4.3 Severe Non-IID：Dirichlet alpha=0.1

配置：

```text
configs/fedprime_d2c_cifar10c_alpha01.yaml
configs/logitavg_prime_cifar10c_alpha01.yaml
configs/fedprime_d2c_dcl_cifar10c_alpha01.yaml
```

关键设置：

```yaml
data:
  dirichlet_alpha: 0.1
```

目的：

```text
验证 D2C 在更严重类别偏斜下是否获得更大的优势。
这是最契合 D2C 论文故事的设置。
```

预期论文模式：

```text
alpha=0.5：FedPRIME-D2C 接近或超过 RAHFL
alpha=0.1：FedPRIME-D2C 优势进一步扩大
```

注意：

```text
当前 alpha=0.1 配置 batch_size=256。
在 Kaggle T4 上运行前，应增加对应的 T4-safe 配置和固定 alpha=0.1 partition。
```

### 4.4 强控制实验：PRIME + DCL + AsymHFL vs PRIME + DCL + D2C

配置：

```text
configs/cifar10c_rahfl_prime.yaml
configs/fedprime_d2c_dcl_cifar10c.yaml
```

算法：

```text
RAHFL+PRIME = PRIME + DCL + AsymHFL
FedPRIME-D2C+DCL = PRIME + DCL + D2C
```

目的：

```text
让两边的本地训练都使用 PRIME + DCL，只比较通信模块：
AsymHFL vs D2C。
```

这是证明 D2C 通信机制价值的重要严格控制实验。

注意：

```text
当前两个配置 batch_size=256，Kaggle T4 可能 OOM。
运行前应创建 T4-safe 版本。
```

## 五、D2C 模块消融实验

消融配置统一位于：

```text
configs/ablations/
```

当前消融配置都基于 alpha=0.5、corrupt_rate=1。

### 5.1 去掉 PRIME

配置：

```text
configs/ablations/fedprime_d2c_no_prime.yaml
```

关键设置：

```yaml
method:
  use_prime: false
```

目的：

```text
验证 PRIME 对抗数据损坏的贡献。
```

### 5.2 去掉 prior debias

配置：

```text
configs/ablations/fedprime_d2c_no_prior_debias.yaml
```

关键设置：

```yaml
d2c:
  use_prior_debias: false
```

目的：

```text
验证去除客户端类别先验污染是否是 D2C 的关键来源。
```

### 5.3 去掉 class-balanced aggregation

配置：

```text
configs/ablations/fedprime_d2c_no_class_balanced.yaml
```

关键设置：

```yaml
d2c:
  use_class_balanced: false
```

目的：

```text
验证按类别选择更擅长客户端的重要性。
```

### 5.4 去掉 complementary KD

配置：

```text
configs/ablations/fedprime_d2c_no_complementary_kd.yaml
```

关键设置：

```yaml
method:
  use_complementary_kd: false
```

目的：

```text
验证“主要学习本地稀缺类别、保护本地优势类别”的个性化蒸馏是否有效。
```

### 5.5 Oracle prior

配置：

```text
configs/ablations/fedprime_d2c_oracle_prior.yaml
```

关键设置：

```yaml
method:
  prior_source: oracle
```

目的：

```text
使用客户端真实标签分布作为 prior 上界。
判断 predicted prior 的估计质量，以及 D2C 尚有多少提升空间。
```

### 5.6 Adaptive beta + EMA prior + self-gate

配置：

```text
configs/ablations/fedprime_d2c_adaptive_ema_gate.yaml
```

关键设置：

```yaml
method:
  use_self_gate: true
d2c:
  adaptive_beta: true
  ema_alpha: 0.9
```

目的：

```text
adaptive beta：根据客户端 prior 偏斜程度自动调整去偏强度
EMA prior：平滑轮次间 prior 估计，降低抖动
self-gate：减少客户端在自身已有优势类别上的过度蒸馏
```

注意：

```text
当前消融配置大多使用 batch_size=256。
批量在 Kaggle T4 运行前，需要生成 T4-safe 消融配置。
```

## 六、Debug 配置

Debug 配置只用于验证代码路径，不用于论文结果。

### 6.1 FedPRIME-D2C debug

```text
configs/debug_fedprime_d2c_cifar10c.yaml
```

```text
1 轮、小数据量、PRIME + D2C、warmup=0
```

### 6.2 FedPRIME-D2C + DCL debug

```text
configs/debug_fedprime_d2c_dcl_cifar10c.yaml
```

```text
1 轮、PRIME + DCL + D2C
```

### 6.3 LogitAvg + PRIME debug

```text
configs/debug_logitavg_prime_cifar10c.yaml
```

```text
1 轮、PRIME + 普通 logit averaging
```

## 七、平台专用配置

### 7.1 Kaggle T4

```text
configs/kaggle_t4_rahfl.yaml
configs/kaggle_t4_fedprime_d2c.yaml
configs/kaggle_t4_fedprime_d2c_warmup3.yaml
```

特点：

```text
batch_size=64
public_batch_size=128
避免单张 T4 上 RAHFL AugMix+DCL OOM
```

### 7.2 普通服务器 / C2NET

```text
configs/server_safe_rahfl.yaml
configs/server_safe_fedprime_d2c.yaml
```

启动：

```text
scripts/run_server.sh
```

特点：

```text
batch_size=64
支持准备数据、审计划分、运行比较、汇总结果
UPLOAD_C2NET=1 时将 outputs 复制到平台 output_path 后回传
```

当前 server-safe FedPRIME-D2C 使用：

```yaml
d2c_warmup_rounds: 0
```

若服务器主实验也要使用 warmup=3，应新增独立配置，避免覆盖 warmup=0。

## 八、诊断与结果分析脚本

### 8.1 数据异构审计

```text
scripts/audit_partition.py
```

输出：

```text
outputs/partition_audit/<experiment_name>/
  client_class_counts.csv
  client_class_proportions.csv
  client_class_counts.png
  partition_summary.json
```

目的：

```text
证明客户端类别分布确实 Non-IID，并为论文提供热力图。
```

### 8.2 弱势类别诊断

```text
scripts/diagnose_underrepresented.py
```

输出指标：

```text
head_acc
tail_acc
missing_acc
```

目的：

```text
验证 D2C 是否真正帮助客户端本地稀缺或缺失类别。
```

### 8.3 损坏组评估

```text
scripts/evaluate_corruptions.py
```

目的：

```text
在 official CIFAR-10-C 上评估 noise、blur、weather、digital 等损坏组。
```

当前限制：

```text
需要 official CIFAR-10-C 的逐损坏 .npy 文件。
目前 prepared data 是 RAHFL-style random corruption，不是 official CIFAR-10-C。
```

### 8.4 多 seed

```text
scripts/run_multiseed.py
```

目的：

```text
至少使用 seed 0、1、2 报告 mean ± std，避免单次结果偶然性。
```

### 8.5 结果汇总

```text
scripts/summarize_results.py
```

输出：

```text
outputs/summary.csv
outputs/summary.md
```

## 九、推荐实验执行顺序

### 阶段 A：先确认能否正面对抗 RAHFL

1. `configs/kaggle_t4_rahfl.yaml`
2. `configs/kaggle_t4_fedprime_d2c_warmup3.yaml`
3. 对比 `avg_acc`、`worst_acc` 和训练趋势

### 阶段 B：确认 warmup 是否必要

1. `configs/kaggle_t4_fedprime_d2c.yaml`
2. 与 warmup=3 结果对比
3. 后续增加 RAHFL communication warmup=3，做严格公平检查

### 阶段 C：证明 D2C 本身有效

1. LogitAvg + PRIME
2. FedPRIME-D2C
3. underrepresented class diagnosis

### 阶段 D：验证 Severe Non-IID 论文主张

1. alpha=0.5
2. alpha=0.1
3. 检查 alpha 越小，D2C 优势是否越明显

### 阶段 E：严格控制与消融

1. RAHFL+PRIME vs FedPRIME-D2C+DCL
2. D2C 核心组件消融
3. adaptive beta / EMA / self-gate 增强版

### 阶段 F：完整论文结果

1. seed 0、1、2
2. clean test 与 corrupted test
3. official CIFAR-10-C corruption group
4. 结果表格、曲线、Non-IID 热力图和弱势类别图

## 十、当前尚缺少的配置

为节省 Kaggle T4 显存，后续应补充：

```text
kaggle_t4_logitavg_prime.yaml
kaggle_t4_fedprime_d2c_alpha01.yaml
kaggle_t4_rahfl_alpha01.yaml
kaggle_t4_rahfl_prime.yaml
kaggle_t4_fedprime_d2c_dcl.yaml
T4-safe ablation configs
RAHFL communication warmup=3 config and implementation
```

在这些配置创建前，不应直接使用 batch_size=256 的正式配置在单张 T4 上跑。
