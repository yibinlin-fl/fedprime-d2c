# 本轮审稿的事实锁与证据状态

日期：2026-10-31。用途：先审V0.11的表达与论证，不启动实验，不自动改正文。
完整推导、历史数值和引用边界见同目录完整交接；本文只提供当前状态入口。

## 1. 权威顺序

科学事实：本文件当前状态 → 完整交接0.0节和冻结结果 → 当前稿。
文字底本：本包V0.11 PDF/源码；V0.6不是改稿底本。
视觉候选：figure_candidates/，它们尚未接入V0.11 PDF。
如果原稿与事实锁冲突，请指出并提出修正，不机械保留错误措辞。
不要假装读取了只给出本地路径、没有实际上传的文件。

## 2. 已有证据与尚缺证据

| 研究问题 | 当前状态 | 写作边界 |
|---|---|---|
| strong CLE directional shortcut | 已有paired/operator DSA及gamma0对照 | 受控class--corruption绑定，不声称所有现实corruption都如此 |
| local-first归因 | 三training-seed已完成，12轮×16 local batches | 三seed DSA机制判据复现；不称所有完整gate均通过，不解释为中介分解 |
| Axis I四目标 | ERM/CVaR-DRO/PEW+GroupDRO/PEW+BER，40轮×16 batches，三training-seed | 同一AsymHFL、map2；不是12轮screen；不可要求默认重跑 |
| cross-binding | map1/map2正式结果 | 不是两个private datasets |
| 跨通信 | FedDF-fidelity成对结果已完成 | DSA下降，效用gate失败；不是主评价协议FedDF领域表那一行 |
| Oracle random/family/operator | 已完成 | 诊断/粒度边界，真实环境不进入部署方法 |
| 主评价剂量响应 | gamma0/.5/.9，seed0，40轮×64 batches已完成 | 实测三点趋势，不声称任意gamma单调性定理 |
| 主评价AsymHFL方法两臂 | seed0，40轮×64 batches已完成 | pooled DSA降低51.21%，四客户端改善方向一致；效用有代价 |
| 主评价HFL领域表 | 完成5/8：Local/FedMD/FedProto/AsymHFL-ERM/AsymHFL+PEW+BER | FedDF/AugHFL/RAHFL三行Formal仍未完成，不填数、不称完整榜单 |
| CIFAR-100第二private task | 输入/代码审计完成，benchmark/Formal未跑 | 不能写成跨数据集复现已完成 |
| bounded taxonomy stress | 最终协议未冻结，未运行 | 不能把公共PEW层旧LOO审计冒充端到端stress |

## 3. 核心数值与统计单位

- Local-first：Delta_HFL=`.119889/.123717/.124638`；Delta_Local=`.107960/.127287/.131210`；
  Delta_Comm=`+.011929/-.003570/-.006573`。
- 相应mean±sample SD：HFL `.122748±.002518`；Local `.122152±.012446`；Comm `.000596±.009929`。
- 主评价ERM DSA：gamma0 `-.0007302367`；gamma.5 `.0396793816`；gamma.9 `.1845142617`。
- 主评价BER DSA `.0900242753`；pooled相对下降51.21%，不是“四客户端各下降51.21%”。
  原Abstract里“reduces DSA by 51.21% on all four clients”可导致这层误读，应列科学措辞精度问题。
- BER−ERM的clean/mixed accuracy变化分别约`−4.350/−1.793` pp；不可只保留正效用。
- FedDF-fidelity DSA `.136989→.029012`，效用gate失败；不能写成无损插件验证。
- 图4误差棒为三个training-seed的样本SD（ddof=1），不是CI。
- source-bootstrap仅覆盖给定checkpoint下的评价source变化，不覆盖训练、partition或dataset随机性。
- local-first seed2完整内部verdict保留undertrained；不能声称三seed完整gate全部GO。

## 4. 必须保留的科学边界

- 论文是CLE-HFL问题—诊断—归因—缓解研究；PEW+BER是taxonomy-assisted本地方法模块。
- PEW为粗粒度family级；operator-level指受控构造/paired DSA评价，不是训练operator PEW。
- 图2应理解为任务loss与PEW伪分组汇合为BER risk，再梯度更新模型；BER不是图片预处理。
- 通信协议不因安装BER而重设计，但不保证任意HFL算法上无损兼容。
- 不声称首次研究FL spurious correlation、环境发现、重加权或counterfactual evaluation。
- PEW公共合成监督使用预定义taxonomy；私有真实corruption metadata不进入部署训练。
- BER有效分布性质成立于其定义和前提下；真实环境传递界当前数值平凡，不改成无条件性能保证。
- CVaR-DRO是本项目协议匹配的tail-risk对照，不宣称逐行复现某一篇完整算法。
- 正文使用mechanism protocol/main evaluation protocol；历史文件名strong/moderate保留。
- 不修改公式含义、符号约定、citation keys、label/ref、实验数字及pending状态。

## 5. 审“AI味”的可操作标准

请评估读者能否准确理解对象、证据和推论，不判断作者身份或AI生成概率：

1. 抽象词替代具体对象：例如strong/robust/comprehensive没有说明强在哪里、哪个数据支持。
2. 主张比证据大：把pooled改成all-client、受控改成普遍、均值小改成通信完全无影响。
3. 段落功能重复：Abstract/Introduction/Discussion反复复述同一限制、贡献清单或流程。
4. 模板过渡与连续防御：频繁“this does not imply/it is not/we do not claim”，打断正向解释。
5. 句子信息过载：一句同时解释问题、定理、方法、结果、边界，术语密度掩盖逻辑。
6. 无证据的因果词：therefore/proves/ensures跨越条件或实验范围。
7. 人为改词导致术语漂移：不要为避免重复把DSA、environment、operator等随意换同义词。

必要限制可以集中说明或移位，不能删除；专有术语、精确限定和正确的平行结构本身不构成问题。
自然学术表达不等于口语化、故意写错、堆少见词或模仿另一论文句子。

## 6. 本轮不做的事

不扩方法，不运行OpenI，不重做所有实验，不强行填满领域表，不把公式数量作为创新评分。
引用审计沿用2026-10-14版本；若需更新文献，须另列原始论文/官方页面和核验日期。
