# 新会话恢复提示词

请先读取仓库根目录 `AGENTS.md`，然后严格按默认顺序只读取：

1. `docs/handoffs/latest.md`
2. `docs/README_ZH.md`

先不要读取大型历史日志，也不要递归扫描仓库。只有当前问题确实需要精确历史证据时，才按
`docs/README_ZH.md` 定位并局部读取 `docs/project/` 或归档材料。读取后先用不超过 12 条要点
复述你理解的当前状态，等我确认；此时不修改代码、不运行实验、不提交、不推送。

必须恢复的核心事实：

- 当前研究主线仍是 CLE-HFL：模型异构联邦学习中，类别与数据损坏形成虚假相关，模型可能
  利用 corruption 猜类别，而不是真正学习 corruption-invariant semantics。
- Phase-A0 已以 paired counterfactual 和 DSA 证明 strong CLE directional shortcut 存在；
  CLE 场景没有被最新实验否定。
- HFL-vs-Local 归因显示该机制主要是 local-first，通信放大/坏教师不是主故事。
- PEW+BER 在固定 CLE 场景上是强 taxonomy-assisted baseline，但依赖人工 corruption
  taxonomy，不能作为当前理想论文核心。
- K0-B taxonomy-free generic probe detector 曾正式 GO，但永久定位为离线审计，不进入训练
  循环；它不能被误写成已经可用的训练方法。
- SDMN、CDR-SNR、CRSF/K1-C-Minimal、P2/P3/P4 targeting 和 CVRS 等后续方法/因果路线均有
  已封存的负结果，禁止通过改阈值、调权重、补 seed 或换说法直接复活。
- 最新结果是 CVRS M0 seed-0 Formal：ResNet10 全部通过；MobileNetV2 上 CVRS 相对 baseline
  将 DSA 降低约 24.97%，并提高 Avg/Worst，但普通 Public-JSD 的 DSA 为 0.116098，优于
  CVRS 的 0.129098。`DSA_JSD-DSA_CVRS=-0.013000`，未达到冻结的 `>=0.02` 门槛。
- 最新正式 verdict：`NO_GO_CVRS`，`full_hfl_training_authorized=false`。这是否定 CVRS 的
  独特跨架构优势，不是否定 CLE 场景。
- 当前没有活动方法、训练任务或 OpenI 实验。下一步先做研究决策：新候选必须解释什么可观测
  信息能够识别真正有害的 pairwise class-corruption routing，并与 Public-JSD、PEW/BER、
  CRSF、CVRS 清楚区分。

当前任务：基于以上冻结证据，和我讨论下一步研究方向。不要马上写代码。任何候选先给出数学
对象、最弱成立假设、与已有方法和项目负结果的区别、理论链条、最小低成本 Kill Test、实验
归因风险和论文价值；我确认后才允许实现。

如果以后我同意实现并需要在 OpenI 跑，完成代码和本地 smoke 后必须主动给我一张完整启动卡，
一次性写清：

```text
1. Git commit，以及是否已经 push；
2. 复用旧数据集还是必须新建/上传；
3. OpenI 数据集显示名称；
4. 要上传文件的 Windows 绝对路径、精确字节数和 SHA256；
5. 启动文件；
6. 每一个运行参数名和准确值，布尔参数也必须给值；
7. GPU 型号/数量，以及这是 smoke、benchmark 还是 formal；
8. 预计耗时和算力成本；长任务必须先 benchmark，再由我确认；
9. 预期生成并需要下载的结果文件名；
10. 下载后应放入的 Windows 绝对文件夹；
11. 需要关注的指标、冻结门槛，以及实验能证明和不能证明什么；
12. 如果存在相似旧包，明确指出唯一有效包和禁止使用的旧包。
```

不要让我再追问“上传哪个文件、参数怎么填、结果放哪里”。Smoke/benchmark 不能当科学证据；
未经我明确同意，不启动付费、长时、Formal、多种子或完整 HFL 实验。
