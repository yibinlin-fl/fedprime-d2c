# CLE-HFL English Draft V0.2 事实审计与修订记录

Updated: 2026-09-14

## 来源

用户提供的网页端初稿：

```text
D:\googleDownload\CLE_HFL_PAPER_DRAFT_V0_1.md
size: 72472 bytes
SHA256: 7E88DE3A47E742B42D5A744C3F4E087143445BC13DCB2B15245BD9D876199873
```

仓库内事实审计版：

```text
deliverables/cle_hfl_paper_draft_v0_2_20260914/CLE_HFL_PAPER_DRAFT_V0_2.md
```

原下载文件未修改。

## 最重要的粒度澄清

最终论文必须稳定地区分三个对象：

```text
CLE-v2 problem/data generation: client- and class-specific concrete operator binding
PEW+BER train-time method: coarse family-level pseudo-environment grouping
DSA sealed evaluation: operator-level paired directional response
```

最终PEW已经冻结为family级分类器，没有operator-level PEW实现，也没有继续开发它的计划。
`Oracle operator BER`只是不读取部署约束的粒度边界消融，用于回答“更细真实分组最多还能带来多少”；
它不是候选方法。当前Oracle family与Oracle operator的DSA差仅`0.001449`，不足以支持开发
operator-level PEW，但也不能推出operator级环境建模在所有问题中无用。

## V0.2已完成修订

1. 将标题状态改为`Fact-Audited English Long Draft`，明确最终方法只包含coarse PEW+BER。
2. Problem Setup不再用CLE-v1四family公式作为当前正式定义；改为CLE-v2的client/class-specific
   concrete-operator binding `b_k: C -> O`，并把v1降为历史先导。
3. 在Problem Setup和PEW小节同时写清operator-level problem/evaluation与family-level
   train-time proxy的区别。
4. 明确Oracle operator是non-deployable boundary audit，不是operator PEW。
5. 从实现补入WCCA与CFG正式定义；它们不再属于`EVIDENCE NEEDED`。
6. 实验章节按RQ顺序调整为：主缓解、held-out map2 matched controls、cross-map、FedDF、Oracle。
7. 增加跨协议比较警告：original-map/map1/map2/FedDF只能解释各自matched contrast，不能用绝对
   DSA互相排名。
8. 删除与主线关系弱且尚无精确引用的泛化corruption方法枚举。

## 已核对且保持不变

- DSA全称为`Directional Shortcut Alignment`；
- CDep没有进入最终方法或当前草稿；
- HFL CLE effect `0.119889`、Local effect `0.107960`、communication add-on `0.011929`；
- original-map、map1、FedDF-fidelity与held-out map2正式数字；
- map2中BER只对CVaR保持pooled优势，c0/c1 DSA仍由CVaR更低；
- FedDF只支持shortcut-suppression跨底座迁移，整体utility gate仍为NO-GO；
- JTT仅有12轮screen，草稿没有宣称正式战胜JTT；
- source bootstrap不覆盖training-seed、partition或dataset uncertainty；
- PEW为taxonomy-assisted，现实部署有效性尚未建立。

## 下一版仍需处理

1. 逐篇精读并消除`[REF TO VERIFY]`，生成正式BibTeX；
2. 设计会议正文压缩版，将证明、实现超参数和审计细节移动到附录；
3. 等纯PEW+BER multi-training-seed和第二private dataset协议冻结后，再决定实验占位符；
4. bounded taxonomy stress test保持“最好补”而非自动授权实验；
5. 最终Abstract不得保留工作态的`[EVIDENCE NEEDED]`占位；
6. 在所有跨设置汇总表下注明仅比较within-setting matched deltas。

