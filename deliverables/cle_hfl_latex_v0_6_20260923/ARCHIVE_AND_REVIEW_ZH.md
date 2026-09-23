# CLE-HFL LaTeX V0.6-SKILL 归档与独立审查

日期：2026-09-23

## 1. 归档清单

本目录保留用户从网页端返回的不可变版本，不覆盖V0.5：

```text
CLE_HFL_LATEX_V0_6_SKILL_OVERLEAF_20260923.zip
CLE_HFL_LATEX_V0_6_SKILL_OVERLEAF_20260923.pdf
source/                         ZIP原样解压后的LaTeX源码与网页端审计材料
```

完整性信息：

```text
ZIP bytes   = 96578
ZIP SHA256  = F441A814B88E921FE36DE632D99E9DBCA16D9A1063B43FDD5B5533570C8B011B
PDF bytes   = 497678
PDF SHA256  = 0E3F91A7B9196AA32FF50DC6BBC61B3D81CC19DA374F8F53779F008DAF6E6896
PDF pages   = 13
PDF size    = US Letter
```

## 2. 总体判断

V0.6相对V0.5是明显的写作升级，可以作为后续修订底稿，但目前只能定位为：

```text
更成熟的conference-style internal manuscript
而不是submission-ready或“最终完美会议稿”
```

它的核心进步是把论文从实验槽位骨架整理成一条可阅读的科学叙事：

```text
CLE-HFL具体问题
-> source-paired DSA诊断与识别边界
-> seed-0 local-first形成证据
-> taxonomy-assisted PEW+BER局部干预
-> Axis I matched shortcut-objective比较
-> cross-map / second communication-base验证
-> Axis II HFL机制家族背景
```

Abstract、Introduction、Related Work、Experiments、Discussion和Conclusion的重复防御语句明显
减少；四项贡献边界更清楚；map2三training-seed Formal已经进入摘要、引言、主表、实验、讨论和
结论；FedDF utility失败、CVaR per-client例外、source-bootstrap边界和taxonomy-assisted定位均被
保留。相较V0.5，它更像一篇连贯的会议论文，而不只是可编译的实验计划书。

## 3. 必须修复的事实状态回退

网页端把包内`audit/authority/CLE_HFL_FULL_PAPER_WEB_HANDOFF_ZH.md`（2026-09-11）设为最高权威，
但V0.5及当前仓库状态已经更新到2026-09-22/23。因此V0.6错误地把两项“协议已冻结且工程已实现、
结果未运行”的实验降回“数据集/协议pending”：

1. **M3第二private dataset**：当前已经冻结为CIFAR-100 private task，CIFAR-10 public carriers，
   四臂、40轮、training seeds 0/1/2；代码、输入包、审计和OpenI入口已完成。正确状态应为
   `protocol implemented / benchmark and Formal pending`，而不是`dataset/protocol pending`。
2. **S1 bounded taxonomy stress**：当前已冻结为PEW public training排除`motion_blur`、private
   CLE/evaluation保留`motion_blur`的有界stress；runner与入口已完成。正确状态应为
   `protocol implemented / benchmark and Formal pending`，而不是`protocol pending`。

这两项仍不能填写结果，但论文内部状态必须与当前项目一致。S2写成Formal指标pending是正确的；
2026-09-23的loader pairing修复属于工程状态，不应被写成科学结果。

## 4. 网页端自带审计的两个不一致

1. 网页端事实审计声称LaTeX labels为`15 -> 15`且没有删除；独立扫描显示V0.5有15个label，
   V0.6交付源码有13个。删除的是未再引用的legacy表label：`tab:auto2`和`tab:cross-setting`。
   这不会造成未解析引用，但说明包内`static_fact_audit.json`并非对最终交付目录的精确复核。
2. README称设置`\draftskeletonfalse`即可查看evidence-only版本。该开关能隐藏部分pending表/行，
   但正文中的`\evidenceneeded{...}`并未由该条件控制，不能保证一键得到无红色占位符的投稿稿。

独立静态检查结果：20个citation keys均能在BibTeX解析；13个现存labels无重复；所有当前ref均有
目标；所有`\input`文件存在。源码仍含13处`\evidenceneeded`和58处`\pendingcell`调用。

## 5. 仍不像最终会议稿的内容

### 5.1 项目内部术语仍进入正文

正文仍出现`Formal`、`M1/M2/M3/S1/S2-v2/O1`、`GO_MAP2_TRAINING_SEED_STABILITY`、
`frozen utility gate`等项目管理语言。它们适合内部稿，不适合最终匿名会议论文。最终稿应分别改为
`full evaluation`、具体实验名称、客观门槛描述和普通统计结论。

### 5.2 “最佳trade-off”需要操作化

摘要和引言中的`higher pooled task utility`、讨论中的`best pooled shortcut--utility trade-off`没有
定义单一utility或trade-off标量。当前证据实际是BER在operator-grid accuracy、Avg、Worst、WCCA
更高且CFG更低，同时CVaR在c0/c1 DSA更低。最终稿宜直接陈述这些预注册指标，或明确定义trade-off，
避免让审稿人认为进行了事后综合排序。

### 5.3 证据仍未闭合

当前仍缺M2 local-first training seeds 1/2、M3第二private task结果、S1 bounded taxonomy stress、
S2十臂HFL context Formal。V0.6正确保留了这些缺口，但在它们完成前不能称“完备论文”。

## 6. PDF视觉检查

13页PDF逐页渲染检查未发现文字裁切、重叠、黑块、缺页或未解析引用；两张机制图清晰，正文层级、
公式和表格总体稳定。仍有以下提交前问题：

- 当前是venue-neutral `article`，不是目标会议官方模板，页数不能据此判断合规；
- PDF为internal skeleton，红色pending内容占据第7、8、11--13页；
- 第13页几乎只有一张pending computation table，留白很大；
- Axis II十臂表字体较密，两张机制图中的小字号需在官方模板下重新检查打印可读性；
- Poppler渲染报告Symbol/ArialUnicode display-font提示，但实际页面未见缺字；换官方模板后仍应复检。

## 7. 推荐版本策略

不要覆盖V0.6。本版本作为网页端Skill流水线的不可变快照保留。下一版建议命名V0.6.1或V0.7，
只先完成一次“事实状态同步”，不要再次大规模Humanizer：

1. 用2026-09-23仓库handoff纠正M3/S1/S2工程状态；
2. 保留所有未完成结果为pending；
3. 清理项目内部代号和`Formal/GO`语言；
4. 将模糊的pooled utility/trade-off改成精确指标表述；
5. 等关键实验返回后再生成无红色占位符的clean conference manuscript；
6. 目标会议确定后切换官方LaTeX class并重新做逐页视觉审计。

最终判定：V0.6在文章结构、节奏和可读性上显著优于V0.5；它已是一篇“像会议论文的完整内部
初稿”，但由于事实状态回退、内部术语、pending证据和非官方模板，尚不是完备或可投稿终稿。
