# FedPRIME-D2C Session Handoff

Updated: 2026-09-17

## 当前主线与暂停点（用户于2026-09-17再次确认）

当前研究主线不是组会PPT。近期FedPIN英文汇报PPT仅为一次性旁支任务；未来上下文压缩时只需保留
“用户曾准备FedPIN组会汇报”这一句，不保留多轮PPT提示词、页数设计或生成过程。

论文线目前主动暂停在LaTeX V0.3：用户将先学习LaTeX，并完整阅读V0.3以理解CLE-HFL的论文
结构和证据闭环。完成这两项后，主线恢复为投稿前实验，不继续无休止润色，也不新增方法。实验恢复
顺序仍以冻结矩阵为准：M1 held-out map2四臂40轮training seeds 1/2，随后M2 HFL-vs-Local
seeds 1/2，再设计与审计M3第二private dataset。任何benchmark、Formal或长任务仍需单独授权。

## LaTeX V0.2已终检，V0.3完成两处纯排版修复

用户回传的V0.2 Overleaf PDF共11页，逐页视觉检查确认无缺页、裁切、重叠、黑块或未解析引用；
V0.1中的Table 3过小、Oracle表漂移到References之后和占位caption问题已修复。终检仅发现两处
剩余排版问题：第4页JSD/DSA逻辑式跨越双栏并挤入页脚，第9页Oracle五列表字体仍偏小。V0.3
已将逻辑式拆为两行，并将Oracle表缩为三列、把last-five数字移入紧邻正文。所有实验小数、18个
citation key和科学边界与V0.2保持一致。V0.3当前等待Overleaf编译确认，没有启动训练实验。

```text
deliverables/cle_hfl_latex_v0_2_20260916/CLE_HFL_LATEX_V0_2_OVERLEAF_COMPILED.pdf
deliverables/cle_hfl_latex_v0_3_20260916/main.tex
deliverables/cle_hfl_latex_v0_3_20260916/V0_2_COMPILE_AUDIT_ZH.md
deliverables/CLE_HFL_LATEX_V0_3_OVERLEAF_20260916.zip
local_runs/paper_compiled_returns/v0_3/
```

V0.3 Overleaf包为`28746` bytes，SHA256
`83CADA48B868E34374B417D9BCECEA59702DD0AC9A1A2AB798E4EEB4FA979443`。

## LaTeX V0.2已按网页端第一轮审稿完成证据受控修订

用户已在Overleaf成功编译LaTeX V0.1，11页PDF逐页检查无裁切、重叠、黑块或缺失引用；视觉审计
发现Table 3字体过小、Oracle Table 4漂移到References之后及占位caption。V0.1源码和编译PDF已
归档保留。网页端第一轮审稿原文及裁决已归档；V0.2接受seed-0限定、local-first措辞降级、精确
CLE sampling law、pooled DSA公式、estimand decomposition解释、BER直观解释、图1弱化及表格
修复，拒绝提前指定第二数据集、删除待补标记或扩展新方法。V0.2现作为已编译历史版本保留；
当前source of truth已晋级为仅含排版修复的V0.3。没有启动任何训练实验。

```text
deliverables/cle_hfl_latex_v0_1_20260915/CLE_HFL_LATEX_V0_1_OVERLEAF_COMPILED.pdf
deliverables/cle_hfl_latex_v0_2_20260916/main.tex
deliverables/cle_hfl_latex_v0_2_20260916/WEB_REVIEW_ADJUDICATION_ZH.md
deliverables/CLE_HFL_LATEX_V0_2_OVERLEAF_20260916.zip
local_runs/paper_compiled_returns/v0_2/
```

V0.2 Overleaf包完整性：`28806` bytes，SHA256
`F81084F4A3AAEFE274A9E26334B62DED47BF8791CAEFF9C9F0BD29FDD930DD5A`。包内仅含
`main.tex`、`references.bib`、`README_ZH.md`及两张图的PDF/SVG，不含网页端原始审稿文本。

## 论文版本与源码职责已明确

网页端英文V0.1、事实审计V0.2、引用审计V0.3、会议压缩V0.4均已独立保留；早期AAAI PDF也作为
legacy快照归档。当前论文事实内容以V0.4为基线，当前可编译源码已晋级为仓库中的LaTeX V0.2
`main.tex`。Overleaf只用于编译与预览；若在Overleaf手工编辑，必须下载source
ZIP交回仓库做diff和合并，禁止仓库与Overleaf同时修改形成双源。

```text
deliverables/cle_hfl_paper_draft_v0_1_20260914/CLE_HFL_PAPER_DRAFT_V0_1.md
deliverables/cle_hfl_paper_draft_v0_2_20260914/CLE_HFL_PAPER_DRAFT_V0_2.md
deliverables/cle_hfl_paper_draft_v0_3_20260914/CLE_HFL_PAPER_DRAFT_V0_3.md
deliverables/cle_hfl_paper_draft_v0_4_20260915/CLE_HFL_PAPER_DRAFT_V0_4.md
deliverables/cle_hfl_latex_v0_1_20260915/main.tex       历史排版快照
deliverables/cle_hfl_latex_v0_2_20260916/main.tex       已编译历史版本
deliverables/cle_hfl_latex_v0_3_20260916/main.tex       当前source of truth
```

## 投稿版完整实验矩阵已整理

当前论文实验按优先级整理为单一规划入口：M1 held-out map2四臂40轮training seeds 1/2、
M2 HFL-vs-Local四臂training seeds 1/2、M3第二private dataset为投稿前必须补；bounded taxonomy
stress与精简faithful HFL context table为强烈建议；JTT Formal、partition seed及更多插件底座为
条件性可选。CDep、operator/hierarchical PEW及新loss明确不做。该规划不构成实验授权，当前仍
没有运行中的训练任务。

```text
docs/experiments/current/CLE_HFL_SUBMISSION_EXPERIMENT_MATRIX_2026_09_15_ZH.md
```

## V0.4已转换为LaTeX会议稿并生成两张论文图

当前V0.4已机械转换为venue-neutral双栏LaTeX，保留18个citation key、7处内部
`[EVIDENCE NEEDED]`标记、四张表及全部冻结边界。两张新矢量图分别展示CLE-HFL证据链/信息
边界，以及paired DSA/proxy non-identifiability。PDF与SVG均已生成，PDF已转PNG逐图检查，
未发现文本重叠、裁切或不可读元素。

```text
deliverables/cle_hfl_latex_v0_1_20260915/main.tex
deliverables/cle_hfl_latex_v0_1_20260915/references.bib
deliverables/cle_hfl_latex_v0_1_20260915/figures/
deliverables/cle_hfl_latex_v0_1_20260915/README_ZH.md
deliverables/CLE_HFL_LATEX_V0_1_OVERLEAF_20260915.zip
scripts/build_cle_hfl_latex.py
scripts/render_cle_hfl_paper_figures.py
```

当前主机未安装`pdflatex/xelatex/latexmk`，故完整论文尚未执行TeX编译；Overleaf包可直接上传，
目标会议确定后再替换官方class/style并进行逐页PDF检查。没有启动或授权任何训练实验。

## 英文会议压缩稿V0.4已完成事实审计

网页端V0.4已审计并提升为当前会议压缩内部主稿，约5165词；V0.3继续保留为完整证据长稿。
V0.4保持operator-level CLE-v2/DSA、family-level PEW+BER、Oracle operator仅边界消融、
CDep缺席、FedDF utility失败、CVaR逐客户端例外、JTT screen-only和source-bootstrap条件性等
冻结边界。关键数字全部可定位回现有冻结证据，18个真实引用key均能解析。

```text
deliverables/cle_hfl_paper_draft_v0_4_20260915/CLE_HFL_PAPER_DRAFT_V0_4.md
deliverables/cle_hfl_paper_draft_v0_4_20260915/CLE_HFL_V0_4_AUTHOR_AUDIT.md
deliverables/cle_hfl_paper_draft_v0_4_20260915/FACT_AUDIT_ZH.md
deliverables/cle_hfl_paper_draft_v0_4_20260915/references.bib
```

当前待补实验仍全部未授权、未运行。优先顺序是：先冻结map2四臂40轮S1/S2协议和成本；再决定
HFL-vs-Local S1/S2；第二private dataset及bounded taxonomy stress需另行设计，未冻结为SVHN
或leave-one-operator-out。新增协作规则：以后指定用户放置文件前，先创建目标文件夹并提供绝对路径。

## 英文长初稿V0.3完成引用审计

网页端V0.1先修订为事实审计版V0.2，现已形成引用审计版V0.3。最终粒度继续冻结为：CLE-v2
数据生成和DSA评价使用concrete operator；PEW+BER训练只使用coarse family伪环境；Oracle
operator仅为不可部署边界消融，不是候选PEW。V0.3已将`[REF TO VERIFY]`清零，正文18个
citation key均有对应BibTeX；独立审计记录正式发表、workshop、仅预印本和实现忠实度边界。
FedCD仍只能写成arXiv 2024；FedDF-fidelity和CVaR-DRO均不得写成官方算法逐行复现。
CDep仍未进入草稿，没有运行或授权新实验。

```text
deliverables/cle_hfl_paper_draft_v0_2_20260914/CLE_HFL_PAPER_DRAFT_V0_2.md
deliverables/cle_hfl_paper_draft_v0_2_20260914/REVISION_NOTES_ZH.md
deliverables/cle_hfl_paper_draft_v0_3_20260914/CLE_HFL_PAPER_DRAFT_V0_3.md
deliverables/cle_hfl_paper_draft_v0_3_20260914/references.bib
deliverables/cle_hfl_paper_draft_v0_3_20260914/CITATION_AUDIT_ZH.md
```

## 相关工作发表格局已形成导师与网页端双用途材料

已核验并整理CLE-HFL最近邻工作的正式发表状态、时间线、研究重合和不可冒领边界。RAHFL明确为
IEEE TPAMI 2025正式论文及ICCV 2023 AugHFL扩展；FedPIN为ICML 2024；个性化FL spurious
features有NeurIPS 2021 workshop早期版和TMLR 2024正式版；FedCD截至本次只核验到arXiv 2024。
文档同时正面纳入GroupDRO、GEORGE、EIIL、JTT和MIDL counterfactual诊断先例。安全定位仍是
`具体CLE-HFL问题 + paired DSA理论 + local-first归因 + taxonomy-assisted PEW/BER + 受控边界`，
禁止声称首次联邦spurious learning、首次伪环境发现或首次反事实shortcut评价。

```text
docs/research/status/CLE_HFL_RELATED_WORK_PUBLICATION_LANDSCAPE_2026_09_14_ZH.md
deliverables/cle_hfl_full_paper_web_handoff_20260911/CLE_HFL_FULL_PAPER_WEB_HANDOFF_ZH.md
```

## 论文结构、模拟审稿与投稿观察表已更新

网页端初稿交接文档已扩展为完整投稿结构：Introduction五段、Related Work四组、RQ1--RQ6
实验组织、主表/机制图、limitations及16类模拟审稿攻击—回答—待补证据。论文主叙事固定为
`CLE-HFL问题 + paired DSA理论 + local-first归因 + PEW/BER结构性干预 + 受控复现与边界`，
不以“新插件结构”作为唯一创新。

截至2026-09-13已核实候选：AISTATS 2027（CCF-C）摘要`2026-09-29`、全文`2026-10-06`
AoE；IJCNN 2027（CCF-C）regular paper `2027-01-31`；ECAI 2027（CCF-B）full paper
`2027-04-14`。AISTATS仅为激进冲刺候选，设置`2026-09-20`内部决策点；若完整英文骨架、
相关工作核验和补实验清单尚未就绪，停止赶该截稿。当前没有因此授权新的付费或Formal实验。

```text
deliverables/cle_hfl_full_paper_web_handoff_20260911/CLE_HFL_FULL_PAPER_WEB_HANDOFF_ZH.md
docs/project/CURRENT_PROJECT_MEMORY.md
```

## Proxy不可识别性与DSA必要性理论闭环完成

BER到真实CLE行为之间的唯一显式理论缺口已按诚实边界闭合。新定理证明：在不约束
`P(E_hat|E,Y)`时，相同`P(Y,E_hat)`可对应真实`Y-E`依赖为0或强依赖的两个潜在世界；因此
proxy下降不能无条件证明真实CLE下降。现有PEW误差TV界被保留为条件性桥，WEC-BER的误差
通道迁移失败被纳入前提边界，禁止用混淆矩阵反演换名复活。

CPU零训练验证读取冻结Stage-1概率缓存和CVRS Formal `result.json`：同一可观测proxy构造的
两个潜在世界真实TV为`0/0.8`；identical-view `JSD=0`但cached `DSA=0.119644`；MobileNetV2
上CVRS相对Public-JSD的proxy下降`0.088854`，真实DSA反升`0.013000`。N0--N4全部PASS，
10项聚焦测试全部通过。由此得到限定推论：在本文CLE证据协议中，方法若只优化proxy目标，必须
额外报告paired DSA或等价target-aligned estimand；不声称DSA是所有shortcut问题唯一指标。

```text
docs/research/status/CLE_PROXY_NONIDENTIFIABILITY_THEORY_2026_09_13_ZH.md
scripts/validate_proxy_nonidentifiability.py
deliverables/cle_proxy_nonidentifiability_20260913/RESULT_SUMMARY_ZH.md
deliverables/cle_proxy_nonidentifiability_20260913/PROXY_NONIDENTIFIABILITY_THEORY.png
```

这完成的是理论逻辑闭环，不是BER无条件性能定理，也不替代training-seed和第二数据集证据。

## Held-out map2 40轮四臂Formal全部冻结门槛通过

原binding map五臂12轮screen已晋级`ERM/CVaR-DRO/PEW+GroupDRO/PEW+BER`，JTT不进入
最终表。新协议在未参与筛选的binding map2上固定40轮、16 local batches、train seed0和strict
AsymHFL-val，禁用AugMix/JSD/DCL/CDep。结果前已冻结shortcut、BER-vs-ERM、BER-vs-
GroupDRO及BER-vs-CVaR准确率—DSA折中门槛。8项单元测试和真实CUDA一轮四臂训练/配对
分析通过；smoke无科学意义。提交`5ea5861`已push至`origin/main`。OpenI Formal完成，输入审计、
四臂各40行metrics、各160条完全匹配local traces及独立预测缓存复算全部通过：

```text
                 DSA       grid acc   last10 Avg  last10 Worst  WCCA   CFG
ERM              0.260308  20.3067    19.8578     16.3007       0.000  32.8325
CVaR-DRO         0.088982  20.4567    18.6700     14.0380       0.125  27.7775
PEW+GroupDRO     0.214575  20.7883    20.9562     17.8140       0.025  31.5275
PEW+BER          0.077741  23.7433    24.4518     20.4727       2.650  22.8025
```

`ERM-BER=0.182567` CI95 `[0.180596,0.184407]`；`GroupDRO-BER=0.136834` CI95
`[0.134952,0.138706]`；`BER-CVaR=-0.011240` CI95 `[-0.012355,-0.010092]`。I0/S0/P1--P5
全部PASS，verdict=`GO_FOUR_ARM_HELDOUT_MAP2`。BER相对ERM/GroupDRO在4/4客户端降低DSA；
相对CVaR只支持pooled优势，c0/c1 DSA仍由CVaR更低。BER grid accuracy在4/4客户端均最高。

```text
scripts/openi_cle_v2_spurious_final_entry.py
docs/experiments/current/CLE_V2_SPURIOUS_FINAL_MAP2_ZH.md
```

正式报告：`deliverables/cle_v2_spurious_final_map2_20260913/RESULT_SUMMARY_ZH.md`。下一步不扩展
新方法；先精读最近shortcut文献，再冻结纯BER training-seed稳定性和第二数据集的最小协议。

## 最终基线审计与spurious-correlation五臂Screen已完成

九种HFL基线已完成代码/协议分级：Local/ERM、FedMD、RHFL、FedProto、AugHFL、FedDF、
KT-pFL、FCCL、RAHFL。最终表必须使用`aughfl_fidelity/feddf_fidelity/kt_pfl_fidelity`，
早期同名适配器12轮结果只作历史筛选；FCCL、FedMD、RHFL、FedProto仍应标为protocol-matched
core adapter。12轮肯定不足以作为最终主表，只能筛选晋级者。

新增五臂保持相同strict AsymHFL-val通信、初始化、single-view输入和fit/audit/test角色：

```text
ERM / JTT / CVaR-DRO / PEW+GroupDRO / PEW+BER
```

JTT和CVaR完全不读取PEW；JTT第一阶段只从fit错误构造冻结集合，第二阶段从共同initial states
重训。PEW+GroupDRO共享冻结PEW分组但不调用BER。全新目录真实CUDA smoke、五臂checkpoint、
4/4客户端完整JTT fit推理、20-source DSA及逐batch配对均通过；smoke数值无科学意义。
OpenI 12轮screen已经完成：CVaR取得最低DSA `0.034980`；PEW+BER DSA为`0.037889`，但取得
最高operator-grid accuracy `21.1267`、最高last-5 Avg `19.3430`和更低CFG。PEW+BER相对ERM
使DSA降低82.36%；相同PEW分组下也显著优于PEW+GroupDRO。该结果仅作选择，晋级四臂见本
handoff首节；若声称完整RAHFL性能，仍须明确处理共享40轮预训练与40轮通信预算。

```text
docs/research/baselines/CLE_HFL_FINAL_BASELINE_AUDIT_2026_09_12_ZH.md
docs/experiments/current/CLE_V2_SPURIOUS_BASELINE_SCREEN_ZH.md
```

## KT-pFL/FCCL独立插件扩展已完成本地smoke

KT-pFL与FCCL并非本轮新实现；仓库已有各自的通信策略。本轮没有修改
`fedprime/communication/baselines.py`，只增加独立四臂协议：

```text
kt_b = standard CE + kt_pfl_fidelity
kt_p = PEW hard-BER CE + identical kt_pfl_fidelity
fc_b = standard CE + FCCL
fc_p = PEW hard-BER CE + identical FCCL
```

四臂均禁用AugMix/JSD/DCL/CDep，并共享数据、初始化、私有batch轨迹和各自成对相同的通信预算。
22项回归测试、本地1-round真实CUDA smoke、四臂checkpoint与paired DSA分析全部通过；KT与FCCL
两组的Base/Plugin轨迹均完全匹配。smoke未形成可识别shortcut，只证明执行链路，不能作为论文
证据。OpenI benchmark与Formal均未授权；下一步先由用户确认协议和Formal门槛，若确认再跑
benchmark估算四臂成本，不能直接启动Formal。

```text
docs/experiments/current/CLE_V2_KT_FCCL_PEW_BER_PLUGIN_ZH.md
scripts/openi_cle_v2_kt_fccl_plugin_entry.py
```

## Cross-map1 Formal四门全过：跨binding-map复现GO

2026-09-12只改变客户端特定class-operator binding map、固定partition/evaluation/training seed、
初始权重、public数据、评价grid与同一冻结PEW的map1 Formal完成。纯两臂为`h9_b`与
`h9_b+PEW/BER`，CDep禁用；输入审计、48条paired local traces和预测缓存独立复算全部通过：

```text
DSA: 0.113761 -> 0.049531
reduction: 0.064230 (56.46%), CI95 [0.063105, 0.065328]
client reductions: [0.064195, 0.047682, 0.085614, 0.059428]
operator-grid pooled: 20.9450% -> 21.4583%
last-5 delta: Avg +0.7040, Worst +1.0760, WCCA +0.3500, CFG -9.8550
gates: I0/L0/C1/C2 all PASS
verdict: GO_PEW_BER_CROSS_MAP1
```

该结果允许主张shortcut形成与PEW+BER缓解不依赖唯一一张偶然binding map；它不覆盖新partition、
训练seed、corruption库、severity、数据集或真实场景。原Stage-2整体NO-GO仍保留，不得被本次
报告性效用指标覆盖。旧两臂cross-runner的map2/all未单独续跑；map2后来按本handoff首节的
四臂40轮held-out协议正式运行并GO。总耗时`8311.44 s`（约2.31 V100小时）。
原始包4,457,718 bytes，SHA256
`8DA64B4E6669CE7534ADEA023E54EFAEB51353073EE85261E0F1768F2BD688E1`。

```text
deliverables/cle_v2_cross_map1_formal_20260912/RESULT_SUMMARY_ZH.md
docs/experiments/current/CLE_V2_CROSS_SCENARIO_BINDING_MAP_ZH.md
```

## Cross-map结果已纳入论文主表、机制图和初稿证据链

论文核心材料已按“问题—诊断—归因—干预—受控验证”闭环生成。统一主表同时列出原AsymHFL map、
新binding map1及native-CE FedDF-fidelity三项Formal DSA结果，并保留各自效用变化与冻结overall
verdict。机制图将CLE形成、paired DSA、local-first、PEW/BER和两类复现轴连接起来，底部明确
taxonomy-assisted、效用非一致以及不能外推到新partition/seed/dataset/real domain的边界。

```text
deliverables/cle_hfl_paper_core_artifacts_20260912/
  PAPER_MAIN_TABLE_ZH.md
  PAPER_MAIN_TABLE.csv
  PAPER_MECHANISM_EVIDENCE_CHAIN.png
  PAPER_MECHANISM_EVIDENCE_CHAIN.pdf
  PAPER_MECHANISM_EVIDENCE_CHAIN.svg
  plot_paper_mechanism_figure.py
  README_ZH.md
```

网页端初稿交接文档与主表/机制图已加入40轮held-out map2四臂Formal；下一步优先做最新文献与
投稿证据缺口审计，不自动启动新方法扩展。

## BER机制理论与CPU Kill Test完成：失衡压缩PASS，PEW误差界仍平凡

当前hard BER已形式化为有效经验分布`Q_gamma`：类内环境质量正比于
`min(n_ce,32)^0.5`，因此无cap区域log支持优势压缩为原来的`0.5`倍；在共同支持且
`gamma=0`的理想条件下，`Y`与伪环境独立。生产PyTorch loss与有效分布公式的直接测试通过。

固定CLE-v2 `seed0_split0/gamma09` strict-fit的CPU零训练审计全部冻结门槛通过：

```text
code/theory identity max error: 7.32e-16
pseudo TV: 0.487505 -> 0.199744, relative -59.03%, 4/4 clients decrease
true-family TV: 0.634914 -> 0.433513, relative -31.72%, 4/4 clients decrease
true-family environment-only Bayes advantage: 0.313840 -> 0.240863, -23.25%
verdict: PASS
```

真实family只用于离线审计，不进入训练。BER有效分布下PEW family误差为`0.507--0.575`，使
`TV(Y,E)<=TV(Y,E_hat)+2 epsilon`上界截断为平凡`1.0`；不得宣称无条件真实环境去相关、DSA
必为零或准确率必提升。client2的真实TV也下降38.66%，故其任务损伤不能归因于“BER没有压缩
CLE分布”。下一平台任务仍是已准备的cross-map benchmark，不因本审计增加Formal授权。

```text
docs/research/status/CLE_BER_MECHANISM_THEORY_2026_09_11_ZH.md
deliverables/ber_mechanism_theory_20260911/
scripts/audit_ber_mechanism.py
```

## DSA识别理论v2与Cross-Binding-Map S2完成

DSA理论已从三个代数性质扩展为完整的识别对象与五项验证：paired contrast在
`m(z,o')=s(z)+r(z,o')`下消除operator-invariant语义基线，识别binding-specific operator
response；补充了受控binding注入恢复、operator-invariant位移消除、shuffled-binding
specificity及source-level推断。v2缓存/构造验证全部冻结门槛通过：已知强度恢复最大误差
`6.66e-16`，不变位移误差`8.33e-17`，真实binding `0.241071`高于null p95 `0.026786`
（`p=0.000999`）；`n=1000`的95%保守Hoeffding半径为`0.085894`。这不等于跨场景因果充分性。

Cross-binding-map协议已完成S0--S2。新map1/map2固定partition seed0、evaluation seed、training
seed0、初始模型、公共数据、评价source和同一冻结PEW checkpoint，只改变binding map；纯两臂为
`h9_b`与`h9_b+PEW/BER`，CDep禁用。两张map的一轮CUDA smoke、20-source分析、配对轨迹、输入
审计和20项测试均通过；smoke数值不是科学证据。唯一有效输入包：

```text
local_runs/cle_v2_cross_scenario/cle_hfl_v2_cross_maps1_2_seed0_split0_with_pew.tar.gz
bytes: 1385820059
SHA256: BEA8E98737BF881C701DCFFF05F4E04C3A1E6095B7CF7702A5177260C2F186F5
entry: scripts/openi_cle_v2_cross_scenario_entry.py
```

map1 Formal的科学结论见本handoff首节。旧`openi_cle_cross_scenario_40round_entry.py`及
seed1_split1/seed2_split2包同时改变partition、重训PEW并含CDep，禁止用于本协议。

```text
docs/research/status/CLE_DSA_IDENTIFICATION_THEORY_2026_09_11_ZH.md
docs/experiments/current/CLE_V2_CROSS_SCENARIO_BINDING_MAP_ZH.md
deliverables/cle_hfl_full_paper_web_handoff_20260911/CLE_HFL_FULL_PAPER_WEB_HANDOFF_ZH.md
```

## CLE-HFL网页端论文初稿交接包已完成

已将完整证据链整理为自包含网页端GPT交接文档：CLE-HFL场景、paired DSA、strong directional
shortcut、HFL-vs-Local local-first归因、DSA识别理论、BER机制理论、AsymHFL与FedDF-fidelity
上的PEW+BER结果、Oracle粒度边界及cross-binding-map Formal复现。文档同时给出统一证据表、
贡献边界、可写/不可写主张、建议论文结构、正式表图路径、投稿前缺口和网页端GPT任务提示。

```text
deliverables/cle_hfl_full_paper_web_handoff_20260911/
  CLE_HFL_FULL_PAPER_WEB_HANDOFF_ZH.md
```

文档以最新CLE-v2 Formal为主链，早期CLE-v1只作为先导场景说明；明确保留“shortcut mitigation
GO / architecture-uniform utility未建立 / frozen overall plugin NO-GO”的分层结论。该交接包
可用于生成初稿，但不等于已经确认无需补实验；下一步先让网页端GPT完成投稿证据缺口审计。

## Native FedDF-fidelity × coarse PEW+BER Formal完成：机制迁移，插件总门NO-GO

用户指出matched-robust设计不能证明真正插件性后，该设计及其smoke立即作废。现冻结第二底座
原生CE两臂：`fd_b=standard CE+FedDF-fidelity`，`fd_p=PEW-grouped hard BER-weighted CE+
相同FedDF-fidelity`。两臂共享同一CLE场景、初始化、single-view私有batch、公共预算和修复后的
post-local FedDF server distillation；AugMix/JSD/DCL/CDep全部禁用。唯一目标差异是PEW分组与
BER重加权。仍只称protocol-matched FedDF fidelity adapter，不声称官方完整recipe复现。

12轮×16 local batches/client/round Formal已完成，输入审计、配置哈希、48条配对local trace、
checkpoint预测及独立缓存复算均通过：

```text
DSA 0.136989 -> 0.029012, reduction 0.107977 (78.82%)
CI95 [0.106498, 0.109417], 4/4 clients positive -> P1 PASS
operator-grid 18.4917 -> 18.4517 -> L0 FAIL（两臂均低于20%）
last-5 delta: Avg -0.6660, Worst +0.2147, WCCA -0.1000, CFG -13.2350
I0 PASS, L0 FAIL, P1 PASS, P2 FAIL
verdict: NO_GO_PEW_BER_FEDDF_PLUGIN_SEED0
```

允许结论是PEW+BER的CLE shortcut抑制从AsymHFL迁移到native-CE FedDF-fidelity；不允许宣称
通用无损插件或架构一致效用。结果不是单调崩溃：operator-grid pooled仅下降0.04pp，但冻结
last-5 Avg下降0.666pp，且逐客户端效用不一致。不得改门槛、调权重或补seed翻案。

原始Formal包4,496,390 bytes，SHA256
`A6328074227F08A6C89BF68727C4725CEEE826EB9D08DB09626C74A241E20F38`，总耗时0.4012 V100
GPU-hours。报告：

```text
deliverables/cle_v2_feddf_plugin_formal_20260911/RESULT_SUMMARY_ZH.md
```

```text
docs/experiments/current/CLE_V2_FEDDF_PEW_BER_PLUGIN_ZH.md
scripts/openi_cle_v2_feddf_plugin_entry.py
docs/research/status/CLE_HFL_PAPER_CLOSURE_2026_09_11_ZH.md
```

## Oracle粒度Formal完成：真实分组有效，但operator细分无实质收益

固定CLE-v2 `seed0_split0`、gamma0.9、training seed 0、12 rounds的Oracle family/operator/
类别内随机operator三臂Formal已完成，输入审计、配置哈希和48条配对训练轨迹通过；独立缓存复算
与平台JSON完全一致。

```text
DSA: OF 0.017232, OO 0.015783, RO 0.067895
RO-OO 0.052112, CI95 [0.051231, 0.052983] -> G1 PASS
OF-OO 0.001449, CI95 [0.001086, 0.001790] -> G2 FAIL
grid Avg: OF 21.9333, OO 21.5200, RO 18.7817 -> L0 FAIL
last-5 Avg: OF 21.0960, OO 20.0273; zero-recall均13 -> G3窄幅FAIL
verdict: NO_GO_OPERATOR_GRANULARITY_GAP
hierarchical_pew_training_authorized: false
```

主结论由G2决定：operator相对family的绝对DSA改善仅`0.001449`，距离`0.02`门槛约13.8倍，
按客户端还出现一正一负的明显异质性。真实operator分组相对随机分组显著有效，说明环境对应关系
确实重要；但粗family已经捕获几乎全部可用粒度收益。不得实现层次化/operator PEW，或用调参、
补seed、改门槛复活。该结果不解决ShuffleNet/client2任务效用问题。

原始包6,633,253 bytes，SHA256
`B133F578255308764A1AB6ECD41ACB66421FA75A73A75AF7864BA16681D7054C`；总耗时3.4518 V100
GPU-hours。报告：

```text
deliverables/cle_v2_oracle_granularity_formal_20260911/RESULT_SUMMARY_ZH.md
```

## DSA理论缓存验证完成

DSA三个预注册命题已在Stage-1正式预测缓存上以CPU零训练方式验证：exchangeable projection
的DSA为`-2.50e-20`，gamma0经验最大绝对DSA为`0.001914`；HFL/Local的11点概率混合曲线
严格单调，最大仿射误差`2.78e-17`；三个完全相同预测视图的最大JSD为0，而h9 HFL DSA仍为
`0.119644`。全部冻结门槛通过，但只支持代数性质、经验零点和JSD不充分反例，不外推跨场景结论。

```text
docs/experiments/current/CLE_DSA_THEORY_CACHE_VALIDATION_ZH.md
deliverables/cle_dsa_theory_validation_20260911/
```

## Pure PEW+BER Stage-2 Formal完成：CLE抑制有效，原四门总判定NO-GO

固定`seed0_split0` gamma0.9、training seed 0、12 rounds、每客户端每轮16 batches的纯插件A/B
已完成：

```text
h9_b = AugMix/JSD/DCL + strict AsymHFL-val
h9_p = h9_b + frozen public PEW + hard BER
```

CDep未使用，48条local trace完全匹配。正式结果为：

```text
DSA 0.119644 -> 0.041252, reduction 0.078392 (65.52%)
source-bootstrap CI95 [0.077191, 0.079601], 4/4 clients positive
last-5 delta: Avg +1.7323, Worst -0.3760, WCCA +0.3500, CFG -9.2600
gates: I0 PASS, L0 PASS, P1 PASS, P2 FAIL
frozen verdict: NO_GO_PEW_BER_STAGE2_SEED0
```

P2仅因预注册`Worst>=+1.0`失败；WCCA通过。事后论文解释可记为`CLE mitigation efficacy: GO / architecture-uniform
utility: NOT ESTABLISHED`，但不得覆盖原Formal verdict。结果包4,454,630 bytes，SHA256为
`1719D2C3801986FCA7BAAEF9CC89AFCCD5D7818470C94AA67C4139FFC993CD79`。协议、结果与归因为：

```text
docs/experiments/current/CLE_V2_PEW_BER_STAGE2_OPENI_ZH.md
scripts/openi_cle_v2_plugin_stage2_entry.py
scripts/analyze_cle_v2_plugin_architecture_attribution.py
deliverables/cle_v2_plugin_stage2_architecture_attribution_20260910/
```

零训练归因发现client2/ShuffleNet的operator-grid accuracy下降5.1733，但不能归因为“小模型容量”，
因为架构与非IID客户端划分固定绑定。损失为类别选择性重分配：automobile/ship/truck召回降为0，
airplane/horse分别提升29.60/21.20。PEW误分不能单独解释（automobile的PEW family accuracy约80%仍
坍缩），全局BER权重过大也不能单独解释（client2最大样本乘数四客户端最低）。当前最准确定位是
`ShuffleNet/client2 pair上的类别决策重分配`。下一步先讨论一个架构×数据交叉的低成本Kill Test；
不得直接调BER权重、补seed或把相关性写成因果。

本地进程直接调用`D:\anaconda3\envs\pytorch\python.exe`会继承Codex优先PATH并可能加载冲突DLL；
已复现原生退出码`0xC06D007F`。改用`D:\anaconda3\Scripts\conda.exe run -n pytorch python`
后Matplotlib与RTX 3050 CUDA均通过。后续本地实验统一完整Conda启动。

## CLE-v2 Mechanism Stage-1 Formal：四门全过，正式 GO

2026-09-10 完成固定 `gamma0/gamma0.9 × HFL/Local` 四臂、training seed 0、12 rounds Formal。
输入审计 PASS，四臂均完成，gamma 内 HFL/Local 本地轨迹匹配；PEW/BER/CDep 均未使用。
结果包 SHA256 为 `A23312C97A1B9FE2A4DAA8341BB83CD326553A550726CE626097FF68AF4FF4B2`。

```text
pooled DSA: h0_b -0.000245, h9_b 0.119644, l0_b -0.001914, l9_b 0.106046
HFL CLE effect:   0.119889, source-bootstrap CI95 [0.118050, 0.121562]
Local CLE effect: 0.107960, source-bootstrap CI95 [0.106145, 0.109678]
communication add-on: 0.011929
Local/HFL share: 90.05%
h9 shuffled-binding: null p95 0.029559, p=0.000999
```

L0/M1/M2/M3 全部通过，verdict 为 `GO_CLE_V2_MECHANISM_STAGE1`；不触发32-batch补跑。
该结果证明固定 CLE-v2 场景中存在 binding-specific directional shortcut，且机制主要
local-first；通信仅在 pooled 平均上形成较小附加效应，客户端通信差值并非全为正。它不证明
PEW+BER 有效，也不证明跨训练 seed 或跨 CLE 场景成立。40,000 是可用私有数据集规模，不代表
每个模型完整遍历全部40,000个唯一样本。

论文机制表、图和独立复算报告位于：

```text
docs/experiments/current/CLE_V2_MECHANISM_STAGE1_OPENI_ZH.md
deliverables/cle_v2_mechanism_stage1_20260910/
```

该纯插件A/B现已作为Stage-2完成实现和本地smoke。不得把旧三seed`PEW/BER+CDep`结果写成纯
PEW+BER-only归因；粒度消融和其他底座必须等待Stage-2 Formal verdict。

## CLE-v2 × PEW+BER 八臂 OpenI Benchmark 已通过，Formal 因成本未授权

用户已批准完成统一的 `HFL/Local × gamma0/gamma0.9 × baseline/plugin` 八臂实现、正式数据与公共 PEW 准备，但未批准 OpenI Formal。正式包、静态审计、5/5 单元测试以及八臂一批次 CUDA smoke 均已完成；同一 gamma 下四臂的 source/AugMix 首批轨迹完全一致。PEW 只在公共 CIFAR-100 上训练一次并冻结；真实 corruption metadata 仍为报告/DSA 专用。

OpenI V100S `mode=benchmark` 已完成。结果包 SHA256 为
`3D3112775B69A794729F4391E8A45EFA3B31F738332D2EA3D90520BDA11194E4`；输入 lineage、
八臂配置及 gamma00/gamma09 内配对训练轨迹均通过完整性审计。八臂实际运行耗时
`1550.4681 s`，峰值显存 `5394.32 MB`。完整协议与冻结六门槛见：

```text
docs/experiments/current/CLE_V2_PEW_BER_FACTORIAL_CLOSURE_OPENI_ZH.md
scripts/openi_cle_v2_factorial_entry.py
```

benchmark 每客户端只执行 8 个 local batches；Formal 为 132 batches、12 rounds。一阶
换算约 `78.45 V100 GPU-hours`，且未计准完整 audit/test/DSA 扩展开销。因此当前判定为
`BENCHMARK_PASS / FORMAL_NOT_AUTHORIZED`。本阶段没有产生科学结果；benchmark 准确率禁止引用。
精确审计见 `deliverables/cle_v2_factorial_benchmark_20260909/RESULT_SUMMARY_ZH.md`。

首次 OpenI benchmark 在首臂 `h0_b` 的 local phase 以 `SIGABRT` 退出。复现审计发现
benchmark/formal 原配置启用了 `num_workers=2`，而 AugMix 数据变换含局部 Lambda；该路径在
Windows 明确触发不可序列化错误，并在 OpenI worker 进程中不稳定。现已将八臂所有模式固定为
`num_workers=0`；这只改变加载吞吐，不改变样本、batch、随机种子、损失、轮数或八臂协议。
修复后原失败臂已按完整 benchmark 设置（每客户端 8 batches）本地 CUDA 跑通，5/5 聚焦测试
通过。失败任务不构成科学结果；成功 benchmark 也只用于链路和成本判断。

2026-09-09 正式输入包已通过 `openi==3.0.1` 命令行上传至
`chujiu/CLE_v2_Factorial_Seed0_PEW_20260909`，CLI 返回 100%。由于上传使用的 Token 曾在聊天
中明文出现，必须撤销并轮换；仓库只记录 CLI 流程与本机凭据路径，不保存 Token。当前不允许
创建 Formal、付费长任务或多种子任务；下一步先讨论降本或代表性 full-round 成本测试。

## Latest Artifact: CLE-HFL / DSA / Local-First / PEW+BER Web Discussion Handoff

A self-contained GPT Web discussion document now records the complete four-stage research chain,
exact mathematical objects and results, RAHFL inheritance, PEW+BER evidence limits, frozen negative
routes, and the mandatory questions for any proposed plugin extension. It explicitly separates the
CLE-v1 Phase-A0/A1a mechanism evidence from the CLE-HFL v2 PEW+BER method evidence and does not
authorize implementation or experiments:

```text
deliverables/cle_hfl_pew_ber_plugin_discussion_20260908/
  CLE_HFL_PEW_BER_GPTWEB_HANDOFF_ZH.md
```

## Latest Decision: WEC-BER Phase-0 Fails Public Error Transfer

The zero-training, CPU-only WEC-BER witness-error transfer gate is complete. The primary channel was
corrected before execution to a `6 observed x 5 latent` matrix: PEW unknown remains an observable
outcome but is not treated as a base corruption family. Each operator cross-fit scores only its true
family column. Public outputs were sealed before any private oracle could be opened.

```text
G1 numerical identifiability: PASS
G2 operator error transfer:   FAIL
G3 public recovery:           PASS
G4 private support/risk:      NOT OPENED
verdict:                      NO_GO_WEC_BER_ERROR_TRANSFER
TRAINING_NOT_STARTED
```

The pooled public Q is full rank with min singular value `0.258820` and condition number `3.813622`,
but operator cross-fit median/p75 column L1 are `0.309333/0.526667`, above the frozen `0.25/0.35`
limits. Dominant-confusion preservation is `68.75%`. Severity dependence is also high (max
Frobenius `0.649774`, maximum diagonal-recall range `57.50 pp`). Therefore pooled invertibility does
not establish a transferable PEW error channel. G3 cannot override the mandatory G2 failure.

Private oracle, DSA and GPU were not opened/used. The exact PEW+BER archive also lacks a classifier
checkpoint (`save_final=false`), so private risk recovery was independently unavailable. Do not
revive WEC-BER by deleting hard operators, selecting families, tuning lambda/threshold/gates, adding
seeds, or implementing severity-aware training. Full evidence:

```text
deliverables/wec_ber_phase0_20260907/RESULT_SUMMARY_ZH.md
deliverables/wec_ber_phase0_20260907/PHASE0_MANIFEST.json
scripts/analyze_wec_ber_phase0.py
fedprime/methods/witness_error_correction.py
configs/wec_ber_phase0_seed0.json
tests/test_witness_error_correction.py
```

## Current Objective: PEW Paper Convergence / Web Draft Handoff Ready

The user has frozen LCRE/CVRS and authorized a conservative small-paper convergence path. The exact
final method is `calibrated hard PEW + hard BER + AugMix/JSD/DCL + strict AsymHFL-val`, with CDep
removed. A zero-GPU static code/evidence audit is complete. The original implementation and entries
remain present; private corruption metadata is diagnostic-only, while training uses hard PEW
pseudo-environments and fit-only client-local class-by-environment counts.

The paper package is:

```text
deliverables/pew_paper_convergence_20260906/PEW_PAPER_WEB_HANDOFF_ZH.md
deliverables/pew_paper_convergence_20260906/PEW_CODE_AUDIT_ZH.md
deliverables/pew_paper_convergence_20260906/EVIDENCE_LEDGER_ZH.md
deliverables/pew_paper_convergence_20260906/TABLE1_MAIN_BASELINES.csv
deliverables/pew_paper_convergence_20260906/TABLE2_ABLATIONS.csv
deliverables/pew_paper_convergence_20260906/TABLE3_OPERATOR_LOO.csv
```

Primary exact-method evidence is fixed-scenario, training-seed-0, 12-round main comparison,
ablation, operator-level LOO and efficiency. Historical three-seed and 40-round positive packages
included CDep and must remain supporting early-system evidence, never exact PEW+BER evidence. The
paper must be framed as a controlled CLE-HFL benchmark plus taxonomy-assisted empirical mitigation,
not taxonomy-free method novelty or universal SOTA. No experiment is active or authorized. The next
action is a web-GPT manuscript draft followed by a sentence-level evidence audit in this repository.

## Current Objective: LCRE M0 Implemented / Local Benchmark Complete / Formal Locked

The user authorized a new cheap method gate for Label-Conditioned Response Equalization (LCRE).
LCRE penalizes the class-balanced between-label component of centered PRIME logit responses,
`mean_q Var_c(E[delta_q|Y=c]) / stopgrad(E_q_bal)`. It uses only private fit images and task labels;
it does not read public CIFAR-100 carriers, corruption metadata, severity, CLE binding or DSA during
training. It is a method hypothesis, not an identification, novelty or causal-mechanism claim.

Implementation is isolated from CVRS history:

```text
method: fedprime/methods/lcre.py
runner: scripts/run_cle_lcre_m0.py
OpenI:  scripts/openi_cle_lcre_m0_entry.py
config: configs/cle_lcre_m0_seed0.json
spec:   docs/experiments/current/CLE_LCRE_M0_CHEAP_METHOD_GATE_ZH.md
tests:  tests/test_lcre.py
```

Focused tests are `13/13 PASS`. The real-checkpoint CUDA smoke completed on RTX 3050 for
ResNet10/client0 and MobileNetV2/client3 with Baseline/Private-PRIME-JSD/LCRE. All output checkpoints
strict-loaded; private/AugMix traces matched across all three arms per architecture; JSD/LCRE probe
traces matched; BN running-stat audits passed; taxonomy/public/oracle assets were not opened. LCRE
active-class counts were 6 and 7 in smoke, with zero skips. Verdict:
`SMOKE_ONLY_NO_SCIENTIFIC_DECISION`.

The corrected 8-step local benchmark also passed all integrity checks. LCRE active-class counts were
`{5:1,6:1}` for ResNet10 and `{7:2}` for MobileNetV2; skip rate was zero. The wall-clock projection
for the complete six-arm, three-epoch training portion is `8150.01 s / 2.2639 RTX-3050 GPU-hours`,
excluding final Phase-A0 evaluation. This local number cannot establish the frozen `<=1 V100
GPU-hour` cost gate. Verdict: `BENCHMARK_ONLY_NO_SCIENTIFIC_DECISION`.

No Formal has started. The next permitted action requires user direction: either stop, or run an
OpenI V100 benchmark using the existing 109142359-byte CVRS M0 v2 archive to obtain a platform-valid
cost estimate. Formal remains locked behind a later explicit user approval and
`--mode=formal --confirm-formal`. Do not tune lambda, active-class rules, probes, epochs or gates.

## Current Objective: Post-CVRS Research Decision / No Active Experiment

The user explicitly ended the open-ended P5/P6/P7 mechanism-audit loop and authorized implementation
of one working method candidate:

```text
CVRS = Class-Visible Routing Suppression
```

For unlabeled public carriers and frozen generic probes, CVRS penalizes normalized persistent
class-directional logit response, `mean_q ||mean_u delta_q(u)||^2 / stopgrad(E_q)`. It does not infer
corruption taxonomy or a private CLE binding. M0 is a cheap matched local-adaptation kill test from
the existing H9 round-40 ResNet10/client0 and MobileNetV2/client3 checkpoints with three arms:
unchanged private AugMix/JSD/DCL only, plus ordinary public JSD, or plus CVRS. The historical artifacts
contain no optimizer state, so all arms use a fresh matched Adam and the experiment must be described
as short adaptation rather than seamless round-41 continuation.

Implementation, focused unit tests, a real-checkpoint CUDA tiny smoke and a local CUDA benchmark are
complete. Unit tests are 7/7 PASS, including one-batch numerical equality with the original RAHFL
private loss. Smoke passed both architectures and all three arms; private batch/AugMix trace hashes
were identical across arms. Smoke verdict is `SMOKE_ONLY_NO_SCIENTIFIC_DECISION`.

The local 8-private-step benchmark also preserved matched traces and returned
`BENCHMARK_ONLY_NO_SCIENTIFIC_DECISION`. On the RTX 3050 it projected `7786.66 s / 2.163
single-GPU hours` for the six-arm 3-epoch training portion, excluding final held-out routing and DSA
evaluation. That local estimate has now been superseded for cost planning by the completed OpenI
V100S benchmark below.

The final compact archive is `109142359` bytes with SHA256
`E9427A55DBE2545AF9D5A1EBD8BEA5B18C41C84D7FE89D06674165F4109E3818`. It contains only private
images/labels for clients 0/3 and no private corruption IDs, methods, severities or source metadata.
The M0-specific loader never opens such metadata. A local smoke verified all six arms and matched
private traces after this tightening. The older 172255488-byte package is superseded and must not be
uploaded.

The OpenI benchmark is now complete on `Tesla V100S-PCIE-32GB`. All frozen input, checkpoint and
Bank-A/B hashes matched; evaluation assets were not extracted; and each architecture's three arms
used an identical private batch/AugMix trace. Mean private/public step times were `0.117390/0.124062 s`.
The projected six-arm, three-epoch training portion is `328.047 s / 0.091124 single-GPU hours`,
excluding held-out routing and final DSA/accuracy evaluation. This is about `23.74x` faster than the
local RTX 3050 projection. Archive SHA256 is
`570BC2B57E8012750DCCD575E617A675E3C492A3E429F6BD25FA202700E6AE7D`; verdict remains
`BENCHMARK_ONLY_NO_SCIENTIFIC_DECISION`.

```text
raw: outputs/openi_downloads/cle_cvrs_m0_seed0_benchmark/cle_cvrs_m0_seed0_benchmark_outputs.tar.gz
```

```text
spec:   docs/experiments/archive/CLE_CVRS_M0_CHEAP_METHOD_GATE_ZH.md
config: configs/cle_cvrs_m0_seed0.json
method: fedprime/methods/cvrs.py
runner: scripts/run_cle_cvrs_m0.py
OpenI:  scripts/openi_cle_cvrs_m0_entry.py
packer: scripts/prepare_cle_cvrs_m0_openi_input.py
tests:  tests/test_cvrs.py
```

Frozen M0 details: public training uses only K0-B Ua (500 unlabeled CIFAR-100 carriers) with Bank A;
held-out routing uses disjoint Ub positions 500:756 with Bank-B probes 0:15. Public steps freeze BN
running statistics. Lambda was set once per regularized arm to 10% of the initial exact private-loss
gradient norm and then frozen. Formal sealed all taxonomy-free training and held-out outputs before
opening the existing DSA oracle. Communication was not modified and no detector trigger, CRSF or
lambda grid was used.

The seed-0 Formal is now complete and independently checked at the artifact/gate-arithmetic level.
Archive SHA256 is `5916858CC71A7AFF1F18B4EEB90F7D3D0C9A13E0B695BFD2F7F7B278861DAB27`.
The taxonomy-free result seal and all six output checkpoint hashes matched. ResNet10 passed all four
gates: CVRS reduced DSA `0.293632 -> 0.219023` (`25.409%`), beat JSD by `0.036972`, and improved
Avg/Worst by `0.500/2.800 pp`. MobileNetV2 reduced DSA `0.172066 -> 0.129098` (`24.972%`) and
improved Avg/Worst by `1.4375/6.100 pp`, but failed the decisive CVRS-vs-JSD gate: JSD DSA was
`0.116098`, so `DSA_JSD - DSA_CVRS = -0.013000`, below the frozen `+0.02` threshold.

```text
verdict: NO_GO_CVRS
full_hfl_training_authorized: false
report: deliverables/cle_cvrs_m0_formal_20260905/RESULT_SUMMARY_ZH.md
raw: outputs/openi_downloads/cle_cvrs_m0_seed0_formal/cle_cvrs_m0_seed0_formal_outputs.tar.gz
new-session prompt: deliverables/cle_cvrs_m0_formal_20260905/NEW_SESSION_PROMPT_ZH.md
```

CVRS lowered its taxonomy-free routing proxy more than JSD on MobileNetV2, yet had worse oracle DSA;
the proxy-to-harm link is therefore not architecture-stable. Do not tune lambda, alter the gate, use
pooled means to override the per-architecture contract, add seeds, or start full HFL. CVRS is frozen
as a negative method result.

## Immediate Next Action

There is no active training, OpenI task or implementation candidate. The CLE-HFL scenario remains
empirically established, but the latest taxonomy-free mitigation is not a method GO. The next session
must first make a research-level decision using the existing evidence; it must not restart an old
negative route merely because a new chat has less context.

Any new method proposal must confront the latest counterexample: a generic public-response proxy can
fall substantially while the true class-corruption DSA is worse than ordinary Public-JSD on another
architecture. A proposal therefore needs either an identifiable signal that is closer to harmful
pairwise class routing, an explicitly justified additional observable, or a narrower claim. Do not
implement or run anything until that proposal has a frozen mathematical objective, distinction from
Public-JSD/PEW/CRSF/CVRS, a minimal attribution design, and a cheap kill gate.

If a later implementation reaches OpenI readiness, follow the required launch card in `AGENTS.md`:
state commit/push status, dataset reuse or upload, exact file path/bytes/SHA256, startup file, every
parameter/value, GPU/cost, expected archive, download folder, metrics and frozen gates.

## Latest Decision: P4 Rejects HFL Decoupling; Marginal Profile Cannot Specify Pairwise Action

P4 used only sealed K0-B/P2, Phase-A1a, P3-A clean-base and P3-A permutation/null outputs. It ran no
model inference, GPU/OpenI, PRIME, training or permutation search. The harmful matrix uses exactly the
original DSA binding-family and valid-source condition, only for post-hoc explanation.

```text
verdict: GENERIC_ROUTING_ALIGNS_WITH_HARMFUL_ROUTING_BUT_TARGET_RULE_FAILED
recommendation: INFORMATION_MAY_EXIST_IDENTIFIABILITY_AUDIT_REQUIRED
method_go: false
```

The proposed HFL-vs-Local decoupling hypothesis is false: H9 mean generic/harmful cosine/Spearman is
`0.8496/0.7091`, versus L9 `0.8459/0.4667`; both have Top-3 overlap `0.8333`. The split instead comes
from how the fixed rank reversal acts on the full pairwise harmful-routing matrix. H9 targeted signed
destructive percentile is only `42.5%`, while L9 is `100.0%`; this matches the P3-A DSA split of
`60.1%` versus `0.0%` (lower DSA percentile is better).

Generic probes therefore recover class marginal salience but have not identified the pairwise map
needed for intervention. P2/P3-A remain NO-GO. Do not search a better permutation, use oracle M_harm
as a loss, start P3-B or train. If the user continues, the only permitted next step is a paper-level
identifiability audit of whether taxonomy-free observables can determine the missing pairwise map.

```text
report: deliverables/post_no_go_p4_routing_targetability_gap_20260904/P4_ROUTING_TARGETABILITY_GAP_AUDIT.md
analyzer: scripts/analyze_post_no_go_p4_routing_targetability_gap.py
```

## Latest Decision: P3-A Complete / Generic Profile Not a Valid HFL Targeter

The user authorized the exact missing-data completion: the same 1,000 Phase-A1a CIFAR-10 clean
sources were forwarded through all 16 H0/H9/L0/L9 round-40 checkpoints. This was clean-only local
RTX 3050 inference: no training, backward, corruption/PRIME generation or checkpoint modification.
Disabling Ampere TF32 matched the sealed V100 reference within `5.36e-7` maximum probability error,
with 100% argmax agreement. The output is `4 x 4 x 1000 x 10`, SHA256
`4D24CFC...0A7A7F`.

The original frozen P3-A then ran as a pure CPU output-space counterfactual. K0-B Bank-A + carrier
half Ua alone defined each arm/client rank-reversal class permutation. Seed `20260904` generated
1,000 unique random derangements per arm/client before binding, corruption family or DSA were read.
All response magnitude/geometry/K0-B-risk invariants passed; maximum error was `3.64e-10`.

```text
verdict: CLASS_IDENTITY_CAUSAL_BUT_GENERIC_PROFILE_NOT_TARGETING
status:  NO_GO_TO_METHOD
```

H9 DSA changed `0.204270 -> -0.013220` and L9 `0.205189 -> -0.061513`, with 4/4 clients positive.
However, the decisive HFL targeting gate failed: H9 targeted was at random-null percentile `60.1%`,
not the required bottom 10%. L9 alone was at `0.0%`. Thus changing class identity strongly changes
signed DSA, but the P2 taxonomy-free profile does not identify a specifically harmful HFL routing
assignment better than random. P2 remains an observational descriptor only. Do not run P3-B, tune
the permutation/profile, design a routing loss or start training.

```text
spec: docs/experiments/archive/P3A_ROUTING_IDENTITY_CAUSAL_AUDIT_ZH.md
clean exporter: scripts/run_p3a_clean_base_completion.py
analyzer: scripts/analyze_post_no_go_p3a_routing_identity.py
report: deliverables/post_no_go_p3a_routing_identity_causal_audit_after_clean_completion_20260904/P3A_ROUTING_IDENTITY_CAUSAL_AUDIT.md
```

## Latest Decision: P2 CLE-Specific Class-Visible Routing Audit Complete

P2 reused the complete K0-B round-40 H0/H9/L0/L9 response grid: four heterogeneous clients, two
independent 64-recipe PRIME banks and two disjoint 500-carrier halves. It ran pure NumPy analysis
only: no checkpoint load, model inference, PRIME generation, training, GPU or OpenI. All 16 response
hashes and their Phase-B0 final-round checkpoint lineage matched. Taxonomy-free outputs were written
and SHA256-sealed before Phase-A1a DSA and original K0-B risk were opened.

```text
CLE_SPECIFIC_CLASS_VISIBLE_ROUTING
+ CLASS_ROUTING_EXCEEDS_GENERIC_FRAGILITY
status: CANDIDATE_MECHANISM_FOR_CAUSAL_AUDIT
```

Across all eight system/bank/half pooled slices, the mean-client H9/H0 or L9/L0 ratios were
`1.596--2.008x` for output-spectrum concentration, `4.178--5.262x` for normalized positive
class-routing strength and `2.176--2.614x` for class-profile concentration. Every slice had 4/4
positive-chi clients. Strong-CLE class profiles were highly stable: minimum cross-half cosine
`0.997664`, minimum cross-bank cosine `0.975400`. Raw centered-response energy increased only
`2.154--3.187x`, so the normalized structural contrast is not just uniform generic fragility.

The new object does not obviously reduce to K0-B R. Across all 16 observations, positive routing
strength correlates with DSA at Pearson/Spearman `0.9810/0.8969`, compared with K0-B R at
`0.8648/0.8616`; its residual Pearson association after K0-B R is `0.9305`. More importantly, across
the eight matched CLE effects, positive routing-strength delta tracks DSA delta at
`0.9459/0.9524`, while `chi_out` delta does not (`-0.0030/0.1667`). Therefore the candidate is the
explicit stable nuisance-to-class routing profile/strength, not another scalar spectrum detector.

This is still observational, single-seed and retrospective. It does not revive CRSF and does not
authorize training or a new loss. The next permitted step is a paper-only causal-routing audit
design that must isolate routing identity/strength from raw response magnitude and K0-B detection;
no paid experiment is currently authorized.

```text
report:   deliverables/post_no_go_cle_specific_routing_audit_20260904/P2_CLE_SPECIFIC_ROUTING_AUDIT.md
seal:     deliverables/post_no_go_cle_specific_routing_audit_20260904/primary_taxonomy_free_manifest.json
analyzer: scripts/analyze_post_no_go_cle_specific_routing_p2.py
```

## Latest Decision: Post-NO-GO Class-Readout Audit P1 Complete

The final zero-cost representation audit loaded only CPU state dictionaries and existing
K1-C-Minimal moments/Grams/DSA. It ran no model inference, PRIME generation, optimization, GPU or
OpenI job. All checkpoint/classifier hashes and representation dimensions matched; all 24 response
matrices reconstructed their saved Grams with maximum error `9.55e-13`.

```text
READOUT_COUPLING_REMAINS_INTACT
+ GLOBAL_CHI_MISSES_CLASS_VISIBLE_GEOMETRY
+ READOUT_WEIGHTED_GEOMETRY_TRACKS_DSA
status: CANDIDATE_MECHANISM_FOR_NEXT_AUDIT
```

CRSF's class/probe routing-matrix cosine is `0.995453` for ResNet10 and at least `0.999689` for
MobileNetV2; dominant-mode readout-coupling cosine is at least `0.997850`; Top-3/Top-5 class-norm
sets remain unchanged. Yet the exploratory readout-weighted chi tracks DSA changes better than
global chi: after removing the functionally duplicated L9/ResNet rows, descriptive Pearson/Spearman
are `0.9980/0.9429` versus `0.9643/0.7143`. RawSpec supplies sign counterexamples where global chi
falls while readout-weighted chi and DSA worsen.

This is not a method GO and does not revive CRSF. H0/L0 feature-space responses, independent seeds
and full client/architecture coverage are missing, so CLE specificity and causal mediation remain
unproven. No new training or paid experiment is authorized.

```text
report:   deliverables/post_no_go_class_readout_audit_20260904/P1_CLASS_READOUT_ROUTING_AUDIT.md
assets:   deliverables/post_no_go_class_readout_audit_20260904/ARTIFACT_AVAILABILITY.md
analyzer: scripts/analyze_post_no_go_class_readout_p1.py
```

## Latest Decision: Post-NO-GO Mechanism Audit P0 Complete

The CPU-only P0 audit is complete. It used only existing K0-B, K1-B0, K1-C0 and K1-C-Minimal
artifacts; it loaded no checkpoint, generated no PRIME view, ran no model forward/backward pass and
used no GPU or OpenI job.

The allowed mechanism diagnoses are:

```text
GLOBAL_SPECTRUM_IS_WEAK_PROXY
+ CLASS_ROUTING_REMAINS_INTACT
+ ARCHITECTURE_DEPENDENT_CONTROLLABILITY
+ SPECTRAL_REDUCTION_BY_TAIL_REDISTRIBUTION
```

CRSF preserved the class-routing vector almost exactly: H9/L9 client-binding cosines were
`0.998579/0.998559`, rank correlations `1.000000/0.998496`, and Top-3/Top-5 overlap was 100%.
Pooled chi-to-DSA conversion efficiency was only `0.433/0.423`. Spectrally, CRSF reduced lambda-1
by 6.43% on average but increased tail energy by 6.21%; the principal-vector cosine remained
`0.999693`. ResNet10 carried most of the weak effect, while MobileNetV2 barely responded. H9/L9
client0 outputs are bit-identical and must not be counted as independent ResNet replications.

```text
report:   deliverables/post_no_go_mechanism_audit_20260904/POST_NO_GO_AUDIT.md
inputs:   deliverables/post_no_go_mechanism_audit_20260904/ARTIFACT_AVAILABILITY.md
analyzer: scripts/analyze_post_no_go_mechanism_p0.py
```

K1-C-Minimal remains `NO_GO_CRSF_INTERVENTION`; this audit does not authorize tuning, B-to-A,
additional architectures, replication, full training or a new method. There is currently no paid
experiment to run. The next step is a research decision about whether CLE should be approached at
the class-conditional routing level or whether the representation-intervention branch should stop;
P0 itself does not make that decision.

## Latest Decision: K1-C-Minimal Causal Intervention NO-GO

The old exact K1-C-FULL route is frozen as `SUPERSEDED_BEFORE_FORMAL`. It produced no calibration or
formal scientific result and must not be restarted. Its specification and implementation remain only
for provenance. K1-C0 remains a 10/10 observational GO: it established a CLE-associated generic
response-spectrum concentration, not that CRSF surgery works.

K1-C-Minimal is now implemented to test only the missing causal arrow:

```text
reduce response-spectrum concentration -> does real CLE DSA decrease?
```

The preregistered primary uses H9/L9, ResNet10/client0 and MobileNetV2/client3, A-to-B only, and
Frozen/CRSF/RawSpec. Correction uses 512 prespecified D_surgery carriers and 16 prespecified Bank-A
probes for five accepted steps at initial LR 1e-4. Every step retains exact post-update objective,
anchor-KL <= 0.02, rollback and deterministic LR halving. There is no separate LR calibration.
Evaluation remains full and independent: D_holdout 2,000 x all 64 Bank-B probes, sealed before CLE
binding and DSA are opened.

```text
spec:    docs/experiments/current/CLE_K1_C_MINIMAL_CAUSAL_GATE_OPENI_ZH.md
config:  configs/cle_k1_c_minimal_seed0.json
runner:  scripts/run_cle_k1_c_minimal.py
OpenI:   scripts/openi_cle_k1_c_minimal_entry.py
tests:   tests/test_cle_k1_c_minimal.py
```

Focused regression is 14/14 PASS. A real-checkpoint CUDA smoke completed with verdict
`SMOKE_ONLY_NO_SCIENTIFIC_DECISION`: frozen selection hashes verified, both CRSF and RawSpec accepted
one exact step, all metrics were finite, bounded in-memory streaming wrote no transformed-input disk
cache, and no oracle/evaluation assets were loaded. This is execution evidence only.

The OpenI benchmark completed and independently passed the engineering cost gate. It ran one H9/ResNet10 context without
oracle/evaluation assets, used 544 MiB bounded prefix arrays and 341.9 MiB peak CUDA allocation, and
projected Minimal Formal at 894.39 seconds / 0.2484 single-GPU hours. Allow 30--45 minutes as a
conservative 2--3x envelope because MobileNetV2 was not directly timed and oracle cost was proxied.
This was not scientific evidence.

Minimal Formal has now completed and was independently recomputed from the sealed response moments and
oracle predictions. Verdict: `NO_GO_CRSF_INTERVENTION`. H9/L9 CRSF unseen-chi reductions were only
`5.369%/5.452%` versus the frozen 15% gate; CRSF-minus-RawSpec advantages were only
`3.838/3.959 pp` versus 10 pp. DSA reductions were `0.005237/0.005193` (`2.322%/2.308%`) versus the
`0.05 or 25%` gate, with only `0.00549/0.00590` advantage over RawSpec versus 0.02. ResNet10 carried
most of the weak effect; MobileNetV2 chi and DSA changes were near zero.

All 31 artifact hashes and 18 pre-oracle seal hashes matched. Raw moments/Gram and all six prediction
files reproduced chi, DSA and reporting metrics exactly. All eight intervention traces completed five
accepted monotone steps; maximum final KL was 0.019646. This is a scientific method failure, not an
execution failure. Stop CRSF: no B-to-A, remaining architectures, tuning, replication or full training.
K1-C0 remains an observational mechanism result only; K0-B remains offline audit only.

```text
benchmark archive sha256: D16E82F85FFA636DBEE50086BF6A083F932BB1F8833F3F7E366F5E90AF24F2D4
formal archive sha256:    E07B9E75E2AEDDE0C1B3A4FF018CE0B4FD90EAA6CB88144D6E0D98588E43D4CA
report: deliverables/cle_k1_c_minimal_formal_20260904/RESULT_SUMMARY_ZH.md
```

## Previous Objective: K1-B0 CDR-SNR Shared Representation Localization

## Current Objective: K1-B0 CDR-SNR Shared Representation Localization

K1-A head-only SDMN formal is now frozen as `NO_GO_DIRECTIONAL_SURGERY`; do not tune or revive it.
K0-B's taxonomy-free detector remains valid, but changing only the classifier head did not close the
detect -> intervene -> CLE reduction loop.

The active stage is the zero-training K1-B0 localization gate. It asks whether the frozen K0-B
high-risk PRIME probes expose a carrier-stable, high-vs-energy-matched-specific and cross-bank
transferable nuisance subspace in the penultimate representations of H9/L9, stronger than matched
H0/L0. It does not yet perform SNR surgery or full FL training.

```text
spec:      docs/experiments/current/CLE_K1_B0_CDR_SNR_OPENI_ZH.md
engine:    fedprime/engine/cle_shared_nuisance_routing.py
analyzer:  scripts/analyze_cle_k1_b0_cdr_snr.py
OpenI:     scripts/openi_cle_k1_b0_cdr_snr_entry.py --mode=formal
selection: fedprime/augmentations/assets/cle_k1_b0/selection_manifest.json
tests:     tests/test_cle_shared_nuisance_routing.py
```

The selection manifest was reconstructed from the independently audited K0-B formal archive
`1E02A16C...88608`; it freezes H9/L9 active probes, top-20% rho probes and weights for both 64-recipe
banks. Local INSPECT verified all 16 frozen checkpoints, both bank hashes, D_select hash
`731B8CFF...F6CA`, and D_rep hash `321C0910...40EE`. The tiny local smoke passed with verdict
`SMOKE_ONLY_NO_SCIENTIFIC_DECISION`; no labels/evaluation assets were read and no optimizer,
backward, training or checkpoint write occurred.

Next action: run the unchanged 535,256,689-byte Phase-B0 OpenI dataset with
`scripts/openi_cle_k1_b0_cdr_snr_entry.py --mode=formal`. Download the compact
`cle_k1_b0_cdr_snr_seed0_formal_outputs.tar.gz`, audit all 20 frozen gates, and stop. A GO only
authorizes design of cross-bank SNR surgery; it does not authorize starting it automatically.

## Latest Formal Gate: K0-B v2 Taxonomy-Free Generic Probe

K0-A formally passed and K0-B v2 is now implemented without training or checkpoint writes. Its
primary object is carrier-stable plus class-selective response, not ordinary split-direction
reproducibility:

```text
spec:       docs/experiments/current/CLE_GENERIC_PROBE_K0B_OPENI_ZH.md
bank code:  fedprime/augmentations/frozen_prime.py
statistics: fedprime/engine/cle_generic_probe_gate.py
analyzer:   scripts/analyze_cle_generic_probe_k0b.py
OpenI:      scripts/openi_cle_generic_probe_k0b_entry.py --mode=smoke
tests:      tests/test_cle_generic_probe_k0b.py
```

Two complete 64-recipe PRIME banks are versioned under
`fedprime/augmentations/assets/cle_generic_probe_k0b/`. Their preregistered canonical hashes are
`6CAE529D...3DC01` and `4A53497E...BF4E`. Every primitive composition, spectral state,
displacement field, color coefficient, filter kernel, strength, mixture weight and depth is fixed
before inference and reused for every carrier.

The formal OpenI result was independently verified from all 16 raw response tensors and is
`GO_TO_K1_CHECKPOINT_SURGERY`. HFL and Local each passed all eight frozen gates; generic-fragility
kill was false. HFL K delta was `+0.252727`, combined R ratio `4.901569`, and both bank ratios were
`5.739226/4.317300`. Local K delta was `+0.232752`, combined R ratio `4.385780`, and both bank ratios
were `5.166668/4.094945`. Both systems had 4/4 positive-R clients. Manual S/Dcf/K/R recomputation
matched the returned metrics to maximum absolute error `5.56e-17`.

```text
result:  cle_generic_probe_k0b_seed0_formal_outputs.tar.gz
bytes:   234888047
sha256:  1E02A16C765D8AB976A692D444FA9DAEBE38C30F8279CD6DCCFC49D1BFF88608
report:  deliverables/cle_generic_probe_k0b_20260902/RESULT_SUMMARY_ZH.md
```

K1-A head-only SDMN is now implemented through INSPECT, smoke and numerical-calibration modes:

```text
spec:    docs/experiments/current/CLE_K1_SDMN_HEADONLY_OPENI_ZH.md
engine:  fedprime/engine/cle_sdmn_headonly.py
runner:  scripts/run_cle_k1_sdmn_headonly.py
OpenI:   scripts/openi_cle_k1_sdmn_headonly_entry.py --mode=smoke
tests:   tests/test_cle_sdmn_headonly.py
```

INSPECT verified all 16 checkpoints, both frozen banks, CIFAR-100 and the existing CLE evaluator
assets. Focused K1/K0 regression is 20/20 PASS. The local H9-client0/A-to-B tiny smoke ran Frozen,
Targeted, Direction-Sham, Random-Probe and Generic-Invariance, wrote full checkpoints/traces/hashes,
kept every anchor KL below 0.003, and reproduced identical split/selection/features/metrics/traces in
a second run. Its verdict is only `SMOKE_ONLY_NO_SCIENTIFIC_DECISION`.

The OpenI smoke is now independently audited and passed: archive/manifest hashes matched, public
split overlap was zero, all four surgery objectives decreased, maximum anchor KL was `0.002898`,
all checkpoints changed only `linear.weight/bias`, and raw unseen responses reproduced reported
S/Dcf/K/R exactly. Verdict remains `SMOKE_ONLY_NO_SCIENTIFIC_DECISION`.

The OpenI calibration artifact was independently audited. All 16 client/fold cases passed at
`1e-4`, two passed at `3e-4`, and none passed at `1e-3`. The raw result had a JSON naming collision
where the per-step LR trace overwrote the candidate scalar; the underlying traces are complete and
the scalar is unambiguously recoverable, so no rerun is required. The corrected schema and frozen
per-client/fold LR values are versioned in
`configs/cle_k1_sdmn_headonly_calibration_seed0.json`.

Formal K1-A is now implemented with Adam, 10 steps, anchor KL `<=0.02`, backtracking factor `0.5`
and at most 12 rollbacks. Its 2,000-image surgery pool preserves the exact calibration hash after
adding a disjoint 2,000-image holdout. It runs Frozen/Targeted/Direction-Sham/Random-Probe/Generic-
Invariance for H9/L9 and both A-to-B/B-to-A folds. Taxonomy-free unseen-bank artifacts are written
and hashed before the CLE binding, true corruption grid, DSA/WCCA/CFG or task labels are opened.

Next action: run the unchanged Phase-B0 OpenI dataset with
`scripts/openi_cle_k1_sdmn_headonly_entry.py --mode=formal`. Stop when K1-A returns one of its three
frozen verdicts; do not start full training, modify communication, or revive PNCB/PEW/BER.

## Latest Formal Gate: K0-A Public-Carrier Transfer Oracle

The CLE-HFL topic remains active, but PNCB-SCDW remains stopped. The only current method-candidate
gate is a zero-training test of the cross-carrier directional moment:

```text
spec:     docs/experiments/current/CLE_PUBLIC_CARRIER_K0A_OPENI_ZH.md
engine:   fedprime/engine/cle_public_carrier_moment.py
analyzer: scripts/analyze_cle_public_carrier_k0a.py
OpenI:    scripts/openi_cle_public_carrier_k0a_entry.py --mode=smoke
tests:    tests/test_cle_public_carrier_k0a.py
```

K0-A reuses the existing 535,256,689-byte Phase-B0 input archive containing CIFAR-100 and all 16
frozen round-40 H0/H9/L0/L9 checkpoints. It selects 1,000 CIFAR-100 train images with seed 20260901,
does not use their labels, and applies the existing 16 operators at severity 3 only as an oracle
mechanism audit. Blind centered class-vs-rest logit responses are saved and hashed before client-class
binding and operator-family truth are opened for scoring.

Focused regression and local smoke passed. The formal OpenI result was independently recomputed and
is `GO_TO_K0_B`: HFL and Local each passed all 10 preregistered gates. H9/H0 mAP was
`0.796627/0.406176` (delta `+0.390451`); L9/L0 was `0.811235/0.411342` (delta
`+0.399894`). Both systems had 4/4 positive clients and both null tests at `p=0.000999001`.
Directional-strength and coherence bootstrap CI95 lower bounds were strictly positive.

```text
result:  cle_public_carrier_k0a_seed0_formal_outputs.tar.gz
bytes:   48651705
sha256:  AA260672FED05C991DDEF2308342BD88150CA8A36FD8366EF9A9E85B2E523168
report:  deliverables/cle_public_carrier_k0a_20260901/RESULT_SUMMARY_ZH.md
```

The next action is K0-B taxonomy-free generic-probe design. K0-A used the true operator bank and
binding only for oracle scoring, so it does not authorize DME/K1 training. Preserve the local-first
interpretation; the result does not support communication amplification.

Full self-contained handoff for GPT Web discussion:

```text
docs/research/status/CLE_PNCB_SCDW_CURRENT_RESEARCH_HANDOFF_FOR_GPTWEB_2026_08_31_ZH.md
```

It records the problem, evidence chain, PNCB/SCDW mathematics, weakest assumptions, Phase-B0 data
and gates, smoke audit, conditional Phase-B1 design, claim boundaries and eight external-review
questions. A formal-result addendum now supersedes its earlier pending status.

## Latest Formal Result: PNCB Bridge NO-GO

The frozen Phase-B0 formal run completed and was independently checked from its returned probability
cache:

```text
result:  cle_public_canonicalization_phase_b0_seed0_formal_outputs.tar.gz
bytes:   31788115
sha256:  8F824A6EF21AFDF8E8CF089530786882FE684504079C17160DFD2205D140BE2C
verdict: NO_GO_PNCB_BRIDGE

PASS: G1 semantic preservation, G4 HFL retrieval,
      G6 relative overlay margin, G7 clean artifact null
FAIL: G2 old-nuisance contraction, G3 family separability reduction,
      G5 Local retrieval
```

The PNCB completed all 10 epochs and its training loss decreased by 18.77%, so the failure is not an
execution failure. It preserved semantics (worst canonical accuracy delta `-0.5875pp`) but increased
within-source cross-operator variance by `12.2267%` instead of contracting it by at least `25%`.
Family separability fell by only `9.4729%` versus the frozen `30%` requirement. Local gamma9
retrieval was mAP `0.593924`, hit `0.65`; both missed their gates. G6 passed only because overlay was
even more dispersive; it cannot override absolute G2 failure.

Decision: stop the current PNCB-SCDW route. Do not implement or run Phase-B1 classifier/SCDW A/B/C,
do not tune SCDW weights, and do not rescue the bridge by epoch/channel/loss-only tuning. SCDW as an
abstract object was not directly trained, but its required contraction bridge is absent. Any future
bridge must be a new paper candidate with a new pre-result argument and gate, not a renamed revival.

```text
report: deliverables/cle_public_canonicalization_phase_b0_20260831/RESULT_SUMMARY_ZH.md
```

## Latest Paper Design: Public Canonicalization + Signed Directional Withdrawal

The paper-only intervention-bridge design is complete:

```text
docs/research/status/CLE_PUBLIC_CANONICALIZATION_DIRECTIONAL_WITHDRAWAL_DESIGN_2026_08_30_ZH.md
```

Directly overlaying another fixed corruption bank on already-corrupted private images is rejected:
it does not overwrite the original degradation and revives an artificial taxonomy. The conditional
candidate instead uses the public unlabeled data already required by heterogeneous logit
communication to learn a frozen Public Nuisance Canonicalization Bridge (PNCB). The bridge is trained
only by public corrupted-to-source reconstruction; it consumes no private class, corruption, family
or binding metadata.

For private image `X`, the client obtains `C(X)` and estimates, for every wrong class, the signed
probability withdrawal `p(c|X)-p(c|C(X))` over samples whose task label is not `c`. The proposed SCDW
loss penalizes only a positive one-sided lower confidence bound, stop-grads the standard-error
threshold and canonical probability, and keeps the existing AugMix/JSD/DCL objective. A separate
canonical-view CE term is mandatory, as is a `bridge-only` arm for attribution. The method adds no
communication and is compatible with heterogeneous backbones because it operates in input and class-
probability spaces.

This is only `CONDITIONAL GO`. Its weakest assumptions are semantic preservation, contraction of the
original hidden degradation, label-independent public reconstruction, paired observability and
limited cancellation of harmful directions. The next gate is bridge-only, before classifier
training:

```text
Identity bridge
vs AugMix overlay
vs public canonicalizer

audit: semantic preservation, source-conditioned nuisance contraction,
       H9/L9-vs-H0/L0 hidden-binding retrieval, clean artifact null,
       and per-client consistency
```

If the public canonicalizer cannot preserve semantics while contracting the old degradation, the
candidate is `NO-GO`; SCDW weights must not be tuned to rescue it. If it passes, the first classifier
experiment is a matched `baseline / bridge-only / bridge+SCDW` 12-round screen. Only the Phase-B0
bridge harness and execution smoke are authorized; no formal OpenI run or classifier training has
started, and no communication method, PEW/BER revival or fixed corruption labels should be added.

Phase-B0 implementation is now complete without classifier training:

```text
spec:       docs/experiments/current/CLE_PUBLIC_CANONICALIZATION_PHASE_B0_ZH.md
model:      fedprime/models/public_canonicalizer.py
engine:     fedprime/engine/cle_directional_withdrawal.py
train:      scripts/train_cle_public_canonicalizer_phase_b0.py
analyzer:   scripts/analyze_cle_public_canonicalization_phase_b0.py
OpenI:      scripts/openi_cle_public_canonicalization_phase_b0_entry.py
tests:      7 focused tests passed; 16/16 final checkpoints strict-loaded
smoke:      4 public images, one CPU batch, execution-only PASS
formal run: NOT STARTED
```

The four approximately 173 MiB H0/H9/L0/L9 Phase-A1a arm archives are now local and verified. Each
contains four final and four round-12 checkpoints; the Phase-B0 input intentionally extracted only
the 16 final round-40 checkpoints. The slim OpenI package is complete:

```text
input:  local_runs/cle_public_canonicalization_phase_b0/
        cle_public_canonicalization_phase_b0_seed0_inputs.tar.gz
bytes:  535256689
sha256: DFB766F6494A5F61AA16F45666EC250A30501066AB54D89C984CD2324293B9BC
entry:  scripts/openi_cle_public_canonicalization_phase_b0_entry.py --mode=smoke
```

It contains only the frozen 1,000-source evaluation arrays, CIFAR-100 public tar, 16 final
checkpoints and a per-file hash manifest; private Phase-A1a training arrays and round-12 duplicate
weights are excluded. The OpenI entry defaults to smoke and verifies both the archive hash and every
manifest file before execution.

The OpenI end-to-end smoke passed on 2026-08-30:

```text
result:  cle_public_canonicalization_phase_b0_seed0_smoke_outputs.tar.gz
bytes:   3344564
sha256:  C57D3A9BE84E04FDDBB35402DF011B59294D78B532951346476182A891E81E54
verdict: SMOKE_ONLY_NO_SCIENTIFIC_DECISION
```

All 19 manifest files verified, all 16 classifiers ran on CUDA, the temporary PNCB completed exactly
one epoch/two batches, and the analyzer used 20 balanced sources, all 16 operators and severity 3.
Independent NPZ inspection found all 12 probability tensors at shape `(4,20,16,10)`, all values
finite and maximum probability-sum error `2.38e-7`. Smoke bridge metrics are not scientific evidence:
the temporary two-batch PNCB showed semantic loss and no nuisance contraction, which is expected to
remain uninterpretable under the frozen smoke rule. The next action is user approval for one formal
`--mode=formal` bridge-only run. Do not alter the seven gates or start classifier/SCDW training. No
Phase-A1a retrain is required.

## Latest Zero-Training Gate: PIDR Recovers Hidden Binding Under Oracle Probes

The paper-only definition, identifiability counterexample, weakest-assumption analysis and 2024--2026
collision audit are complete:

```text
docs/research/status/CLE_LOCAL_FIRST_DIRECTIONAL_SHORTCUT_IDENTIFIABILITY_AUDIT_2026_08_30_ZH.md
```

The current CIFAR CLE construction is a valid controlled stress test and DSA is a binding-specific
directional diagnostic, but neither is evidence that class--corruption spurious correlation is a newly
discovered real-world phenomenon. Real acquisition/quality shortcuts are documented; the cyclic four-
family map and gamma 0.9 are synthetic controls and must be described as such.

The population local-first directional shortcut risk is the DSA causal estimand. It is not identifiable
from a single already-corrupted image, its task label and i.i.d. AugMix views: two worlds can have the
same complete observable transcript while the model uses the degradation in one and the semantic
feature in the other. A view-residual, spectral or clustering rewrite would either miss persistent base
shortcuts or revive frozen C3R/CRSR-style negatives.

The only conditional candidate is Probe-Indexed Directional Promotion Risk (PIDR). It requires a
semantic-preserving intervention bridge whose distinguishable probes overwrite rather than merely
overlay the relevant degradation and cover every harmful direction. Under those assumptions PIDR
upper-bounds the unknown binding-set probability promotion up to coverage and intervention errors.
This is not yet a method GO: FedPIN, FedCD, ShortcutProbe, FedDDL, FedCAug, GIC and recent unlabeled
debiasing methods create strong collisions.

The approved zero-training oracle gate is complete. It reused the existing round12/round40 prediction
caches; no training and no inference ran. The estimator read only probabilities, task labels and 16
probe tensor identities. Binding and operator-family metadata were opened only after all promotion
matrices existed, for scoring and null tests.

Round-40 formal result:

```text
arm   PIDR       mAP       AUC       class-to-family hit
H0    0.024559   0.441855  0.510677  0.225
H9    0.175479   0.844847  0.923906  0.850
L0    0.025401   0.430622  0.507083  0.275
L9    0.174728   0.865557  0.933177  0.875

HFL mAP delta:      +0.402993
Local mAP delta:    +0.434935
positive clients:   4/4 and 4/4
class/probe null p: 0.000999 for H9 and L9
verdict:             GO_TO_INTERVENTION_BRIDGE_DESIGN
```

All six frozen gates passed for both HFL and Local. Round12 already showed the same pattern. This
establishes oracle-side directional observability: distinguishable degradation probes can recover the
hidden class binding without giving the estimator a family taxonomy. It does not establish that
i.i.d. AugMix views overwrite the base degradation, that training without clean sources is identified,
or that PIDR is method-novel relative to ShortcutProbe/FedCD.

```text
engine:     fedprime/engine/cle_probe_directional_promotion.py
analyzer:   scripts/analyze_cle_pidr_zero_training_gate.py
tests:      10 focused tests passed
result:     deliverables/cle_pidr_zero_training_gate_20260830/
commit:     bd37fc6 (local only; not pushed)
```

Next action superseded by the PNCB-SCDW bridge-only gate above. Do not implement a classifier loss or
start 12-round training until that gate passes.

## Current CLE Result: Directional Shortcut Is Local-First; Communication Amplification NO-GO

The bad-teacher / wrong-routing / shortcut-propagation story remains parked. Historical D2C and
Oracle D2C results, FedFalsify attribution and the beta0/beta4 coupling screen do not support making
teacher selection the dominant CLE mechanism.

A narrower zero-training screen asked whether the historical `gamma=0.9` RAHFL
models directionally move predictions toward classes bound to an applied corruption family more
than matched `gamma=0.0` models. Existing assets were audited: the 272,728,582-byte historical
archive contains all four `gamma=0/0.9` client checkpoints. Existing balanced tests repeat the same
source indices across corruptions but independently sample severity and omit explicit source IDs,
so the formal audit must regenerate a deterministic paired evaluation grid from 1,000 clean images:
16 old-CLE operators, severity 3, eight checkpoints, 128,000 forward items.

Formal inference ran on OpenI; the local RTX 3050 was used only for archive/hash checks, tiny
synthetic dry-runs and an independent recomputation from the returned probability cache. The frozen
protocol used Directional Shortcut Alignment, a `gamma09-gamma00` contrast, group-size-preserving
shuffled binding maps, paired bootstrap, integrity gates and five scientific promotion gates.

Implementation and artifact status:

```text
spec:           FROZEN
implementation: COMPLETE
focused tests:  3 passed
checkpoint load: 8/8 strict CPU PASS
input package:  cle_shortcut_alignment_phase_a0_seed0_inputs.tar.gz
input bytes:    184575308
input sha256:   C1F6823E186DDAF6DB44A38BCBDA300C78B9F1B4702C5E84B7AEE3A485499EFE
OpenI run:      COMPLETE
result archive: cle_shortcut_alignment_phase_a0_seed0_outputs.tar.gz
result bytes:   4883205
result sha256:  CCC91ED4EC3F08C6CA6433CA275423AD3E32EDB824993F7760F94A8A49B4B76F
code commit:    e005689 (pushed to origin/main)
```

Frozen specification:

```text
docs/experiments/current/CLE_SHORTCUT_ALIGNMENT_PHASE_A0_OPENI_ZH.md
```

All input-integrity checks passed. The OpenI summary and an independent recomputation from the
returned `predictions.npz` agree exactly:

```text
gamma00 pooled DSA: -0.0003018894
gamma09 pooled DSA:  0.2013210658
delta DSA:           0.2016229552
paired CI95:         [0.1964123272, 0.2072188988]
positive clients:    4/4
pooled shuffled p:   0.000999001
client shuffled p:   0.000999001 for all 4 clients
verdict:             GO_TO_MATCHED_PARTITION_DESIGN (G1--G5 all PASS)
```

Secondary changes from `gamma00` to `gamma09` were Avg `52.2422 -> 46.8250`, Worst
`43.4188 -> 37.9375`, WCCA `36.0000 -> 19.3125`, CFG `2.6500 -> 11.8813`, and paired
prediction-flip rate `0.148325 -> 0.353404`. This is strong evidence that CLE induces a directional
corruption-to-class shortcut under the historical RAHFL protocol. It does not yet show that the
effect is caused or amplified by federation, communication or model heterogeneity. Passing Phase-A0
permits only the matched-global Local-D/Local-E attribution design; it is not a paper or method GO.

Checkpoint provenance was re-audited on 2026-08-30. The eight weights were copied byte-for-byte
from `outputs/cle_rahfl_diagnostic_outputs.tar.gz`, specifically the four final client checkpoints
under each of `diag_rahfl_cle_alpha05_gamma00_seed0` and `diag_rahfl_cle_alpha05_gamma09_seed0`.
They are project-trained historical diagnostic models with matched alpha 0.5, seed 0, architectures,
optimizer, local epochs, public batches and AugMix/JSD+DCL+AsymHFL configuration; only the intended
CLE dataset root differs. Their resolved configs set `pretrain_epochs=0` and `rounds=40`. Therefore
Phase-A0 is an internally valid post-hoc matched diagnostic, but it is not evidence from the canonical
40-pretrain + 40-communication RAHFL schedule. The old prepared gamma00/gamma09 dataset archives are
not currently present locally, so exact client-label/source-index identity cannot now be byte-audited;
the deterministic generator uses the same alpha, seed and `partition_seed=seed`, which establishes
the intended matched partition. Any promoted paper experiment must persist partition/source hashes.

Phase-A1a completed on OpenI on 2026-08-30 using four jointly matched 40-round arms:
`H0/H9 = strict AsymHFL-val at gamma 0/0.9`, `L0/L9 = Local-only at gamma 0/0.9`.
All arms used the same code, source partition, labels, severities, persisted fit/audit split,
per-client initial weights and arm-independent private-loader RNG. Runtime hashes verified that HFL
and Local saw identical first-batch labels and all AugMix/DCL views in every round and client. Final
test labels remained reporting-only. The returned probability tensors were independently recomputed.

Round-40 formal DSA result:

```text
H0  0.0015228341        H9  0.2042704937
L0  0.0008234010        L9  0.2051892788
HFL CLE effect:          +0.2027476596
Local CLE effect:        +0.2043658778
communication A_pool:    -0.0016182182
paired source CI95:      [-0.0033365882, 0.0001283891]
positive clients:        1/4
top-1 amplification:     +0.0025799851
H9 shuffled-map p:       0.000999001
verdict:                 NO_GO_FL_SPECIFIC_AMPLIFICATION
```

G1--G3 failed; G4--G5 passed. At round 12, `A_pool=-0.0169168048` with CI entirely below zero,
but this early suppression disappeared by round 40. CLE itself is strong and reproducible in both
training systems: controls have DSA near zero, while gamma09 has DSA about 0.204; Avg falls about
6--7pp, WCCA falls 20--26pp, CFG rises from about 2.8 to about 12, and paired prediction flips rise
from about 0.15 to about 0.36. The supported mechanism is therefore `CLE -> local-first directional
shortcut`; strict AsymHFL neither materially amplifies nor durably suppresses it.

```text
code commit: 6199dd8 (pushed to origin/main)
input: cle_shortcut_amplification_phase_a1a_seed0.tar.gz
input bytes/sha256: 408228487 / 6322F16513C6980CDC5904D7EF91204A241205BC76DCCE8BC450E635519B4202
result: cle_shortcut_amplification_phase_a1a_seed0_analysis_outputs.tar.gz
result bytes/sha256: 19322309 / FDF1BEC2395334DD3816BC9C3F594B01D814DB22C0CC245CD1378A45180F397C
spec: docs/experiments/current/CLE_SHORTCUT_COMMUNICATION_AMPLIFICATION_PHASE_A1A_ZH.md
```

The user explicitly chooses to continue CLE. Do not revive bad-teacher, routing-amplification,
D2C/Oracle-D2C, PEW/BER-as-core, or tune/repeat Phase-A1a to rescue communication amplification.
The next stage is paper-only design and collision audit for a genuinely local-first directional
shortcut-suppression object that uses no environment/corruption labels, no clean counterpart and no
source index. It must be distinguished from AugMix/JSD, AugMax, consistency regularization,
IRM/VREx, GroupDRO/CVaR, counterfactual invariance and all frozen project negatives before code or
new training. The Phase-A1a bootstrap covers evaluation-source uncertainty only; seed 0 alone does
not establish training-seed stability.

## Superseding Topic-Selection State: Stop Extending RAHFL by Adding One More Difficulty

There is currently no active paper method, implementation task, local run or OpenI run. Preserve
all RAHFL/CLE-HFL/PEW+BER code and evidence, but do not restart any candidate below without a new
paper-level argument that changes its mathematical core.

The adversarial-attack / dual-robustness extension is now formally `PAPER NO-GO`. Model-
heterogeneous FL with adversarial robustness, malicious clients, logit poisoning and trustworthy
heterogeneous distillation already has direct adjacent literature. A narrower story such as
`robustness-profile conflict`, `teacher-ranking conflict`, `routing-amplified poisoning` or
`corruption-camouflaged poisoning` does not by itself create an independent method contribution;
its likely defense reduces to existing detection, trust weighting or robust logit aggregation.
Do not run the proposed Phase-A0, add PGD/FGSM/AutoAttack, modify RAHFL, implement a poisoning
harness or design a defense for this route.

Current corruption/robustness topic pool:

```text
severity heterogeneity:                 NO-GO
compound corruption:                    NO-GO as a paper core
corruption--label coupling:             EXPERIMENTAL SCREEN NO-GO
communication-induced forgetting:       low-value diagnostic; do not invest
robustness propagation:                 composition-risk conditional idea; no active test
data cleaning / sample removal:         NO-GO for the intended paper narrative
availability--corruption coupling:      NO-GO due direct collisions
dynamic corruption:                     NO-GO / highly crowded
compression robustness:                 high composition risk; not active
adversarial attack / dual robustness:   FORMAL NO-GO
PEW+BER:                                strong baseline, not the paper method core
```

The search rule is changed. Do not ask what extra factor RAHFL lacks. Start from an effective
2024--2026 centralized robustness objective or method and identify a necessary information object
that becomes unavailable, non-decomposable or incomparable in FL: global distributions, cross-
sample pairing, global hard-example order, cross-domain statistics, a common parameter space,
stable/unstable feature identities, a centralized reference model, or global corruption
statistics. A candidate advances only if all of the following are made explicit before code:

```text
centralized problem and why its method works
the exact necessary information that is broken by federation
why the fracture is nontrivial and not a toy protocol choice
the weakest assumption that repairs identifiability/executability
2024--2026 collision audit, including federated variants
available datasets/code and a cheap falsification test
```

Next action: paper-only reverse search from centralized robustness methods to genuine federated
structural fractures. Do not implement, run experiments or proactively commit during topic
selection.

## Completed Historical Result: RAHFL Corruption--Label Coupling Phase-A

This completed screen asked whether a fixed amount of annotation noise causes an additional penalty
when wrong labels concentrate on high-severity corrupted samples, and whether RAHFL/HFL amplifies
that penalty. It is retained for provenance and is not an active topic.

Frozen Phase-A1a comparison:

```text
Independent:     beta=0
Strong Coupled:  beta=4
same frozen corrupted images, disjoint client partition, fit/audit split,
20% noisy-label count and true-class -> wrong-class transition matrix
```

Each of four heterogeneous clients has 10,000 mutually exclusive CIFAR-10 samples and an exact
9,000/1,000 fit/audit split. The fit labels contain exactly 1,800 errors per client. The same noisy
label manifest is used by local CE pretraining and communication-round CE/DCL. Trusted clean audit
labels are routing-only; final test labels are reporting-only. The actual legacy RAHFL IID helper is
disjoint, but the default `Network/pretrain.py` and `HHF/RAHFL.py` IID branches independently sample
clients and may overlap; the new disjoint split is an explicit protocol choice.

Implementation status:

```text
commit: 999fee0 (pushed to origin/main on 2026-08-24)
focused tests: 2 passed
beta0/beta4 1+1 smoke: passed
artifact audit: all checks passed
formal 40+40: not started
OpenI seed0 10+10 screen: completed; no promotion
```

The artifact audit recovered a frozen image SHA256 beginning `e20128a7ad50`, verified 40,000 unique
client samples, exact 9,000/1,000 splits, identical flip matrices, clean audits and unchanged tests.
Mean severity among noisy fit samples is 2.5029 for beta=0 and 3.5699 for beta=4.

OpenI input and entry:

```text
dataset: rahfl_coupling_phase_a_seed0_prepared.tar.gz
bytes: 327018418
sha256: AE5F9524AF594963C790016FB386BD0EB600ACD84BBC1D12EF57DA7393D1835F
entry: scripts/openi_rahfl_coupling_phase_a_entry.py
args: --mode=both
```

The completed entry copied condition archives and the paired JSON summary to
`c2net_context.output_path`, then called `upload_output()`. Returned files included
`rahfl_coupling_phase_a_screen_seed0_beta0_outputs.tar.gz`,
`rahfl_coupling_phase_a_screen_seed0_beta4_outputs.tar.gz`,
`rahfl_coupling_phase_a_screen_seed0_both_outputs.tar.gz`, and
`rahfl_coupling_phase_a_screen_seed0_summary.json`.

The OpenI 10+10 screen completed on 2026-08-24. Independently recomputed beta0-minus-beta4 results
were final Avg/Worst `-1.90/-1.83pp`, last-3 Avg/Worst `-2.01/-1.37pp`, and all-10 Avg/Worst
`-0.40/-0.07pp`. Beta4 was better, not worse. Before the first collaborative phase, beta4 clean-audit
accuracy was already +2.45pp on average and higher for all four clients, so the reverse signal began
in local pretraining rather than an AsymHFL amplification failure. Configs differed only by name and
beta; both have rounds 0--9 and four checkpoints. Verdict: `SCREEN NO-GO`; do not promote to 40+40,
Local/Centralized diagnostics, beta/noise sweeps, or a post-hoc inverted paper claim. Evidence:
`deliverables/rahfl_coupling_phase_a_screen_20260824/RESULT_SUMMARY_ZH.md`.

## Latest Topic Reset: Federated Source-Graph Weak Supervision Is the Conditional Primary

After parking corruption/CLE-HFL/PEW-BER as the paper mainline, a benchmark-first screen of recent
federated topics removed heterogeneous LoRA, federated calibration/conformal prediction, open-world
category discovery, incomplete-modality multimodal FL, federated causal overlap recovery and generic
pairwise-risk optimization because direct 2024--2026 methods already cover their central mechanisms.

One candidate survives for a strict theory gate:

```text
Federated Source-Reliability Completion
under Private and Partially Overlapping Labeling Functions
```

Clients hold unlabeled data and private subsets of programmatic labeling functions. They do not send
samples, LF code or per-sample weak-label matrices; they send aggregate LF-pair agreement moments.
The union co-observation graph can contain reliability information absent from every local graph. In
the minimal binary conditionally-independent model, `M_jk = a_j a_k`. Disconnected graphs are not
globally alignable; connected bipartite graphs retain a multiplicative scale ambiguity; a connected
non-bipartite graph plus one orientation anchor can identify source reliability in the ideal model.

This is not a claim that federated weak supervision is new. WSHFL already mines and shares
parameterized candidate LFs, and US20230237321A1 already transfers weak labels through cross-client
sample similarity without sharing LF code. The only possible new core is the identifiability and
recovery of partially overlapping private source reliabilities from aggregate co-observation
statistics. WRENCH and BOXWRENCH provide public weak-supervision tasks and code.

Verdict:

```text
source-graph weak supervision: CONDITIONAL GO FOR THEORY GATE
global survival concordance:   HIGH-RISK BACKUP ONLY
implementation / experiment:   NONE / NONE
commit:                        NONE (user requested no proactive commits)
```

Evidence:

```text
docs/research/status/CCF_B_FEDERATED_TOPIC_RESET_SCREEN_2026_08_17_ZH.md
```

Next action: paper-only impossibility and identifiability audit for the source-graph candidate. It
must handle disconnected/bipartite graphs, abstention, client-dependent LF accuracy, finite-sample
error, natural WRENCH/BOXWRENCH domain partitions and exact collisions with WSHFL/FlyingSquid/the
weak-supervision patent. Do not implement or run experiments before this gate passes.

## Latest Theory Gate: Federated Ordinal Boundary Completion Is NO-GO

The proposed topic on completing missing ordinal severity boundaries across clients has completed
its paper-only audit. The application is real, and the direct Federated Ordinal Learning study
already observes that missing adjacent classes reduce the information needed to learn ordinal
boundaries. However, the candidate does not create a new identifiable FL object.

For homogeneous models, every cumulative cutpoint risk remains a weighted sum of client risks, so
one-step FedSGD exactly recovers the centralized gradient even when each client's local cutpoint
optimum diverges. Multiple-local-step bias is ordinary client drift; cutpoint-wise prevalence
correction collides with FedLC, while class deficiency is covered by FedGELA and SSDI. The proposed
boundary-complementarity statistic reduces to `Var_w(p_i,k)` via the law of total variance.

For heterogeneous models, client score/cutpoint scales are incomparable without shared parameters,
common inputs or a shared proxy, and label counts plus boundary support cannot identify the missing
input-label mapping. Adding common inputs reduces the protocol to FedMD/FedH2L plus ordinal
encoding; adding a shared proxy reduces it to FedeKD-style reliable bidirectional distillation.
FedeKD already evaluates model/data heterogeneity on RetinaMNIST and Diabetic Retinopathy.

Verdict:

```text
ordinal severity task / missing-boundary phenomenon: REAL / ALREADY DOCUMENTED
client-boundary observability diagnostic:            VALID BUT SIMPLE
homogeneous FL method novelty:                        NO-GO
heterogeneous FL without bridge:                      UNIDENTIFIABLE
heterogeneous FL with bridge:                         KD COMPOSITION / NO-GO
implementation / experiment / commit:                 NONE / NONE / NONE
```

Evidence:

```text
docs/archive/methods/FEDERATED_ORDINAL_BOUNDARY_COMPLETION_AUDIT_2026_08_17_ZH.md
```

Do not implement boundary-support weighting, cutpoint teacher routing, ordinal public-logit
distillation or a shared ordinal proxy. Select a new problem only after establishing a target that
is neither additively solved by standard federated gradients nor bridged by existing heterogeneous
KD.

## Latest New-Topic Screen: No Paper-GO Candidate Outside Model-Heterogeneous HFL Yet

The user has explicitly allowed the new paper topic to leave model-heterogeneous HFL and rejected
any CLE-HFL + PEW/BER benchmark fallback. A final problem-first screen therefore removed corruption,
public logits, heterogeneous backbones and KD from the required core.

Federated backward-compatible representations are paper `NO-GO`: centralized BCT, BC-Aligner and
multi-generation compatibility already cover the mechanism, and federating their local loss does
not create a new FL object. Client optimizer autonomy is `NO-GO` because Federated Blended
Optimization already treats heterogeneous local optimizers as black boxes, while FedPM/FedPAC
directly cover incompatible preconditioner geometries. Cross-client deduplication is `NO-GO` due
EP-MPD and FedRW. Heterogeneous label maturity/delay is real but currently reduces to local
inverse-maturity correction plus delayed-feedback learning and ordinary aggregation, so it is also
method `NO-GO`.

The final theory gate also rejected repeated adaptive use of client-private validation sets. Across
rounds, server proposals do depend on reused audit responses, so the statistical problem is real.
Define the useful diagnostic:

```text
FVI_i(T) = I(H_T ; A_i | training transcript and other clients' audits).
```

For sub-Gaussian loss, the expected client validation optimism is controlled by
`sqrt(2 sigma^2 FVI_i(T) / m_i)`. However, Reusable Holdout/Thresholdout already permits arbitrarily
adaptive queries based on the entire previous transcript. Running one mechanism per client remains
valid when the server uses other clients' answers to construct later proposals: relative to a target
audit set, the rest is adaptive post-processing and mechanisms on disjoint data. Thus the per-client
FVI vector adds reporting granularity but no new joint validity mechanism. DP-Hype additionally
covers local evaluation, noising, secure aggregation and private federated selection.

Verdict:

```text
federated adaptive validation-reuse problem: REAL
per-client FVI diagnostic:                   VALID BUT NOT A NEW METHOD
paper core / implementation / experiment:    NO-GO / NONE
all PEW/BER or model-HFL fallbacks:           EXCLUDED BY USER
```

Evidence:

```text
docs/research/status/FEDERATED_NEW_TOPIC_FINAL_SCREEN_2026_08_17_ZH.md
```

Next action: obtain the user's boundary decision: either the new topic may leave federated learning
entirely, or it must remain in FL while accepting new real data / observable side information.
Do not change code, run experiments or commit.

## Latest Theory Gate: Client Architecture Transition Core Is NO-GO

The conditional primary on personalized knowledge continuity after the same client changes
architecture has completed its paper-only identifiability/nontriviality audit. If the old client
function is queryable on the relevant input distribution, ordinary local old-to-new distillation is
the optimal projection into the new hypothesis class for the pure preservation objective; no
federated transcript can lower that objective. If old outputs are unavailable or observed only on a
finite transfer set, two worlds can have identical local observations and federated transcripts but
opposite old personalized functions on an unqueried positive-mass region, so uniform recovery is
impossible.

Adding current federated knowledge changes the objective from preservation to collaborative task
improvement and reduces to historical-local + current-global dual/multi-teacher distillation. This
directly collides with pFedSD/comprehensive KD, pFedKT and FedPSD, while AdaptFL and FedKDNAS already
cover real-time resource changes, changing client architectures and heterogeneous KD. Cross-
architecture KD and adaptive dual-teacher CAKD cover the remaining transfer mechanism.

Verdict:

```text
architecture-transition scenario reality:       GO
ATR / FTG as controlled evaluation metrics:      GO
FL-specific identifiable preservation target:   NO-GO
non-dual-teacher method object:                  NO-GO
CCF-B core / implementation / experiment:        NO-GO / NONE
```

Evidence:

```text
docs/archive/methods/CLIENT_ARCHITECTURE_TRANSITION_IDENTIFIABILITY_AUDIT_2026_08_17_ZH.md
```

Do not implement an architecture-switch runner, old+global teacher loss, architecture-specific
temperature/projector/gate, or run local/OpenI experiments. The only survivor from the prior screen
is the architecture-conditioned collaboration-benefit diagnostic, and it remains a benchmark-only
conditional route rather than a CCF-B method paper.

## Latest Topic Convergence: One Conditional Primary and One Diagnostic Backup

A top-down screen compared six non-corruption topics against current literature, frozen negative
routes, existing infrastructure and a one-to-two-month CCF-B window. Proxy-data coverage, receiver
capacity/learnability, heterogeneous label spaces and model-heterogeneous federated unlearning all
have direct collisions and are paper `NO-GO` for this project.

The only conditional primary is personalized knowledge continuity when the *same client* changes
architecture during federation. AdaptFL already assigns resource-specific architectures under
round-wise resource changes, and FedKDNAS selects an architecture every round while exchanging
public-reference logits. Therefore dynamic architecture selection itself is not new. The remaining
question is whether architecture-switch loss and personalized-knowledge continuity admit an
FL-specific object that beats the strongest local old-to-new KD and cannot reduce to an old-local +
current-federation dual teacher. This has not passed its theory gate and must not be implemented yet.

The sole backup is a diagnostic/benchmark on architecture-conditioned collaboration benefit, using
matched data twins across backbones to separate architecture from data. Collaborative fairness,
individual benefit over Local and low-end-device inclusiveness already exist, so this is not yet a
method-paper contribution and is retained only if the user accepts a diagnostic paper.

Evidence:

```text
docs/research/status/CCF_B_TOPIC_CONVERGENCE_MATRIX_2026_08_17_ZH.md
```

Next action: paper-only identifiability/nontriviality audit for architecture transitions. Do not
change code, run experiments, revive frozen routers, or commit for this screen.

## Latest Theory Gate: Missed-Knowledge Path Is Valid but Paper NO-GO

The ordered replay gate for a returning model-heterogeneous client has completed without code or
experiments. A minimal non-injective student-response construction proves that endpoint-only and
mean-teacher distillation can remain at an information-degenerate point, while chronological and
reverse replay select opposite final branches. Thus order can contain information absent from the
final aggregate teacher. A standard contraction bound also relates recovery error to initial
staleness, weighted teacher-path compression error, and local optimization/capacity error.

The paper gate nevertheless fails. Pro-KD, progressive distillation's implicit-curriculum theory,
Continuation-KD and curriculum extraction directly cover the advantage of intermediate teacher
checkpoints over a final teacher. FAPD already brings progressive, capacity-aware distillation into
federated learning; FedGKD uses historical global teachers; FedLFH uses client historical
trajectories. Path-variation recovery bounds are standard dynamic-regret/tracking results. The exact
returning-client/public-logit protocol was not found verbatim, but its method is an obvious
composition rather than a new CCF-B core.

Verdict:

```text
ordered-path mathematical phenomenon: GO
new KD principle:                     NO-GO
CCF-B rejoining-client core method:   NO-GO
implementation / experiment:          NONE
```

Evidence:

```text
docs/research/status/REJOINING_HETERO_LOGIT_RECOVERY_AUDIT_2026_08_17_ZH.md
docs/archive/methods/MISSED_KNOWLEDGE_PATH_THEORY_GATE_2026_08_17_ZH.md
```

Keep data-corruption x model-HFL parked as the current submission mainline, preserve all existing
RAHFL/CLE-HFL/PEW+BER assets, and do not implement or tune the missed-path candidate.

## Latest New-Topic Audit: Public-Logit Privacy Is Real but Not a New Core

The public-logit privacy route was audited against the exact current protocol. Every round exposes
per-client, per-public-sample 10-way probability vectors before receiver-specific AsymHFL routing.
The resulting transcript can support label-distribution and membership inference; this is a real
risk, not merely a paper story.

It does not pass the novelty and feasibility gates. PoPETs 2025 already attacks public-dataset-assisted
federated distillation, including CIFAR-10 private / CIFAR-100 public. USENIX Security 2024 directly
studies architecture-dependent privacy leakage. FedMD-NFDP, one-shot/noisy federated KD,
Selective-FD, CoFedMID and secure heterogeneous FD aggregation surround the available defenses.
Under formal DP, the worst-case sensitivity of probability releases is architecture-independent;
under empirical architecture-aware perturbation, protection is attack-specific and provides no
worst-case privacy claim. Privacy-budgeted teacher-query routing is the only less-directly occupied
formulation, but query utility is counterfactual and fewer deterministic queries do not themselves
provide DP.

Verdict:

```text
privacy risk in the current communication:              REAL
heterogeneous leakage as an empirical audit:             VALID BUT WEAK
architecture-aware logit release as a paper core:        NO-GO
privacy-budgeted teacher-query routing:                  CONDITIONAL, THEORY INCOMPLETE
implementation / local attacks / OpenI:                  NONE
```

Evidence:

```text
docs/archive/methods/MODEL_HETERO_PUBLIC_LOGIT_PRIVACY_AUDIT_2026_08_17_ZH.md
```

Do not implement per-backbone noise, temperature, top-k, argmax, query dropping or an attack-AUC
controller. The next independent topic screen should examine late-joining/intermittent heterogeneous
clients and must first distinguish itself from asynchronous FL, stale-update revival and continual HFL.

## Latest Theory Audit: Compound Degradation Does Not Rescue the Current Paper Core

The compound-degradation route was formalized without corruption labels, quality metadata, client
filtering or sample downweighting. Its sole candidate, Federated Compound Interaction Risk (FCIR),
measures whether adding an augmentation after an existing augmentation history creates positive
loss synergy beyond the same augmentation's singleton damage. A valid telescoping bound controls
long compositions when every prefix interaction is bounded.

This is only a closed augmentation-library result. It cannot cover unknown real corruptions without
a label-preserving augmentation-to-corruption coverage and loss-smoothness assumption. The local
behavior is surrounded by AugMix, AugMax, CoCor and representation straightening; communicating
class-by-augmentation interaction tables reduces to FedAvP-style shared augmentation policy. The
route also has no mechanism that increases training quality in the private weak class-environment
cells responsible for BER's WCCA/CFG gains.

Verdict:

```text
compound degradation as an evaluation extension: CONDITIONAL GO
compound degradation as a new problem claim:     NO-GO
FCIR closed-library mathematics:                  VALID BUT WEAK
FCIR as a standalone paper core:                  NO-GO
implementation / local / OpenI:                   NONE
```

Evidence:

```text
docs/archive/methods/COMPOUND_DEGRADATION_INVARIANT_KNOWLEDGE_AUDIT_2026_08_17_ZH.md
```

Do not implement FCIR, shared augmentation-interaction communication, or a renamed compound
AugMix module. If the user still rejects a taxonomy-assisted benchmark paper, stop the data
corruption x model-HFL method mainline and select a new topic with an observable target and a
short, existing-data experimental path.

## Latest Theory Audit: Latent Degradation Risk Identifiability Is Not a New Paper Core

The proposed paper route formalized class-conditional latent degradation risk as
`r_c = Pi_c rho_c`, constructed observationally equivalent worlds with different hidden
worst-cell risks, and derived the known-mixture rank condition, transcript invariance and the
additional confounding caused by model-specific risk vectors. These statements are mathematically
valid and explain why ordinary public-logit communication cannot replace missing environment side
information.

They do not pass the novelty gate. Unknown-group worst risk is already covered by fairness without
demographics and its federated version; aggregate-statistic bounds and partial identification have
direct precedents; multi-mixture component identifiability is covered by mutual-contamination theory.
The proposed heterogeneous shortcut-transfer estimand also collides with knowledge-distillation
mechanism-transfer work and AugHFL/RAHFL's corrupted-knowledge motivation.

Verdict:

```text
latent degradation risk as an internal diagnostic: VALID
P0 identifiability as a standalone paper core:      NO-GO
P1 shortcut transfer as a standalone paper core:    NO-GO
implementation / offline audit / experiment:        NONE
```

Evidence:

```text
docs/archive/methods/LATENT_DEGRADATION_RISK_IDENTIFIABILITY_AUDIT_2026_08_17_ZH.md
```

Do not implement identified-set optimization, communication difference-in-differences, or another
hidden-environment proxy. The next strategic choice must be a transparent PEW+BER empirical paper,
a real-metadata/observable-fault problem change, or leaving the data-corruption x model-HFL mainline.

## Latest Theory Audit: Safe Model-Heterogeneous FTTA Is Paper NO-GO

The proposed safe collaborative continual test-time adaptation setting was audited before any
implementation. With no target task labels, two worlds can share identical source data, target
images, public responses, model outputs and communication histories while reversing whether a
collaboration helps or harms. Therefore the sign of target collaboration harm is not identifiable
from the current observation set. Source audit certifies source behavior only.

The usual repairs do not pass the paper gate. Covariate shift plus overlap is mathematically
sufficient but untestable and implausible for arbitrary information-destroying corruptions. A
calibrated unlabeled loss proxy assumes the key risk relation and collides with AETTA and NeurIPS
2025 TTA risk monitoring. Collaborative and continual FTTA are already covered by FedTHE, ATP,
FedTSA, CoLA, FedCTTA and Latte; adding arbitrary backbones is an implementation intersection,
not yet a new identifiable object.

Verdict:

```text
fully-unlabeled safe collaborative MH-CTTA: PAPER NO-GO
implementation / local smoke / OpenI:       NONE
sparse unbiased delayed target labels:      CONDITIONAL REFRAME ONLY
```

Full report:

```text
docs/archive/methods/SAFE_MODEL_HETERO_FTTA_IDENTIFIABILITY_AUDIT_2026_08_17_ZH.md
```

The delayed-label route requires shadow candidates or randomized actions to estimate
counterfactual collaboration harm and changes the task to delayed-supervision online/continual
learning. Do not implement it unless the user explicitly accepts that problem change.

## Latest Candidate Audit: FCNT / FPER / FRT Do Not Enter Implementation

Three explicit side-information routes were formalized and checked against both frozen project
evidence and primary literature without changing training code or running experiments:

```text
FCNT: continuous nuisance coordinates + class-conditional federated OT
FPER: paired restoration intervention + degradation-effect risk
FRT:  public multi-view response tensor factorization
```

None passes the current core-method gate. FCNT is surrounded by CCDB/FG-CCDB,
class-conditioned Wasserstein DRO, FedWaD/FedDaDiL and SLOT-Align; a Wasserstein barycenter
also provides no lower mass bound for latent weak cells. FPER requires an unverifiable
label-preserving nuisance-removal oracle, does not prevent minority-cell dilution and collides with
counterfactual invariance/generation plus the frozen C3R/FedCISA reasoning. FRT lacks an identifiable
semantic/shortcut decomposition, cannot connect public responses to private weak-cell mass, and
repeats the public multi-view/shared-subspace risks already rejected by CCAD/IRD/FedCIS/EBST.

Verdict:

```text
FCNT current-protocol core:       NO-GO
FCNT with explicit real metadata: CONDITIONAL REFRAME ONLY
FPER observed-only core:          NO-GO; paired/clean ORACLE ONLY
FRT communication:               NO-GO
implementation / experiment:     NONE
```

Full report:

```text
docs/archive/methods/FCNT_FPER_FRT_THEORY_NOVELTY_AUDIT_2026_08_17_ZH.md
```

The next action is a strategic route choice, not a fourth taxonomy-free module: explicitly add a
realistically available side-information assumption, retain PEW+BER for a conservative empirical
paper, or stop CLE as the method-paper mainline. Do not implement or run any of these three candidates
before the user selects the route.

## Latest Paper-Claim Audit: PEW+BER Is Baseline GO, Core-Method NO-GO

The implementation, existing evidence and external literature were audited without
running experiments. The current exact method is a six-way public synthetic
corruption-family classifier followed by class x predicted-environment reweighted
ERM. It does not read private operator metadata for training, but it is not
taxonomy-free. BER is neither GroupDRO nor CVaR; it is a support-shrunk grouped
average risk and does not define a new robust-optimization principle.

External collision is material: Corrupted CIFAR-10 in Learning from Failure
(NeurIPS 2020) already couples labels with corruption types; SSA/BARACK already
use predicted spurious/group attributes for downstream robust training; CCDB and
FG-CCDB directly study class-conditional distribution balancing. CLE-HFL can be
positioned only as a controlled model-heterogeneous federated extension with
client-specific mappings and operator-cell evaluation, not as the first
class-corruption entanglement problem.

Verdict:

```text
PEW+BER empirical mechanism on fixed CLE:       GO
PEW+BER taxonomy-assisted diagnostic baseline:  GO
PEW+BER as the sole paper-level core method:     NO-GO
CLE-HFL as a federated benchmark extension:      CONDITIONAL GO
```

Do not spend the next stage merely adding exact PEW+BER seeds/rounds or renaming
the scenario. Preserve PEW+BER as the positive anchor; a new candidate must add
an FL-specific mathematical object and pass paper-level collision checks before
implementation. Full evidence:

```text
docs/research/status/PEW_BER_PAPER_CLAIM_AUDIT_2026_08_16_ZH.md
```

## Current Objective

The user has rejected all CLE-HFL/PEW+BER benchmark fallbacks and allowed the submission topic to
leave model-heterogeneous HFL while clarifying that heterogeneous models remain allowed and may be
desirable. Model heterogeneity is therefore a permitted protocol condition, not a required novelty
claim. The first broader FL screen and the subsequent Federated Ordinal Boundary Completion audit
have no paper-GO survivor. Preserve all prior assets and negative evidence, but do not force the
topic back into corruption, KD, public logits or architecture heterogeneity. There is currently no
active method candidate.

## Latest Theory Result: LCC - NO-GO Before Implementation

Latent Correction Conflict (LCC) was formalized as class-conditioned
per-sample last-layer gradient grouping followed by a minimum-norm common
descent update. It requires no environment labels and differs from the frozen
project methods, but it does not pass the external novelty gate:

```text
gradient clustering -> latent robust groups   GRASP collision
minimum-norm common descent                   MGDA/CAGrad collision
last-layer gradient KNN soft neighborhoods    GoG (KDD 2025) collision
```

Verdict: `THEORY NO-GO`. Do not implement LCC, change its clustering/graph, or
spend GPU/OpenI time on it. Evidence:

```text
docs/archive/methods/LCC_NOVELTY_AUDIT_ZH.md
```

## Taxonomy-Free Identifiability Boundary

Using client identity as an unlabeled mixture view was also checked before
turning it into a communication module. For class `c`, observable client risks
satisfy `r_c = Pi_c rho_c`; centered client contrasts can identify at most
`K-1` environment-risk directions. In the frozen four-client CLE mapping, the
effective family-contrast ranks by class are:

```text
class: 0 1 2 3 4 5 6 7 8 9
rank:  1 2 1 3 1 1 2 2 0 2
```

Only class 3 has full four-family contrast coverage; class 8 has none. Model
heterogeneity further confounds client-risk differences. Pure client-class
variance/DRO therefore cannot replace BER with a clean guarantee and must not
be promoted as the next communication innovation. Evidence:

```text
docs/research/status/TAXONOMY_FREE_IDENTIFIABILITY_2026_08_11_ZH.md
scripts/audit_mixture_contrast_identifiability.py
```

## Latest Result: CRSR Audit 0 - NO-GO

Class-conditional Residual Spectral Risk (CRSR) used only fit-internal labels
and predictions:

```text
r(x,c)   = softmax(f(x)) - one_hot(c)
Sigma_c  = Cov(r | y=c)
S_c      = sqrt(lambda_max(Sigma_c))
L_CRSR   = class-balanced CE + 2.0 * mean_c S_c
```

The frozen local Audit 0 completed on client 1/ResNet12 and client
3/MobileNetV2 without reading private audit or final test. Operator IDs were
used only for post-hoc cell evaluation. Independent recomputation matched the
script: G0--G3 passed; G4--G6 failed.

```text
median top share             0.752411  PASS
median direction cosine      0.975226  PASS
median transfer advantage    0.639379  PASS
median spectral cell rho     0.069658  FAIL (< 0.25)
median advantage vs baseline -0.903889 FAIL (< 0.02)
mean CE delta, clients 1/3   +0.006485 / +0.000360  FAIL
worst-cell CE delta          +0.101343 / -0.045068  FAIL
verdict                      NO-GO
```

Interpretation: the class-residual spectrum is active, stable across disjoint
splits, and nonredundant with sample CE/Brier, but it does not consistently
identify weak class-operator cells and its optimization is not mean-risk
noninferior. Freeze CRSR. Do not tune its weight, support thresholds, probe
size, power-iteration count, or gates; do not connect it to the runner or run
12/40 rounds.

Evidence and retained isolated implementation:

```text
docs/experiments/archive/CLASS_RESIDUAL_SPECTRAL_RISK_AUDIT_ZH.md
outputs/class_residual_spectral_risk_audit0/result.json
outputs/class_residual_spectral_risk_audit0/signals.npz
fedprime/methods/class_residual_spectral_risk.py
scripts/audit_class_residual_spectral_risk.py
tests/test_class_residual_spectral_risk.py
```

## Current Formal Positive Result

The selected local path remains calibrated hard PEW + hard BER; the legacy
strict three-seed positive package also included the then-active CDep term.
On fixed CLE-HFL v2 `seed0_split0`, that package's matched 12-round
training-seed 0/1/2 deltas versus AugMix/JSD/DCL control were:

```text
mean Avg +4.5880, Worst +4.2169, WCCA +5.5500, CFG -6.7150
```

The 40-round training-seed-0 durability result also passed all frozen gates.
These results establish the empirical target to preserve, not a defense of
PEW's five-family taxonomy.

## Other Frozen Recent Negatives

```text
Multi-label PEW + Soft-BER: NO-GO (0/4 matched last-five gates)
PIE/MPIE: NO-GO; do not implement PBR
C3R: NO-GO; do not implement its training loss
CRSR: NO-GO; stable geometry but invalid weak-cell surrogate
```

Also obey the permanent frozen-negative list in `AGENTS.md`; do not revive
FedCIS, continuous-witness, IRD/PCCD, or communication methods already archived
as negative. A new object must additionally be distinguished from
GroupDRO/CVaR and CCAD instead of merely renaming their objective.

## Next Action

Review and freeze the KT-pFL/FCCL four-arm protocol and its per-base Formal gates. If the user
authorizes the next platform step, run benchmark only to estimate runtime and cost. Do not treat
smoke/benchmark as evidence and do not start Formal, multi-seed, map2, or 40-round work without
separate explicit authorization.
