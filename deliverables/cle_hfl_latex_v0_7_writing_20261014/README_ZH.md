# CLE-HFL V0.7-writing交付说明

日期：2026-10-14。

本目录是冻结V0.7的独立写作分支。原V0.7源码和上传包没有被覆盖。

## 使用入口

```text
source/main.tex
```

将`source/`中的内容作为Overleaf项目根目录上传即可。当前默认保留
`\draftskeletontrue`，会显示红色待证据标记；它适合内部审阅，不是投稿净稿。

## 本轮目标

- 统一论文定位为CLE-HFL的诊断、归因与缓解研究；
- 收紧首创性、通用性和utility表述；
- 让local-first结论与HFL研究价值同时成立；
- 区分已完成固定预算证据、moderate Axis II前三行和待运行实验；
- 为后续结果回填建立论断—证据台账。
- 用五张可复现的学术图替换旧版模板化视觉；
- 建立RQ1--RQ7实验骨架，逐项标注已完成和pending证据。

## 必读文件

```text
source/CLAIM_EVIDENCE_AUDIT_ZH.md
source/REVISION_LOG_ZH.md
source/EXPERIMENT_TABLE_MATRIX_ZH.md
source/CITATION_AUDIT_ZH.md
source/FIGURE_MANIFEST.md
source/WRITING_STYLE_AUDIT_ZH.md
source/SOURCE_AUDIT_ZH.md
```

本机没有LaTeX编译器，因此本轮完成的是源码静态检查；仍需在Overleaf编译并检查版面、浮动体、
断行、引用和页数。

## Overleaf上传包

```text
CLE_HFL_LATEX_V0_7_WRITING_V3_OVERLEAF_20261014.zip
size: 902524 bytes
SHA256: 4E96FFA45CB5150FDCFFE4B376628C079C3D09353F725C46CB07D2D82F78E14C
```

ZIP内含53个条目，`main.tex`、`references.bib`、正文分节、表格、五张图的三种格式、生成脚本和六份审计/清单文档均已检查存在；无`__pycache__`。

旧`CLE_HFL_LATEX_V0_7_WRITING_OVERLEAF_20261014.zip`和`..._V2_...zip`只保留为历史快照，不再用于Overleaf续写。

Overleaf编译后的PDF请放入已经创建好的目录：

```text
C:\Users\asus\Desktop\FedPRIME-D2C\local_runs\paper_compiled_returns\v0_7_writing_v2
```
