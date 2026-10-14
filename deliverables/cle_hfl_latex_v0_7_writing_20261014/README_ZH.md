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

## 必读文件

```text
source/CLAIM_EVIDENCE_AUDIT_ZH.md
source/REVISION_LOG_ZH.md
source/EXPERIMENT_TABLE_MATRIX_ZH.md
source/CITATION_AUDIT_ZH.md
```

本机没有LaTeX编译器，因此本轮完成的是源码静态检查；仍需在Overleaf编译并检查版面、浮动体、
断行、引用和页数。

## Overleaf上传包

```text
CLE_HFL_LATEX_V0_7_WRITING_OVERLEAF_20261014.zip
size: 52779 bytes
SHA256: A4CCF6554ABD80DE0DAF47ACA91D3F548AEBCC3F149BB6394DDC4E497F41964C
```

ZIP内含31个条目，`main.tex`、`references.bib`、正文分节、表格、图片和四份审计文档均已检查存在。
