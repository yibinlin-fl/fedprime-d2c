# CLE-HFL LaTeX V0.2

V0.2基于已归档的LaTeX V0.1创建，没有覆盖V0.1。它落实网页端第一轮审稿中经项目证据核验后
接受的修改，并修复V0.1 PDF视觉审计发现的Table 3可读性、Oracle Table 4浮动和占位caption。

```text
main.tex                         当前V0.2 LaTeX正文
references.bib                   与V0.1相同的18条引用数据库
figures/                         V0.2论文图
WEB_REVIEW_ADJUDICATION_ZH.md    网页端意见的接受/可选/拒绝裁决
WEB_REVIEW_RAW_ZH.md             网页端第一轮审稿原文
qa/                              图形视觉检查预览
```

科学边界不变：operator-level CLE-v2/DSA、family-level PEW+BER、CDep缺席、FedDF utility
失败、CVaR逐客户端例外、JTT screen-only、source-bootstrap不覆盖training randomness。所有
`[EVIDENCE NEEDED]`继续保留；没有新增或运行训练实验。

本机没有TeX发行版，因此V0.2需上传Overleaf重新编译。编译后PDF放入：

```text
C:\Users\asus\Desktop\FedPRIME-D2C\local_runs\paper_compiled_returns\v0_2
```

Overleaf根目录上传包：

```text
deliverables/CLE_HFL_LATEX_V0_2_OVERLEAF_20260916.zip
```

V0.2已于2026-09-16在Overleaf编译并归档为：

```text
CLE_HFL_LATEX_V0_2_OVERLEAF_COMPILED.pdf
```

逐页终检发现第4页过宽公式和第9页Oracle表可读性问题；两处均在独立保留的V0.3源码中修复。
