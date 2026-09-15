# CLE-HFL LaTeX会议稿V0.1

本目录由事实审计后的英文会议压缩稿V0.4转换而来，使用与具体venue解耦的双栏`article`模板。
目标会议确定后，只需替换document class、官方style和bibliography style，不应重新生成研究内容。

## 文件

```text
main.tex
references.bib
figures/fig1_cle_hfl_evidence_chain.pdf
figures/fig1_cle_hfl_evidence_chain.svg
figures/fig2_dsa_proxy_boundary.pdf
figures/fig2_dsa_proxy_boundary.svg
qa/fig1_preview.png
qa/fig2_preview.png
```

两张PDF为论文使用的矢量图；SVG用于后续人工编辑；PNG仅用于本地视觉审计。

## 编译

推荐在Overleaf新建空白项目后上传本目录内容，并将编译器设为`pdfLaTeX`。本地具有完整TeX Live时可运行：

```text
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

或：

```text
latexmk -pdf main.tex
```

## 当前边界

- 这是内部会议稿，不是最终投稿版本；
- 作者信息仍为`Anonymous Authors`；
- `[EVIDENCE NEEDED]`仍然保留；
- 当前环境没有LaTeX发行版，因此已完成静态语法、引用、环境配对和图形PDF视觉检查，但尚未在本机执行完整TeX编译；
- 目标会议模板确定后，必须重新进行逐页PDF视觉检查。

## 可重复生成

```text
python scripts/build_cle_hfl_latex.py
python scripts/render_cle_hfl_paper_figures.py
```

生成脚本从仓库中已审计的V0.4读取正文，不读取大型历史日志，也不运行任何训练实验。
