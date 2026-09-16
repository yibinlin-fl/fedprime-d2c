# LaTeX V0.2编译终检

Updated: 2026-09-16

输入PDF：

```text
local_runs/paper_compiled_returns/v0_2/CLE_HFL_LATEX_V0_2_OVERLEAF_20260916.pdf
```

文件大小`463996` bytes，SHA256：
`EDF18CA84509BFB1732389F3E244A81C63DB9C3F1430C0A0CB643D74B55B8357`。

## 通过项

- 共11页，Letter尺寸，PDF可完整解析；
- 无缺页、裁切、重叠、黑块或未解析引用；
- V0.1中Table 3过小、Oracle表漂移至References之后和占位caption均已修复；
- 两张论文图清晰，主结果表、公式、References及内部证据占位边界完整；
- PDF文本检查未发现`[?]`或`??`引用标记。

## 仍需修复

1. 第4页JSD/DSA逻辑式过宽，跨越双栏并挤入页脚；
2. 第9页Oracle表虽不再漂移，但五列缩放后字体仍偏小。

V0.3只修复上述两处排版问题：逻辑式拆为两行；Oracle表保留三列，last-five Avg/Worst
数字移入紧邻正文。所有小数实验数字和18个citation key与V0.2一致。
