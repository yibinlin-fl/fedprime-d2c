# CLE-HFL LaTeX V0.9

V0.9于2026-10-26从不可变V0.8快照派生，只回填已审计的local-first training seeds 1/2 Formal。

主要更新：

- Main local-first表加入seeds 0/1/2及mean +/- sample SD；
- 明确完整预注册verdict为Seed 0/1 GO、Seed 2 undertrained；
- Abstract、Introduction、Method、Experiments、Discussion和Conclusion同步该边界；
- 删除local-first seeds 1/2的`[EVIDENCE NEEDED]`；
- 将旧seed-0堆叠图改为三seed matched effect与communication-difference图；
- 保留Axis II三行、第二private dataset和taxonomy stress为pending。

当前源码入口：

```text
deliverables/cle_hfl_latex_v0_9_20261026/source/main.tex
```

事实依据：

```text
deliverables/cle_v2_local_first_multiseed_20261026/RESULT_SUMMARY_ZH.md
```

V0.8没有被覆盖。V0.9已通过桌面内置LaTeX编译；尚未完成用户侧Overleaf PDF逐页视觉终检，
因此仍是内部写作稿，不是投稿净稿。

有效Overleaf包：

```text
deliverables/cle_hfl_latex_v0_9_20261026/CLE_HFL_LATEX_V0_9_LOCAL_FIRST_MULTISEED_OVERLEAF_20261026.zip
bytes: 920081
SHA256: 579C17F8F6D24A0DFF9D266C015F537328072FD54C17190AD6CD32484B0FA9F6
```

ZIP含49个条目，根目录存在`main.tex`，没有`./`前缀或反斜杠路径。
