# CLE-HFL LaTeX V0.9

V0.9于2026-10-26从不可变V0.8快照派生，只回填已审计的local-first training seeds 1/2 Formal。

主要更新：

- Main local-first表加入seeds 0/1/2及mean +/- sample SD；
- 正文围绕DSA主终点报告3/3机制门槛复现；辅助学习下限只保留在内部正式审计报告；
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
deliverables/cle_hfl_latex_v0_9_20261026/CLE_HFL_LATEX_V0_9_DSA_PRIMARY_OVERLEAF_20261026.zip
bytes: 908032
SHA256: D9B2FDEF2C9837A09BBE2D9A2946357CA853E351CFAC89C211B2B2FB49DF75F3
```

ZIP含49个条目，根目录存在`main.tex`，没有`./`前缀或反斜杠路径。
此前`LOCAL_FIRST_MULTISEED`包保留为本次修改前快照，不再用于上传。
