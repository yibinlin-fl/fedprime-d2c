# CLE-HFL LaTeX V0.3

Updated: 2026-09-16

V0.3只修复V0.2编译终检发现的两处排版问题，不改变科学内容、实验数字、引用或证据边界：

1. 将第4页过宽的JSD/DSA逻辑公式拆为两行，避免跨越双栏和页脚；
2. 将Appendix C的Oracle表缩为可读的三列，其余last-five数字移入紧邻正文并全部保留。

V0.2编译终检确认：11页PDF完整，无缺页、裁切、黑块或未解析引用；V0.1的Table 3过小、
Oracle表漂移到References之后及占位caption问题已经修复。V0.3仍需上传Overleaf编译一次，
确认上述两处视觉问题消失。

编译后PDF放入已经创建的目录：

```text
C:\Users\asus\Desktop\FedPRIME-D2C\local_runs\paper_compiled_returns\v0_3
```

所有`[EVIDENCE NEEDED]`、seed-0限定、taxonomy-assisted边界、FedDF utility失败、CVaR
逐客户端例外、JTT screen-only及source-bootstrap条件性均保持不变。
