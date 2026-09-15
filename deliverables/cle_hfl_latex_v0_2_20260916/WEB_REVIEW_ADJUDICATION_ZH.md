# 网页端第一轮审稿意见裁决

Updated: 2026-09-16

输入：

```text
local_runs/paper_web_review_returns/v0_1/网页端回复.md
local_runs/paper_compiled_returns/v0_1/CLE_HFL_LATEX_V0_1_OVERLEAF_20260915.pdf
```

## 必须接受并已落实

1. 摘要、Introduction、contribution与Conclusion中的local-first明确限定为seed-0受控结果。
2. `Local-First Mechanism Attribution`降级为`Local-versus-Communication Formation Analysis`，
   避免将matched factorial写成完整causal mediation。
3. 主文补充CLE-v2的精确operator sampling law。
4. 主文补充pooled DSA的分层平均公式。
5. 明确`m=s+r`是paired estimand的identification decomposition，不是神经网络生成假设。
6. 明确BER不均匀化各组，而是次线性压缩support-induced risk-mass advantage。
7. 图1将`Local shortcut formation`弱化为`Shortcut-prone local training`。
8. 保留FedDF learning-floor fail、utility-gate fail和CVaR逐客户端例外。
9. 保留全部`[EVIDENCE NEEDED]`，不把待补实验写成已有结果。
10. 将结论中的`large`改为更克制的`substantial pooled`。
11. 修复Oracle表的占位caption与跨双栏浮动问题；压缩过宽且不可读的cross-setting表。
12. 将`1e-16`量级结果明确标为numerical identity checks，而非统计精度。

## 可选、暂不实施

1. 将Figure 2移入附录：等目标会议模板和新增结果页数确定后再决定。
2. 进一步压缩Section 4.3、7.1和Discussion：当前先保留理论边界完整性，待正式页数约束后处理。
3. 把Oracle正文压成2--3句：目标模板不足时优先压缩。
4. 正式JTT、partition seed与更多通信底座：只按投稿实验矩阵的条件触发。

## 不接受或需要纠正

1. 第二private dataset不能因审稿建议直接指定为某个数据集；必须先单独冻结协议。CIFAR-100
   目前只是规划中的推荐候选，不是已授权或已运行实验。
2. 不通过新增loss、operator PEW、hierarchical PEW或恢复CDep解决“方法简单”评价。
3. 不删除内部`[EVIDENCE NEEDED]`来制造投稿完成假象；只有正式结果进入后才能替换。
4. 不把source-bootstrap、cross-map或PEW witness LOO写成training-seed、第二数据集或端到端
   taxonomy-stress证据。

## PDF视觉审计新增发现

V0.1共11页，编译完整，无裁切、重叠、黑块或缺失引用。需要修复的主要排版问题是：

- page 8的cross-setting Table 3字体过小；
- Oracle Table 4在References开始后漂移到下一页顶部；
- Table 4 caption仍是`Experimental summary 4.`占位文本；
- page 11参考文献结束后存在较大空白，但venue-neutral内部稿暂不以此判错。

前三项已在V0.2源码中处理；V0.2仍需Overleaf重新编译后逐页复核。
