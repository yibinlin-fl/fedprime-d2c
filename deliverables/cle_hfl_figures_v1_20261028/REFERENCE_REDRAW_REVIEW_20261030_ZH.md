# 图1/2参考原文后的TikZ重绘记录

日期：2026-10-30。V0.11保持不变，两张新图为独立候选，等待作者判断视觉风格。
这次直接打开原论文框架图，不仅参考前一次文字审计。没有复制他人图形、论文数值或模型结构。

## 实际查看的参考图

| 论文/出处 | 原图位置 | 本次采用的组织原则 | 不照搬的部分 |
|---|---|---|---|
| RAHFL，TPAMI 2025，[原文](https://arxiv.org/abs/2503.09206) | PDF第5页Fig.3 | 明确区分私有本地学习、公共数据上的协作；图片→模型→预测→损失均可追踪 | DCL、AsymHFL选择矩阵、立体logit柱，不属于PEW/BER的新机制 |
| AugHFL，ICCV 2023，[官方页面](https://openaccess.thecvf.com/content/ICCV2023/html/Fang_Robust_Heterogeneous_Federated_Learning_under_Data_Corruption_ICCV_2023_paper.html) | PDF第4页Fig.3 | 数据角色和客户端阶段清楚，输入与损失的连接有明确语义 | 三客户端完整重复、大块多色背景，不为制造复杂度而加入 |
| FCCL，CVPR 2022，[原文](https://openaccess.thecvf.com/content/CVPR2022/papers/Huang_Learn_From_Others_and_Be_Yourself_in_Heterogeneous_Federated_Learning_CVPR_2022_paper.pdf) | PDF第4页Fig.2 | 区分冻结模型与优化模型，并区分前向路径与损失更新 | 相关矩阵和多个教师分支，不属于本方法 |

RAHFL和AugHFL的配图本身并非“只有一个颜色的极简图”。本次借鉴的是计算对象与箭头的一致性、
研究对象的具体性及紧凑层级，不能把“白底少颜色”本身当作顶会质量标准。

本机查阅路径：

```text
D:\googleDownload\01_联邦学习研究\抗数据损坏与频域论文\RAHFL增强流派\2503.09206v1.pdf
D:\googleDownload\01_联邦学习研究\抗数据损坏与频域论文\RAHFL增强流派\AugHFL.pdf
D:\googleDownload\01_联邦学习研究\抗数据损坏与频域论文\RAHFL增强流派\Learn_From_Others_and_Be_Yourself_in_Heterogeneous_Federated_Learning_CVPR_2022_paper.pdf
```

上述页面均以Poppler实际渲染并打开；预览在local_runs/figure_reference_review_20261030/。
另检索过FedTGP，但AAAI下载失败，未将其未看见的框架图作为本轮视觉依据。

## 两张新候选

- 图1：`figures/tikz_cle_paired_dsa_v7.tex/.pdf/.png/.svg/.json`
- 图2：`figures/tikz_pew_ber_execution_v7.tex/.pdf/.png/.svg/.json`

改用TikZ确定物理坐标，字体与LaTeX正文使用同一Latin Modern家族；公式保留为矢量。
文字以黑色为主，只用低饱和蓝色细线强调BER/梯度路径，没有阴影、渐变或装饰神经网络图标。
不是用图像模型重新生成论文图；真实缩略图保持原像素。

图1保留真实的client0/1三类别binding示例、四张同源severity3的operator图像、共享模型和
DSA对比定义。删去彩色标签墙、重复说明块和横幅，保留C=1和合法target pair条件。

图2保留公共训练/冻结、私有分组与task loss、BER聚合及既有通信接口。私有图像分别进入
冻结PEW和任务模型；BER消费分组权重和损失，梯度仅回到任务模型。公共预测通信另列一行，
不再呈现“BER处理图片→模型→上传私有logits”的含混链条。图中省略辅助severity head，图注注明。
公共合成小图仍为示意，不冒充正式训练日志或定量证据。

## 修复历史（全部保留）

图2：v4首轮存在长公式挤占模块和标签邻近问题；v5缩短模块内公式、分开汇合路径；v6收敛
输出标签并改为直角任务模型；v7明确节点尺寸与标签间隔。共三轮修复后停止。
图1：v5首轮有过长标题与模型图元变形；v6简化标题、修正模型边界及类别标签间距；v7明确
节点尺寸并调整评分输出空间。全部旧Python图和TikZ中间PDF/PNG/TeX均保留。

## 核验结果

1. 两个候选分别通过pdfLaTeX编译；未发现Overfull或未定义命令。
2. 技能附带TikZ检查均为0 CRITICAL。薄箭头/无圆角警告是本次刻意采用的样式；少量标签偏移
   警告来自面向中文的通用估计，已在实际英文渲染图中检查间隔。
3. 直接读取PDF中图像变换矩阵：图1四图、图2六图全部显示宽高比1.000，矩阵无旋转或镜像。
4. 解码PDF嵌入图像与原始资产逐像素哈希匹配；未通过修复像素“扶正”原物体姿态。
5. SVG经原生PGF/dvisvgm后端导出并嵌入原PNG字节，不依赖外部图片路径。SVG文字为矢量轮廓，
   修改文字/公式时以对应TeX源码为准，不把它称为可直接改文字的SVG。
6. 已打开两张最终完整PNG，并以论文7.06in通栏宽度生成两页局部排版预览、逐页打开检查。
   这不是V0.11全篇的重新分页，当前没有将新图接入论文。

每张JSON记录源文件和产物SHA256、图像物理尺寸、像素一致性。复核脚本：
`verify_tikz_candidates_20261030.py`。编译导出脚本：`build_tikz_candidates_20261030.py`，
默认拒绝覆盖已有版本。图片资产提取：`prepare_tikz_assets_20261030.py`；现有提交资产齐全时，
只编译TeX不再需要重新读取数据包。

## 下一步

先由作者比较图1/2的新旧风格与科学表达，再决定是否接入下一篇论文快照。
图3—5的数据和图形结构继续保留；后续聚焦标题/副标题重复、BER外圈和端点拥挤等细节，
不为了“高级”增加未经测量的分布、曲线、三维柱或装饰图形。
“无AI味”和“顶级”是作者/读者的审美判断，当前结论仅是具体缺陷修复并完成候选交付。
