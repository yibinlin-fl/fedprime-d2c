# CLE-HFL五图与RAHFL图形语言审计（2026-10-25）

## 结论

五张正文图在数量上足够。问题不在“图太少”，而在当前三张概念图的角色重复，且缺少一张真正
展示PEW+BER执行机制的技术图。继续增加流程框并不会更像论文；应重分配五张图的科学职责。

## RAHFL带来的启发

核对RAHFL正文及补充材料后，其主要视觉不是装饰性信息图，而是：

1. 用真实图像、客户端架构和损坏样例解释研究场景；
2. 用紧凑柱状图呈现关键定量差异；
3. 用密集但可追踪的模块图解释本地与协作阶段；
4. 用传统收敛曲线和数据分布图承载证据。

它的“手搓感”来自内容与实验对象一一对应、字号克制、留白少、视觉语法稳定，而不是来自复杂配色。

## 当前V0.8五图审计

| 当前图 | 判断 | 投稿版处理 |
|---|---|---|
| Fig.1 evidence chain | 像汇报时间线，标题大、留白多 | 与paired DSA场景示意合并；加入真实source/operator小图和binding方向 |
| Fig.2 DSA/proxy boundary | 科学内容有效，但大色块与口号感偏强 | 保留逻辑，改为小型构造例+数学箭头，减少装饰框 |
| Fig.3 local-first | 已是数据图，但seed-0右侧比例容易被读成因果分解 | seeds 1/2完成后改成seed-level dot/forest plot，直接画`Delta_HFL/Delta_Local/Delta_Comm` |
| Fig.4 Axis-I | 当前最接近会议论文结果图 | 压缩版式；优先改成DSA--utility双轴或shortcut--utility frontier |
| Fig.5 protocol map | 更像项目管理图，不值得长期占正文图位 | 移至附录或改成紧凑协议表 |

## 推荐投稿版五图

1. **CLE场景与paired DSA**：同一source跨operator；显示客户端binding方向和DSA读数。
2. **PEW+BER方法图**：公共合成数据训练PEW、私有样本得到粗环境、类别内支持统计、BER权重、
   本地更新、通信协议保持不变。该图目前缺失，优先级最高。
3. **local-first多seed图**：三个training seeds的HFL、Local、communication add-on及区间。
4. **Axis-I目标比较图**：ERM/CVaR/PEW+GroupDRO/PEW+BER的DSA与utility权衡。
5. **剂量响应与中等难度上下文图**：`gamma=0/0.5/0.9`响应，以及moderate场景的关键utility/DSA。

## 去除“AI味”的硬约束

- 正文数据图只从冻结CSV/JSON生成，禁止手工填数；
- 避免渐变、圆角卡片墙、发光、巨型编号和宣传式短句；
- 单图最多一个强调色，其他方法使用灰阶或低饱和色；
- 直接标注关键值和置信区间，减少图例来回查找；
- 方法图使用统一线宽、直角/轻圆角、有限图标，并显示真实数学对象；
- 每张图只回答一个RQ，caption写清统计单位、seed和协议边界。

当前不应立刻重画全部五张。先完成local-first多seed，再冻结Fig.3；PEW+BER方法图可以提前重画，
因为其数据依赖最少、也是当前最明显的视觉缺口。

