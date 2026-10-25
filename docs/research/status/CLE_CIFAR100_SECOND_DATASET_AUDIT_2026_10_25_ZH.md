# CIFAR-100第二private dataset当前代码审计（2026-10-25）

## 审计结论

协议的科学角色成立，但旧OpenI入口的失败恢复不合格。现已完成兼容修复和聚焦测试，并随
`a3b7841`推送至`origin/main`；尚未确认OpenI benchmark已启动，Formal仍需benchmark成本审计。

## 冻结协议

```text
private task: CIFAR-100, 100 classes
public PEW carrier: CIFAR-10, public synthetic corruption supervision
clients/models: 4 heterogeneous clients
private partition: Dirichlet alpha=0.5
private corruption rate: 1.0
conditions: gamma=0 and gamma=0.9
arms: gamma0 ERM / gamma0.9 ERM / gamma0.9 CVaR-DRO / gamma0.9 PEW+BER
Formal budget: 40 rounds x 16 local batches/client/round
training seeds: 0/1/2, matched within seed
DSA grid: 1000 semantic sources x 15 operators, severity=3
```

这是第二受控图像任务的**机制外部复现**，不是moderate Axis-II效用表。它允许回答CLE/DSA/BER是否
只出现在CIFAR-10 private task；不能回答真实医院、汽车或开放世界corruption。

## 数据与信息边界

- 输入包大小：`681493067` bytes；
- SHA256：`650C3363B708554EB164BDEB94A3566B98CADBFF697FC093B9DE7F061C7DA708`；
- 包内private CIFAR-100、public CIFAR-10，避免主实验CIFAR-100 public carrier与private source重叠；
- PEW只用CIFAR-10公共图像的合成corruption标签训练，不读取CIFAR-10语义标签或CIFAR-100任务标签；
- gamma0/gamma.9共享source indices、labels、severity draws、fit/audit split和初始化；该旧strong包中
  所有private训练样本均被corrupt，因此没有单独保存moderate协议才需要的corruption mask；
- 最终test标签只供报告与DSA计算，不参与训练、routing、选择或调参。

## 本轮修复

1. 启动前锁定输入包精确大小和SHA256；
2. 要求OpenI output context可用，否则训练前失败；
3. PEW训练前先审计immutable数据角色与paired输入；
4. 数据审计新增gamma条件语义配对、public archive、initial states及PEW annotation/checkpoint哈希；
5. 每个seed四臂训练完成后，先上传含checkpoint的recovery包，再执行audit和DSA分析；
6. audit/analysis失败时仍上传recovery/status并保留非零退出码；
7. 分析设备改为`auto`，GPU lease结束时可回退CPU；
8. benchmark永远不被标成科学证据。

## 验证

- 输入压缩包大小与SHA256现场复核通过；
- 直接从681MB压缩包流式读取数组复核：12/12组`labels/source_indices/severity`跨gamma完全一致，
  1000个评价source、15个operator，四客户端所有private训练severity均大于0；
- 四臂配置审计确认private/public角色、40轮、16 batches、loader seed、初始权重和gamma条件；
- 入口失败恢复、Formal锁、精确输入校验与已有moderate恢复测试合计`15/15`通过；
- 四个核心脚本通过Python编译检查。

尚未运行新的端到端GPU smoke或OpenI benchmark。修复只说明“入口现在具备运行条件”，不产生任何
CIFAR-100科学结论。

