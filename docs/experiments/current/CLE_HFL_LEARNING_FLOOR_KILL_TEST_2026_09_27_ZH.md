# CLE-HFL Local学习水平Kill Test

Updated: 2026-09-27

## 1. 目的

当前held-out map2领域表固定`40 rounds x 16 local batches/client/round`，每客户端约
`40*16*64=40960`次样本访问。strict-fit集约8500张，因此只相当于约4.8个完整epoch。已完成
Local、FedMD和RAHFL-style结果的last-10 Avg均约20--23%，无法排除领域表处于低学习水平。

本Kill Test只增加Local/ERM的优化预算，不引入通信、JSD、DCL、PEW或BER。它回答：现有16-batch
预算是否明显欠训练。它不比较HFL方法，也不能证明PEW+BER有效。

## 2. 冻结协议

复用现有held-out map2输入、partition seed 0、training seed 0、initial states、batch size 64、
评价grid与数据角色。已有16-batch Formal直接复用；只新增：

```text
b32: 40 rounds x 32 batches/client/round ~= 9.6 epoch-equivalents
b64: 40 rounds x 64 batches/client/round ~= 19.2 epoch-equivalents
```

唯一训练臂为`Local/ERM`。`pretrain_epochs=0`，`communication=none`，`lambda_jsd=0`，
`cl_module=none`。任何非有限loss/gradient均立即失败，不允许静默跳过batch。

入口：

```text
scripts/openi_cle_hfl_learning_floor_entry.py
```

## 3. 执行顺序

当前只授权工程验证，不授权Formal：

1. 本地静态编译、单元测试和config审计；
2. OpenI分别运行b32、b64 benchmark，均为2轮并使用真实目标batch预算；
3. 根据benchmark冻结成本；
4. 用户明确批准后才能运行40轮Formal；
5. 将b16既有Formal与b32/b64 Formal离线比较。

benchmark不能作为学习水平证据。

## 4. 预注册解释

主要比较`last-10 Avg`与`operator-grid accuracy`，同时报告DSA、Worst、WCCA、CFG和曲线。

```text
若b64相对b32在Avg和grid均<1.0pp，且末段没有持续上升趋势：认为接近预算平台；
若b32相对b16或b64相对b32任一主要效用增益>=2.0pp：现有较小预算明显欠训练；
1.0--2.0pp为灰区，只允许报告为预算敏感，不得事后选择最好预算包装收敛。
```

该Kill Test不会改变M1四臂三seed的matched机制结论。它只决定S2领域表应保留为受控固定预算表、
提高统一预算，还是降入附录。

## 5. 工程修复边界

同时完成两项独立修复：

- AugHFL/RAHFL本地JSD改用稳定概率归一化；HFL context配置强制
  `skip_nonfinite=false`。旧AugHFL Formal含486次非有限梯度跳过，永久无效。
- FedProto改为从实际本地优化的同一批次累计class prototypes，并在下一轮聚合；不再每轮额外完整
  扫描private fit集。FedTGP保留原private-scan路径，不受此次修改影响。

修复后的AugHFL、RAHFL和FedProto必须重新smoke/benchmark；未经新结果不得进入最终表。

## 6. 本地真实输入验证

2026-09-27使用真实held-out map2输入完成以下验证：

- Local b32 smoke：1轮、每客户端2个batch，训练、checkpoint与分析全部完成；该结果仅验证执行。
- FedProto pairing：Local与FedProto均运行2轮、每客户端每轮2个batch；两臂各产生8条local-batch
  trace，分析器确认`all_arms_match=true`。
- FedProto在第2轮成功执行基于上一轮本地训练批次累计prototype的通信，不再触发旧版全量
  private-fit扫描。
- 以上smoke/pairing数字均设置`scientific_evidence=false`，不得进入论文结果表。

因此代码已具备OpenI b32/b64两项benchmark条件；40轮Formal仍未授权。AugHFL/RAHFL修复后的
独立真实benchmark尚未完成，在其完成前不得复用旧AugHFL包或把修复后方法列入最终领域表。
