# CLE-HFL S2-v2基线覆盖与FedTGP Kill Test

Updated: 2026-09-22

## 1. 目的与结论

S2只承担HFL文献背景定位，不承担BER因果归因。V2把此前为控制成本而缩减的六臂扩展为十臂：

```text
Local/ERM / FedMD / FedProto / FedTGP / FedDF
KT-pFL / FCCL / RHFL / AugHFL / RAHFL
```

FedMD、FedProto和RHFL此前已经实现；本次只是接回统一40轮S2 runner。FedTGP依据AAAI 2024
原论文和作者官方仓库实现其核心服务器目标，并完成CPU单元测试与一轮真实CLE smoke。结论为：

```text
GO_AS_PROTOCOL_MATCHED_FEDTGP_CORE_ADAPTER
NOT_AN_UNTOUCHED_OFFICIAL_RECIPE
BENCHMARK_REQUIRED_BEFORE_FORMAL
```

## 2. 覆盖的机制家族

| 机制家族 | 代表方法 | S2口径 |
|---|---|---|
| 无通信 | Local/ERM | 直接下界 |
| 公共logit交换 | FedMD | protocol-matched adapter |
| 类原型聚合 | FedProto | protocol-matched adapter |
| 可训练全局原型 | FedTGP | protocol-matched core adapter |
| 服务端蒸馏 | FedDF | fidelity adapter |
| 个性化知识迁移 | KT-pFL | fidelity adapter |
| 公共响应互相关 | FCCL | protocol-matched core adapter |
| 标签噪声鲁棒 | RHFL | protocol-matched core adapter |
| corruption鲁棒 | AugHFL | fidelity adapter |
| corruption鲁棒升级 | RAHFL | strict AsymHFL-val anchor |

该集合覆盖当前论文最相关的公共响应、服务端蒸馏、原型、个性化迁移和鲁棒HFL家族，但不声称
穷尽全部HFL研究。FedGKT的客户端小模型/服务端大模型分工、DENSE的data-free生成器、
MH-pFLID的医疗个性化注入，以及ReT-FHD的多层温度与额外安全组件改变了当前协议或需要独立
重实现，保留为Related Work，不机械加入主表。

## 3. FedTGP纸面审计

正式来源：Zhang et al., *FedTGP: Trainable Global Prototypes with Adaptive-Margin-Enhanced
Contrastive Learning for Data and Model Heterogeneity in Federated Learning*, AAAI 2024。官方代码：
`https://github.com/TsingZ0/FedTGP`。

官方核心链路为：

1. 每个客户端按类别汇总本地representation prototype；
2. 服务器收集每个client-class prototype；
3. 根据类均值原型之间的最近距离形成adaptive margin；
4. 服务器训练一个class-id到global prototype的可训练生成器；
5. 下一轮客户端使用MSE向相应global prototype对齐。

项目四个客户端模型的forward均返回`(logits, embedding)`，且projector输出均为1024维，因此满足
FedProto/FedTGP共享特征空间的最低接口假设。S2仍使用项目共同local optimizer、CLE数据角色和
40轮预算，因此只能称protocol-matched core adapter，不能称官方完整recipe逐行复现。

## 4. 实现与Kill Test

实现位置：

```text
fedprime/communication/baselines.py
scripts/run_cle_hfl_context.py
```

冻结S2 adapter参数来自作者官方示例：

```text
proto_weight = 10.0
server_learning_rate = 0.01
server_epochs = 100
server_batch_size = 10
margin_threshold = 100.0
```

Kill Test结果：

```text
K0  四模型embedding共享1024维                         PASS
K1  client-class prototypes可收集且类缺失可处理         PASS
K2  adaptive margin与服务器CE均为有限值                 PASS
K3  服务器生成num_classes x feature_dim全局原型          PASS
K4  本地MSE有限且梯度回到客户端representation             PASS
K5  真实CLE smoke完成1轮、四客户端checkpoint完整          PASS
```

20项相关回归测试通过。真实smoke仅验证执行，不构成科学证据；末轮`avg_acc=10.17`、
`col_loss=2.9006`等数值禁止进入论文。

## 5. 公平性边界与下一步

S2-v2统一使用40轮、相同初始状态、相同CLE partition、相同local batch预算。它回答不同HFL
机制在同一CLE协议下的DSA与utility，不是各论文原始训练recipe排行榜。当前不单独启动AugHFL/
RAHFL官方40+40表。若未来需要官方预算参考，必须另立协议，不能与S2-v2或M1混为同一因果比较。

下一步先等待并分析M2；随后S2-v2必须先运行OpenI benchmark，核对十臂成本和学习完整性，
benchmark通过并经用户授权后才可启动40轮Formal。
