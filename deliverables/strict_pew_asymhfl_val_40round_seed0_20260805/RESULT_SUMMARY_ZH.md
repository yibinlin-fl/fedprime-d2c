# strict PEW + AsymHFL-val 40 轮 seed-0 结果

Updated: 2026-08-05

## 结论

固定 CLE `alpha05_gamma09_seed0_split0`、training seed 0 的 40 轮 durability
实验通过全部八项预注册门槛，判定 `GO`。这证明 12 轮优势在该训练种子下
可持续到 40 轮；尚未证明 40 轮多训练种子或跨 CLE scenario 泛化。

## Candidate-minus-control

```text
window    Avg       Worst     WCCA      CFG
last-10   +4.9292   +3.2987   +9.8750   -5.4700
last-5    +4.7140   +3.1747   +9.7000   -5.0800
final     +4.3183   +2.2200   +5.0000   -5.4000
```

最后十轮四项指标在 10/10 轮均保持正确方向；最后五轮也为 5/5。最后十轮
扩展增益包括 worst-group `+6.4250`、worst-client-group `+6.0900`、seen
Avg/Worst `+4.7486/+3.2045`、unseen Avg/Worst `+5.4256/+3.5575`。

## 完整性与公平性

- 两臂均为完整 rounds 0-39，核心指标无 NaN。
- 包内比较与独立重算完全一致。
- 返回配置与 Git 正式配置哈希一致，`resume=false`、`save_final=false`。
- 数据、模型、训练协议、AsymHFL-val 和 strict fit/audit 划分匹配。
- 固定划分 SHA-256 为
  `75C6BD9DC4B7714F505EEA2C047F1B882582DA311D00D099B6CAAC1B5BA4D2EC`。
- 两臂前 12 轮与此前正式 seed-0 结果完全一致。

原始归档位于 `outputs/strict_pew_asymhfl_val_40round_seed0_outputs.tar.gz`，
原始输出默认不提交 Git。
