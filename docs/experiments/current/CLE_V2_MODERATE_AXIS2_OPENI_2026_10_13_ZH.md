# CLE-v2 Moderate 剂量响应与 Axis II OpenI 协议

Updated: 2026-10-13

## 1. 目的与论文职责

本协议只使用冻结输入 `alpha=1.0, rho=0.5, map2, seed0/split0`。它与旧 strong
`alpha=0.5, rho=1.0, gamma=0.9` 机制协议不混表。

- `dose_all`：在相同 AsymHFL 通信与 ERM 目标下比较 `gamma=0/0.5/0.9`，并在
  `gamma=0.9`加入`PEW+BER`，检验 CLE 强度的剂量响应和缓解方向。
- `axis2_all`：在`gamma=0.9` moderate场景比较八种 HFL 方法/锚点，回答论文在 HFL
  领域中的位置；它不是 BER 的单因素归因实验。

八行 Axis II：

```text
Local/ERM
FedMD
FedProto
FedDF
AugHFL
RAHFL
AsymHFL--ERM
AsymHFL+PEW+BER
```

## 2. 冻结输入

```text
OpenI显示名（建议）：CLE_v2_Moderate_Alpha10_Rho05_Map2_Seed0_PEW_20261013
本地文件：C:\Users\asus\Desktop\FedPRIME-D2C\local_runs\cle_v2_moderate_alpha10_rho05\cle_hfl_v2_moderate_alpha10_rho05_map2_seed0_split0_with_pew.tar.gz
bytes：893421220
SHA256：DEA800EC1465C45037E74E4B948DFB8D778F1354E8522B551438C6A515C9E588
包目录：cle_hfl_v2_moderate_alpha10_rho05_map2_seed0_split0
```

输入包含三种gamma的四客户端私有训练集、clean/balanced-corrupted/seen/unseen测试集、
1000-source x 15-operator DSA网格、公共CIFAR-100、固定初始化、strict fit/audit split和三套
冻结PEW annotations。完整性审计为PASS。

## 3. 工程入口与预算

```text
OpenI启动文件：scripts/openi_cle_v2_moderate_entry.py
本地runner：scripts/run_cle_v2_moderate.py
分析器：scripts/analyze_cle_v2_moderate.py
```

| mode | rounds | max local batches/client/round | 证据性质 |
|---|---:|---:|---|
| smoke | 1 | 1 | 仅执行验证 |
| benchmark | 2 | 8 | 仅成本验证 |
| formal | 40 | 64 | 完整后才可作科学证据 |

统一参数为batch size 64、public batch size 128、每轮4个public batches、training seed 0、
pretrain 0、`skip_nonfinite=false`。普通基线使用单视图standard loader；AugHFL/RAHFL保留
各自AugMix路径。

## 4. 分片

```text
dose_all
axis2_all
axis2_plain      # Local/FedMD/FedProto/FedDF
axis2_robust     # AugHFL/RAHFL
axis2_asym       # AsymHFL--ERM/AsymHFL+PEW+BER
```

也可用具体arm名运行单臂。跨账号运行时，每个方法必须完整地在一个任务中完成，不能跨账号
拼接轮次；所有账号必须使用相同commit、输入SHA256、镜像、GPU、seed和预算。

## 5. 指标

- `pooled_dsa`：paired operator变化是否把概率质量推向训练binding类别；越接近0越好。
- `client_dsa`：四个异构客户端分别的DSA，检查收益是否只来自某个架构。
- `operator_grid_accuracy`：同源样本在15个operator上的语义准确率。
- `clean_accuracy`：clean测试集准确率。
- `corrupted_balanced_accuracy`：均衡corruption测试集准确率。
- `seen_operator_accuracy`：私有训练中出现过的operator上的准确率。
- `unseen_operator_accuracy`：私有训练中未出现的held-out operator上的准确率。
- `rho05_mixed_accuracy`：按真实协议`0.5*clean + 0.5*corrupted`得到的总体效用。
- `reference_shuffled_binding_null`：打乱binding后DSA的零基准、p95和p-value。
- `last10_training_metrics`：旧训练记录中的Avg/Worst/WCCA/CFG，仅作补充，不替代上述
  统一checkpoint推理结果。

Smoke和benchmark的数值都不能进入论文。

## 6. 当前验证状态

- 压缩包217 entries/178 files二次流式审计通过。
- runner对协议、包名、alpha/rho/gamma、strict split、paired DSA、四类测试split及PEW文件
  执行fail-closed检查。
- `dose_all` prepare-only通过。
- `local_erm`真实CPU smoke完成一轮一batch、四模型checkpoint和全指标分析。
- 3项聚焦测试通过。

2026-10-14状态：`local_erm`、`fedmd_adapter`和`fedproto_adapter`三个seed-0、40轮x64 batches
Formal已经完成并通过合同、完成标记、统一指标及跨账号batch轨迹审计；结果见
`deliverables/cle_v2_moderate_axis2_plain_formal_seed0_20261014/RESULT_SUMMARY_ZH.md`。
`feddf_fidelity`及两条AsymHFL Formal尚未运行。AugHFL/RAHFL仅完成benchmark，因每臂Formal
机械估算约17小时/102积分而暂缓。`dose_all`尚未运行。

## 7. OpenI上传状态

2026-10-13已使用本机OpenI CLI上传完成：

```text
repo_id：chujiu/CLE_v2_Moderate_Alpha10_Rho05_Map2_Seed0_PEW_20261013
remote file：cle_hfl_v2_moderate_alpha10_rho05_map2_seed0_split0_with_pew.tar.gz
CLI result：100%，852.0 MiB，约1.0 MiB/s，00:13:04，exit code 0
```

凭据仍只保存在`C:\Users\asus\.openi\token.json`，仓库不记录Token值。训练任务可直接搜索并
挂载该公开数据集；禁止误选旧1.3GB cross-maps包或旧strong输入包。
