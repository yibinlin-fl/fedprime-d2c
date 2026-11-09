# Kaggle 启动卡｜普通图像损坏下的 RAHFL 协作预检

**当前版本先完成八臂的第 8 轮，再从各自权重续到第 12 轮。已上传的 Dataset 可以直接复用，需使用下面最新版的两个单元。**

## 实验和上传包

一次执行两个训练种子（0、1），每个种子顺序运行 `local`、`uniform`、`concentrated_a`、`concentrated_b`。四个等样本量客户端分别使用 ResNet10、ResNet12、ShuffleNet 和 MobileNetV2。三个联邦臂的损坏比例依次为 50/50/50/50%、50/50/20/80%、50/50/80/20%，总损坏数均为 4,000 张。主观察对象是客户端 0。Local 保留其 RAHFL 的 AugMix/JSD/DCL，只关闭公共前向和蒸馏。

使用原作者模型、项目已有的稳定版 JSD/DCL 训练函数及按准确率排序的 RAHFL 蒸馏规则。独立干净 CIFAR-10 验证集用于路由，独立的预检评价集用于结果。每个客户端 2,000 张训练图，公共端 1,000 张无标签 CIFAR-100 图。训练损坏为高斯噪声与散焦模糊，固定严重度 3。这是普通损坏下的受控预检，数据、批次和轮数与作者原始复现协议有区别。

- 新建 **私有 Kaggle Dataset**，建议显示名称为 `HFL Peer Corruption Pilot 20261110`。只上传下面一个 ZIP。Notebook 也保持 Private。账号名、实际 Dataset slug 和 GPU 额度由你在页面确认。
- 本机上传路径为 `C:\Users\asus\Desktop\FedPRIME-D2C\local_runs\peer_corruption_20261110\hfl_peer_corruption_kaggle_20261110.zip`。
- 文件大小为 **31,076,380 bytes**，SHA256 为 `746630cf285089d92583d9e83a299f0d454b046fb50dfefa0a0af5c2e1ab65fa`。已带固定数据及所需源码。Notebook 校验压缩包或解包后的 manifest，再逐文件校验内容。它无需旧 CLE 包、PEW、GitHub Token 或从 GitHub 拉私有代码。
- 基础仓库 Git 版本为 `a694982a4357c99dd0a582ac5df9a0263e48a437`。ZIP 提供冻结输入与基础源码。第一段在核对原包后加载已经验证的两阶段入口并核对新入口的源码哈希。无需重传 Dataset 或拉取远程分支。

## 网页设置

新建 Python Notebook，通过 Add Input 挂载上面的私有 Dataset。Accelerator 选择当前账号可用的 **GPU T4**。即使界面分配两张卡，这个入口也只用第 0 张。Internet 保持 **Off**，使用镜像已装的 PyTorch、torchvision、numpy、Pillow、pandas、matplotlib、seaborn 和 sklearn。第一段会检查依赖和 CUDA；若报缺依赖，训练尚未开始，应先修复环境。

将以下两段分别粘贴到两个代码单元。在页面使用 **Save Version → Save & Run All** 启动一次后台执行。先核对账户实际剩余 GPU 时长和会话上限，留够打包与保存时间。无需先在编辑会话把训练跑一遍。Kaggle 后台版本执行和输出见 [官方 Notebook 文档](https://www.kaggle.com/docs/notebooks)。

### 单元 1｜核验并挂载

~~~python
from pathlib import Path
import base64
import hashlib
import importlib.util
import json
import shutil
import zipfile
import zlib

import torch
import torchvision

INPUT_ROOT = Path("/kaggle/input")
SOURCE = Path("/tmp/hfl_peer_corruption_20261110")
WORK = Path("/kaggle/working")
OUT = WORK / "peer_corruption_pilot_20261110_r8_r12"
BUNDLE = "hfl_peer_corruption_kaggle_20261110.zip"
ZIP_SHA = "746630cf285089d92583d9e83a299f0d454b046fb50dfefa0a0af5c2e1ab65fa"
MANIFEST_SHA = "6484ab1e9f22b64505d66e753410492b12cdf1d3bd51f5619040ff2f21a8caf2"
INPUT_FILE = "peer_corruption_input.npz"

def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

missing = [name for name in ("numpy", "PIL", "pandas", "matplotlib", "seaborn", "sklearn")
           if importlib.util.find_spec(name) is None]
assert not missing, f"Missing dependencies: {missing}"
assert torch.cuda.is_available(), "Select a GPU accelerator before running"
assert not OUT.exists(), "This session already has an output directory"
print("GPU:", torch.cuda.get_device_name(0), "torch:", torch.__version__, "torchvision:", torchvision.__version__)

archives = sorted(INPUT_ROOT.rglob(BUNDLE))
if archives:
    assert len(archives) == 1, archives
    assert file_sha(archives[0]) == ZIP_SHA, "Incorrect input ZIP"
    SOURCE.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archives[0]) as bundle:
        bundle.extractall(SOURCE)
else:
    matches = [p for p in INPUT_ROOT.rglob("manifest.json") if file_sha(p) == MANIFEST_SHA]
    assert len(matches) == 1, "Attach the private Kaggle Dataset containing the pilot ZIP"
    manifest = json.loads(matches[0].read_text())
    for relative in [*manifest["source_files"], INPUT_FILE, "manifest.json"]:
        target = SOURCE / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(matches[0].parent / relative, target)

assert file_sha(SOURCE / "manifest.json") == MANIFEST_SHA
manifest = json.loads((SOURCE / "manifest.json").read_text())
assert file_sha(SOURCE / INPUT_FILE) == manifest["input_sha256"]
for relative, expected in manifest["source_files"].items():
    assert file_sha(SOURCE / relative) == expected, relative

# Drop-in two-stage runner. The original Dataset stays unchanged.
runner_bytes = zlib.decompress(base64.b64decode("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"))
runner_sha = "c84a4c615d0aab897ae32beddf998c14b15bd1c2cf3505668fb3ef1e9f60b35b"
assert hashlib.sha256(runner_bytes).hexdigest() == runner_sha, "Invalid two-stage runner"
runner = SOURCE / "peer_corruption_pilot/run.py"
runner.write_bytes(runner_bytes)
runtime_manifest = dict(manifest, source_files=dict(manifest["source_files"]))
runtime_manifest["source_files"]["peer_corruption_pilot/run.py"] = runner_sha
runtime_manifest["original_manifest_sha256"] = MANIFEST_SHA
(SOURCE / "runtime_manifest.json").write_text(json.dumps(runtime_manifest, indent=2))
print("Input and source verified")
~~~

### 单元 2｜直接跑八臂和打包

~~~python
import os
import subprocess
import sys
import tarfile

command = [
    sys.executable, "-u", "-m", "peer_corruption_pilot.run",
    "--input", str(SOURCE / INPUT_FILE),
    "--output", str(OUT),
    "--device", "cuda",
    "--limit-hours", "8",
]
env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
train_log = WORK / "peer_corruption_train.log"
with train_log.open("w") as log:
    process = subprocess.Popen(command, cwd=SOURCE, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in process.stdout:
        print(line, end="", flush=True)
        log.write(line)
        log.flush()
    return_code = process.wait()

launch = WORK / "peer_corruption_launch.json"
status = OUT / "status.json"
launch.write_text(json.dumps({
    "command": command, "return_code": return_code,
    "state": json.loads(status.read_text()) if status.exists() else {"state": "failed_before_status"},
    "formal_evidence": False,
}, indent=2))

result = WORK / "peer_corruption_pilot_20261110_r8_r12_analysis.tar.gz"
with tarfile.open(result, "w:gz") as archive:
    archive.add(train_log, arcname=train_log.name)
    archive.add(launch, arcname=launch.name)
    archive.add(SOURCE / "manifest.json", arcname="source_manifest.json")
    archive.add(SOURCE / "runtime_manifest.json", arcname="runtime_manifest.json")
    if OUT.exists():
        for file in sorted(OUT.rglob("*")):
            if file.is_file() and file.suffix in {".json", ".csv", ".npz"}:
                archive.add(file, arcname=str(file.relative_to(WORK)))
print(launch.read_text())
print("Analysis archive:", result, "SHA256:", file_sha(result))
~~~

## 预算、结果保存和验收

- 跳过 Kaggle benchmark。每客户端预训练 5 个 epoch。先完成全部八臂的第 8 轮并保存预测与权重，再加载第 8 轮状态续训到第 12 轮。后续每轮 1 个本地 epoch，私有 batch size 64、公共 batch size 128、Adam 学习率 0.001。每个种子下，客户端 0 的预训练复用于四臂。云端运行不传 `--smoke`。
- 共用 8 小时软上限，实际 T4 耗时未测。入口在每个预训练 epoch 和训练轮前检查预算，进行中的轮可能略过上限。若第 12 轮未跑齐，仍按第 8 轮完整矩阵分析。两阶段在同一任务中续训，按轮重建优化器和随机种子。任意中断点的通用恢复不属于本次入口。
- 对应 Notebook 版本的 Output 出现后，下载 **`peer_corruption_pilot_20261110_r8_r12_analysis.tar.gz`**。本机接收目录已建好，路径为 `C:\Users\asus\Desktop\FedPRIME-D2C\outputs\kaggle_downloads\peer_corruption_20261110`。权重保留在对应版本的 `peer_corruption_pilot_20261110_r8_r12/round_08/` 和 `round_12/` 中，正常分析只需要小包。
- 核验总 `status.json` 的 `completed_milestones`，以及每个里程碑的 `collaboration_gains.json` 是否显示完整。再核对目标客户端预训练权重、实际本地批次和评价源图。每个里程碑独立验收，跨臂比较使用共同轮次。若为 `budget_stopped` 或 `failed`，下载已保存的结果。
- 主指标为客户端 0 在高斯噪声和散焦模糊上的等权平均准确率。各联邦臂减去同 seed、同轮次的 Local，得到协作收益。预检信号为均匀臂相对 Local 至少 +1.0 pp、集中臂相对 Local 至多 −1.0 pp。第 1、4、8、12 轮记录蒸馏前后的损坏路由验证准确率。比较两个里程碑时检查方向是否持续。预检只筛查本协议下的负迁移和位置依赖，正式结论仍需要确认。
