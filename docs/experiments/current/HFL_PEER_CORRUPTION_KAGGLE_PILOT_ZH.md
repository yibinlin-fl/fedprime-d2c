# Kaggle 启动卡｜普通图像损坏下的 RAHFL 协作预检

## 实验和上传包

一次执行两个训练种子（0、1），每个种子顺序运行 `local`、`uniform`、`concentrated_a`、`concentrated_b`。四个等样本量客户端分别使用 ResNet10、ResNet12、ShuffleNet 和 MobileNetV2。三个联邦臂的损坏比例依次为 50/50/50/50%、50/50/20/80%、50/50/80/20%，总损坏数均为 4,000 张。主观察对象是客户端 0。Local 保留其 RAHFL 的 AugMix/JSD/DCL，只关闭公共前向和蒸馏。

使用原作者模型、项目已有的稳定版 JSD/DCL 训练函数及按准确率排序的 RAHFL 蒸馏规则。独立干净 CIFAR-10 验证集用于路由，独立的预检评价集用于结果。每个客户端 2,000 张训练图，公共端 1,000 张无标签 CIFAR-100 图。训练损坏为高斯噪声与散焦模糊，固定严重度 3。这是普通损坏下的受控预检，数据、批次和轮数与作者原始复现协议有区别。

- 新建 **私有 Kaggle Dataset**，建议显示名称为 `HFL Peer Corruption Pilot 20261110`。只上传下面一个 ZIP。Notebook 也保持 Private。账号名、实际 Dataset slug 和 GPU 额度由你在页面确认。
- 本机上传路径为 `C:\Users\asus\Desktop\FedPRIME-D2C\local_runs\peer_corruption_20261110\hfl_peer_corruption_kaggle_20261110.zip`。
- 文件大小为 **31,076,380 bytes**，SHA256 为 `746630cf285089d92583d9e83a299f0d454b046fb50dfefa0a0af5c2e1ab65fa`。已带固定数据及所需源码。Notebook 校验压缩包或解包后的 manifest，再逐文件校验内容。它无需旧 CLE 包、PEW、GitHub Token 或从 GitHub 拉私有代码。
- 基础仓库 Git 版本为 `a694982a4357c99dd0a582ac5df9a0263e48a437`。新入口的确切版本由 ZIP 和其中的源码哈希标识。使用本机打包的版本，和远程分支无关。

## 网页设置

新建 Python Notebook，通过 Add Input 挂载上面的私有 Dataset。Accelerator 选择当前账号可用的 **GPU T4**。即使界面分配两张卡，这个入口也只用第 0 张。Internet 保持 **Off**，使用镜像已装的 PyTorch、torchvision、numpy、Pillow、pandas、matplotlib、seaborn 和 sklearn。第一段会检查依赖和 CUDA；若报缺依赖，训练尚未开始，应先修复环境。

将以下两段分别粘贴到两个代码单元。在页面使用 **Save Version → Save & Run All** 启动一次后台执行。先核对账户实际剩余 GPU 时长和会话上限，留够打包与保存时间。无需先在编辑会话把训练跑一遍。Kaggle 后台版本执行和输出见 [官方 Notebook 文档](https://www.kaggle.com/docs/notebooks)。

### 单元 1｜核验并挂载

~~~python
from pathlib import Path
import hashlib
import importlib.util
import json
import shutil
import zipfile

import torch
import torchvision

INPUT_ROOT = Path("/kaggle/input")
SOURCE = Path("/tmp/hfl_peer_corruption_20261110")
WORK = Path("/kaggle/working")
OUT = WORK / "peer_corruption_pilot_20261110"
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

result = WORK / "peer_corruption_pilot_20261110_analysis.tar.gz"
with tarfile.open(result, "w:gz") as archive:
    archive.add(train_log, arcname=train_log.name)
    archive.add(launch, arcname=launch.name)
    archive.add(SOURCE / "manifest.json", arcname="source_manifest.json")
    if OUT.exists():
        for file in sorted(OUT.rglob("*")):
            if file.is_file() and file.suffix in {".json", ".csv", ".npz"}:
                archive.add(file, arcname=str(file.relative_to(WORK)))
print(launch.read_text())
print("Analysis archive:", result, "SHA256:", file_sha(result))
~~~

## 预算、结果保存和验收

- 按要求跳过 Kaggle benchmark。预检预算固定为每客户端 5 个本地预训练 epoch、8 个后续训练轮、每轮 1 个本地 epoch，私有 batch size 64、公共 batch size 128、Adam 学习率 0.001。每个种子下，客户端 0 的预训练复用于四臂。所有参数由单元和源码锁定，`--smoke=false` 对应不传 `--smoke`。
- 8 小时是计划的运行上限，实际 T4 耗时未实测。入口在每个预训练 epoch 和训练轮前检查是否到达上限，保留已完成臂的数据和状态，再退出打包。进行中的轮可能略过上限。每臂第 4 轮及训练结束分别写入模型权重。权重用于回收和分析，完整断点续训尚未验证。
- 对应 Notebook 版本的 Output 出现后，只需下载 **`peer_corruption_pilot_20261110_analysis.tar.gz`**。预先建好的本机目录为 `C:\Users\asus\Desktop\FedPRIME-D2C\outputs\kaggle_downloads\peer_corruption_20261110`。各臂的 `weights.pt` 和 `midpoint_weights.pt` 保留在该版本的 `peer_corruption_pilot_20261110/seed_<seed>/<arm>/` 下，需要再下载；分析包已经包含预测、完整性记录和日志。
- 首先核验 `status.json` 与 `collaboration_gains.json` 是否显示八臂完成，并验证目标客户端预训练权重、实际本地批次与评价源图一致。Kaggle 界面完成和八臂科学验收分别判断。若为 `budget_stopped` 或 `failed`，下载保留下来的结果，按部分预检处理。
- 主指标为客户端 0 在高斯噪声和散焦模糊上的等权平均准确率。各联邦臂减去同 seed 的 Local，得到协作收益。预检复验信号为均匀臂相对 Local 至少 +1.0 pp、集中臂相对 Local 至多 −1.0 pp。再核对这个方向是否跨两个 seed 和两个集中位置出现。第 1、4、8 轮另记录蒸馏前后的损坏路由验证准确率。预检只能在本协议下筛查负迁移和位置依赖，不能建立广泛机制结论或代替正式 RAHFL 基准。
