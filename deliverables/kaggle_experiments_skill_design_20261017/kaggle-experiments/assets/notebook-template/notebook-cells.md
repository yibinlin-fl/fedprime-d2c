# Kaggle网页Notebook：可复制代码单元

在网页Create → New Notebook后依次粘贴。先适配配置；默认不联网/训练。


# Kaggle experiment draft
默认 run_enabled=False：不联网、不拉代码、不训练。先适配当前项目、取得运行授权、核对配额。
代码与大缓存放临时区；结果放 /kaggle/working/<run_id>。写入working不等于已持久保存。

### 代码单元 1

```python
CONFIG = {
    "run_enabled": False,
    "mode": "smoke",
    "run_id": "configure-project-and-run-id",
    "source_mode": "github",  # github or mounted
    "repo_url": "",           # public HTTPS GitHub only; no credentials
    "commit": "",             # full 40-character SHA
    "repo_dir": "/tmp/kaggle-project",
    "mounted_source": "",     # pre-extracted, verified private source if needed
    "data_root": "",          # actual mounted input directory
    "input_files": [],         # [{"path": ..., "sha256": ...}]
    "require_gpu": True,
    "train_script": "",       # repository-relative real entry
    "train_args": [],         # real argv; supports {output_dir}, {data_root}
    "checkpoint_files": [],   # expected relative files under output_dir
    "analysis_script": "",    # empty means analysis NOT configured
    "analysis_args": [],
}
print("Execution enabled:", CONFIG["run_enabled"])
```

## 预检
首次启用前填配置。模板不安装/升级依赖；先核对项目所需环境。
不把T4×2当自动并行或合并显存。此模板最终科学标记保持false，留给项目独立审计。

### 代码单元 2

```python
if CONFIG["run_enabled"]:
    import hashlib, json, platform, re, subprocess, sys, tarfile
    from pathlib import Path
    import torch
    assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,100}", CONFIG["run_id"])
    data_root = Path(CONFIG["data_root"])
    assert CONFIG["data_root"] and data_root.is_dir(), "Configure actual input path"
    assert CONFIG["checkpoint_files"], "Declare expected checkpoint files before training"
    assert CONFIG["input_files"], "Declare and verify actual input hashes"
    if CONFIG["require_gpu"] and not torch.cuda.is_available():
        raise RuntimeError("GPU required but unavailable")
    output_dir = Path("/kaggle/working") / CONFIG["run_id"]
    output_dir.mkdir(exist_ok=False)
    for name in ("metadata", "configs", "logs", "checkpoints", "metrics", "analysis"):
        (output_dir / name).mkdir()
    def write_status(stage, error=None):
        (output_dir / "status.json").write_text(json.dumps({
            "run_id": CONFIG["run_id"], "mode": CONFIG["mode"], "stage": stage,
            "scientific_evidence": False, "audit": "pending", "error": error,
        }, indent=2), encoding="utf-8")
    def digest(path):
        h = hashlib.sha256()
        with Path(path).open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    def pack(label, names):
        suffix = "" if label == "final" else "_" + label
        target = output_dir / (CONFIG["run_id"] + suffix + "_outputs.tar.gz")
        with tarfile.open(target, "x:gz") as tar:
            for name in names:
                path = output_dir / name
                if path.exists():
                    tar.add(path, arcname=name)
        return target
    env = {"python": platform.python_version(), "torch": torch.__version__,
           "cuda": torch.version.cuda, "devices": [
               {"name": torch.cuda.get_device_name(i),
                "memory_bytes": torch.cuda.get_device_properties(i).total_memory}
               for i in range(torch.cuda.device_count())]}
    (output_dir / "metadata/environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    (output_dir / "configs/resolved.json").write_text(json.dumps(CONFIG, indent=2), encoding="utf-8")
    write_status("prepared")
```

## 代码与输入
公有仓库启用Internet后固定commit；私有代码使用提前挂载并验证过的源码目录。
依赖、LFS/子模块需由当前项目补充；不要把访问令牌写进配置/命令/输出。

### 代码单元 3

```python
if CONFIG["run_enabled"]:
    if CONFIG["source_mode"] == "github":
        assert re.fullmatch(r"https://github.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", CONFIG["repo_url"])
        assert re.fullmatch(r"[0-9a-fA-F]{40}", CONFIG["commit"])
        repo = Path(CONFIG["repo_dir"])
        assert not repo.exists(), "Use a new source directory; do not delete old files"
        subprocess.run(["git", "clone", "--no-checkout", CONFIG["repo_url"], str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "checkout", "--detach", CONFIG["commit"]], check=True)
        actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        assert actual.lower() == CONFIG["commit"].lower()
        source_identity = {"commit": actual, "repo": CONFIG["repo_url"]}
    elif CONFIG["source_mode"] == "mounted":
        repo = Path(CONFIG["mounted_source"])
        assert CONFIG["mounted_source"] and repo.is_dir()
        source_identity = {"mounted_source": str(repo), "source_audit": "must supply source manifest"}
    else:
        raise ValueError("Unknown source_mode")
    verified = []
    for item in CONFIG["input_files"]:
        path = Path(item["path"])
        actual = digest(path)
        assert actual.lower() == item["sha256"].lower(), "Input hash mismatch"
        verified.append({"path": str(path), "bytes": path.stat().st_size, "sha256": actual})
    (output_dir / "metadata/source.json").write_text(json.dumps(source_identity, indent=2), encoding="utf-8")
    (output_dir / "metadata/input_manifest.json").write_text(json.dumps(verified, indent=2), encoding="utf-8")
    def project_command(script, args):
        path = (repo / script).resolve()
        assert script and path.is_relative_to(repo.resolve()) and path.is_file()
        substitutions = {"{output_dir}": str(output_dir), "{data_root}": str(data_root)}
        resolved = []
        for arg in args:
            for key, value in substitutions.items():
                arg = arg.replace(key, value)
            resolved.append(arg)
        return [sys.executable, "-u", str(path), *resolved]
```

## 项目训练与分析前恢复包
训练runner必须按配置将checkpoint写入output_dir，并自行实现需要的周期保存/完整续跑。
训练结束打包只是本地写盘；仍需保存版本Output，不能声称已外部回传。

### 代码单元 4

```python
if CONFIG["run_enabled"]:
    try:
        write_status("training")
        with (output_dir / "logs/train.log").open("w", encoding="utf-8") as log:
            subprocess.run(project_command(CONFIG["train_script"], CONFIG["train_args"]),
                           cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=True)
        for name in CONFIG["checkpoint_files"]:
            path = (output_dir / name).resolve()
            assert path.is_relative_to(output_dir.resolve()) and path.is_file(), "Missing checkpoint"
        write_status("training_complete_analysis_pending")
        pack("recovery", ["checkpoints", "metadata", "configs", "logs", "metrics", "status.json"])
    except Exception as error:
        write_status("training_failed", type(error).__name__)
        pack("failure", ["checkpoints", "metadata", "configs", "logs", "metrics", "status.json"])
        raise
```

### 代码单元 5

```python
if CONFIG["run_enabled"]:
    if CONFIG["analysis_script"]:
        try:
            with (output_dir / "logs/analysis.log").open("w", encoding="utf-8") as log:
                subprocess.run(project_command(CONFIG["analysis_script"], CONFIG["analysis_args"]),
                               cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=True)
            write_status("execution_complete_pending_audit")
            pack("final", ["analysis", "metadata", "configs", "logs", "metrics", "status.json"])
        except Exception as error:
            write_status("analysis_failed", type(error).__name__)
            raise
    else:
        write_status("analysis_not_configured")
    artifacts = [p for p in output_dir.iterdir() if p.is_file() and p.name != "artifact_manifest.json"]
    manifest = [{"file": p.name, "bytes": p.stat().st_size, "sha256": digest(p)} for p in artifacts]
    (output_dir / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Local working artifacts:", [p.name for p in artifacts])
    print("Next: save/inspect version Output, then download; audit remains pending.")
```

## 保存版本与交回本机
Save & Run All会新建会话从头执行，不能用它保存当前进程进度。长任务选后台版本执行路线；
交互已跑完时先核验Quick Save输出保存选项或直接下载，不误启动重复长训练。
去对应版本Output检查文件后，下载恢复包、最终包、status及manifest到当前项目预建目录。
此骨架未验证真实Kaggle环境，不含配额绕过/keep-alive/账号切换，也不调用c2net。
