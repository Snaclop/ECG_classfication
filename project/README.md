# 心电信号抗噪分类项目

本项目研究如何结合原始心电波形与 DP 编码，提高噪声条件下的多标签分类表现。

## 首次运行

**准备 Python 环境。** 推荐使用 Python 3.12。进入含有 `requirements.txt` 的 `project/` 目录，创建并激活自己的虚拟环境。下文的 `python` 指这个环境的解释器。各系统和终端的激活方式见 [Python 虚拟环境文档](https://docs.python.org/3/library/venv.html#how-venvs-work)。

**安装 PyTorch。** 在已激活的环境中，按操作系统和计算设备选择安装方式。CPU 和 NVIDIA GPU 用户可使用 [PyTorch 安装选择器](https://pytorch.org/get-started/locally/)。AMD GPU 用户可查阅 [AMD ROCm 兼容性与安装文档](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/)。

**安装其余依赖、数据和测试** 以下命令可在项目目录中执行。最后一步默认使用 CPU，读取两条真实心电记录，完成一次前向传播、反向传播和参数更新。

```text
python -m pip install -r requirements.txt
python scripts/download_data.py
python scripts/prepare_data.py
python scripts/check_environment.py
```

结果保存在 `outputs/setup/environment_check.json`，心电与噪声波形图保存在 `outputs/setup/waveform_preview.png`。已经配置好 NVIDIA CUDA 或 AMD ROCm 的读者，可以用以下命令在 GPU 上执行同一检查，结果写入 `outputs/setup/environment_check_gpu.json`。

```text
python scripts/check_environment.py --device cuda
```

PTB-XL 归档默认分块下载并发为 8，可用 `--workers 1` 改为单连接，或用 `--workers 16` 调整并发。中断后重新运行同一命令可复用已完成的分块。NSTDB 默认只下载实际使用的 `bw/ma/em` 三个记录（六个 `.hea/.dat` 文件）和官方 `SHA256SUMS.txt`；若需要完整数据库，额外传入 `--full-nstdb`。

`--direct` 让本次下载直接连接数据源。下载缓存位于 `data/archives/`，解压后的记录位于 `data/raw/`。准备脚本校验 PTB-XL 全部文件和 NSTDB 上述六个必需文件的官方 SHA256，校验失败会报错。

## Python 库用途

| 库 | 用途 |
|---|---|
| PyTorch | 一维卷积、门控模型及训练 |
| WFDB | 读取心电和噪声的 `.hea/.dat` 文件，转换成物理单位 |
| NumPy、pandas | 数组计算、诊断标签整理和数据清单 |
| SciPy | 重采样与信号处理 |
| scikit-learn | 后续计算 AUROC、AUPRC、F1 等指标 |
| Matplotlib | 波形与实验结果可视化 |
| tqdm、ipykernel | 进度显示与 Notebook 支持 |


## 数据来源与目录

| 数据 | 官方来源 | 本地目录 | 用途 |
|---|---|---|---|
| PTB-XL v1.0.3 | https://physionet.org/content/ptb-xl/1.0.3/ | `data/raw/ptb-xl-1.0.3/` | 主要使用 500 Hz、十秒、十二导联记录；100 Hz 版本也随官方归档保留 |
| NSTDB v1.0.0 | https://physionet.org/content/nstdb/1.0.0/ | `data/raw/nstdb-1.0.0/` | 使用 `bw` 基线漂移、`ma` 肌电、`em` 电极运动三个原始噪声记录 |

PTB-XL 的数据许可为 CC BY 4.0；NSTDB 为 Open Data Commons Attribution License v1.0。保留归档内许可文件，并在报告和论文中引用数据与原始论文。数据公开可下载。NSTDB 还包含已经叠加噪声的心电记录，本项目不把它们当作原始噪声源。

## 标签与划分

固定类别顺序：`NORM, MI, STTC, CD, HYP`，分别对应正常、心肌梗死、ST/T 改变、传导异常、肥厚。根据官方 `scp_statements.csv` 将诊断代码映射到大类，一条记录可有多个标签。

- 保留官方出现的诊断代码；`scp_codes` 中置信度为 0 表示置信度未知，不视为阴性。
- 没有任何目标大类标签的记录单独列入 `excluded_no_target.csv`，不标成正常。
- 使用官方 `strat_fold`：1–8 训练、9 验证、10 测试。
- `data/manifests/train.csv`、`validation.csv`、`test.csv` 包含记录路径和五列 0/1 标签。
- `data/manifests/audit.json` 记录文件校验数量、样本数、各类阳性数及噪声形状。
- `data/manifests/noise_splits.json` 预先将每段原始噪声按时间分为约 60%/20%/20%，边界保留十秒间隔，使用左闭右开采样索引。后续应先按该范围切片，再独立重采样，避免跨集合滤波。

检查脚本固定随机种子为 42，输入两条十二导联心电记录，每条 5000 个采样点。小型一维卷积网络为每条记录输出五个类别分数，使用 Adam 优化器和 0.001 的学习率更新一次参数。检查通过时，结果文件中 `output_shape` 为 `[2, 5]`，`parameters_updated` 为 `true`。这些参数和输出用于检验数据读取与计算流程。
