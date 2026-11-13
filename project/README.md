# 心电信号抗噪分类项目

## 首次运行

**准备 Python 环境。** 推荐使用 Python 3.12。进入含有 `requirements.txt` 的 `project/` 目录，创建并激活自己的虚拟环境。下文的 `python` 指这个环境的解释器。

**安装 PyTorch。** 以Windows系统 + AMD显卡为例，其余平台请自行上网查询。[AMD ROCm 兼容性与安装文档](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/)。

**安装其余依赖和下载原始数据。** 以下命令可在项目目录中执行。

```text
python -m pip install -r requirements.txt
python preprocessing/download_data.py
python preprocessing/prepare_data.py
```

NSTDB 只下载实际使用的 `bw/ma/em` 三个记录（六个 `.hea/.dat` 文件）。下载缓存位于 `data/archives/`，解压后的记录位于 `data/raw/`。

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

PTB-XL 的数据许可为 CC BY 4.0；NSTDB 为 Open Data Commons Attribution License v1.0。保留归档内许可文件，并在报告和论文中引用数据与原始论文。数据公开可下载。

## 标签与划分

固定类别顺序：`NORM, MI, STTC, CD, HYP`，分别对应正常、心肌梗死、ST/T 改变、传导异常、肥厚。根据官方 `scp_statements.csv` 将诊断代码映射到大类，一条记录可有多个标签。

- 保留官方出现的诊断代码；`scp_codes` 中置信度为 0 表示置信度未知，不视为阴性。
- 没有任何目标大类标签的记录单独列入 `excluded_no_target.csv`，不标成正常。
- 使用官方 `strat_fold`：1–8 训练、9 验证、10 测试。
- `data/manifests/train.csv`、`validation.csv`、`test.csv` 包含记录路径和五列 0/1 标签。
- `data/manifests/noise_splits.json` 预先将每段原始噪声按时间分为约 60%/20%/20%，边界保留十秒间隔，使用左闭右开采样索引。后续应先按该范围切片，再独立重采样，避免跨集合滤波。

## 生成信噪比为 6 dB 训练集

完成数据准备后，在 `project/` 中运行下列命令。生成约 4.10 GB 的波形到 `data/derived/train_6db/`。

```text
python preprocessing/generate_train_6db.py
```

训练入口使用产物内的 `train.csv`。17,084 个 ECG 各有一条 `original` 和一条加噪样本，共 34,168 行，标签顺序沿用上文的五类顺序。`waveform_path` 相对于 `project/`。当 `format=wfdb` 时，使用 `wfdb.rdrecord(path).p_signal.T.astype("float32")` 读取原始记录；当 `format=npy` 时，使用 `numpy.load(path, allow_pickle=False)`。两者均为 `[12, 5000]`，单位 mV、采样率 500 Hz。

打乱按 ECG 编号排序的训练记录，均分到 `bw`、`ma`、`em`、`mix` 四个条件。`bw_start_360`、`ma_start_360`、`em_start_360` 保存对应原始噪声文件中从零计数的起点，空白表示未使用。第一通道用于六个肢体导联，第二通道用于 V1–V6。`mix` 将三种噪声按通道去均值、调至相同有效值后相加。随后逐导联将总噪声调到心电去均值有效值的 `10**(-6/20)` 倍。人工 50 Hz 工频干扰留给最终测试对不同噪声的普适性。

## 生成最终测试集

在 `project/` 中执行下列命令，为官方第 10 折的 2,158 条心电各生成一条 50 Hz、6 dB 的模拟工频干扰噪声，输出到 `data/derived/test_50hz_6db/`。

```text
python preprocessing/generate_test_50hz.py
```

产物中的 `test.csv` 共 4,316 行，每条心电配有 `original` 和 `50hz` 两个版本，沿用训练清单的标签顺序、`format` 和 `waveform_path` 读取方式。加噪波形在 `waveforms/` 中，格式为 `float32[12, 5000]`，采样率 500 Hz、单位 mV。十二导联共享频率 50 Hz、初相位 0 的正弦波，逐导联按去均值有效值缩放到 6 dB。恒定导联保留原值并标记。

## 生成一阶 DP 编码

一阶融合模型使用波形及对应的 DP1 两路输入。在 `project/` 中执行下列命令，可为已有训练集、原始验证集和 50 Hz 测试集生成编码到 `data/derived/dp1/`。

```text
python preprocessing/generate_dp1.py
```

产物中的 `train.csv`、`validation.csv`、`test.csv` 分别有 34,168、2,146、4,316 行，保留来源清单的标签与波形信息，增加 `dp1_path`。两个路径字段都相对于 `project/`。原始验证集的行已补齐 `version=original`、`format=wfdb` 和 `waveform_path`。编码文件为 `int8[12, 5000]`，值为 -1、0、1，进入编码分支前转换为 float32。波形基线读取 `waveform_path`，融合模型额外读取同一行的 `dp1_path`。

采用论文式 (1) 的一阶导数过零规则。本项目直接对模型读取的 float32 波形编码，差分计算采用 float64。具体约定为 `d[t]=x[t]-x[t-1]`；当 `d[t]>0` 且 `d[t-1]<=0` 时记 +1，当 `d[t]<0` 且 `d[t-1]>=0` 时记 -1，其余记 0。论文原实验使用 100 Hz 波形和三点滑动平均，本项目使用 500 Hz 波形、无滑动平均。[DP 论文](https://arxiv.org/html/2405.00724v1)

## 生成二阶 DP 编码

完成一阶编码后，在 `project/` 中运行下列命令，为同一批 40,630 个样本生成二阶编码到 `data/derived/dp2/`。

```text
python preprocessing/generate_dp2.py
```

产物中的 `train.csv`、`validation.csv`、`test.csv` 分别有 34,168、2,146、4,316 行。每行同时包含 `waveform_path`、`dp1_path` 和 `dp2_path`，路径相对于 `project/`，标签沿用一阶清单。

二阶编码直接从对应波形计算 `d2[t]=(x[t]-x[t-1])-(x[t-1]-x[t-2])`，再按一阶编码使用的正负过零规则生成事件。输入为实际加载的 float32 波形，内部采用 float64 差分，平滑设置为 `none`，开头三个采样点置零，输出为 `int8[12, 5000]`。

二阶融合模型加载一阶与二阶数组，用 `numpy.concatenate([dp1, dp2], axis=0).astype("float32")` 得到 `[24, 5000]` 的编码输入。前 12 个通道为一阶导联 I–V6，后 12 个为二阶导联 I–V6。原始波形仍由波形分支读取。

## 原始波形 CNN 基线

只向基线模型提供波形，训练仍使用上述 34,168 条原始与 6 dB 加噪样本。验证使用第 9 折 2,146 条原始记录。

四层普通一维卷积的通道数为 `32→64→128→128`，卷积核均为 7、步长 1、padding 3，无卷积偏置，每层接 BatchNorm 和 ReLU。前三层后接 `MaxPool1d(4)`，末尾使用全局平均池化、Dropout 0.2、五维线性层。`WaveformCNN.encode` 返回 `[B,128]` 特征，`forward` 返回 `[B,5]` logits。模型包含 190,405 个参数。类别顺序固定为 `NORM, MI, STTC, CD, HYP`，预测使用逐类 sigmoid，允许多个或零个阳性标签。

### 训练配置与选择规则

| 设置 | 默认值 |
|---|---|
| 随机种子 | 42 |
| 优化器 | AdamW，学习率 0.001，weight decay 0.0001，`foreach=False`；不使用学习率调度 |
| 损失 / 精度 | 无类别加权的 `BCEWithLogitsLoss` / FP32 |
| 选模 | 验证集宏平均 AUROC 最高；并列时验证 BCE 较低优先|

### 启动与恢复训练

以下命令均在 `project/` 目录执行。

```text
python training/train_baseline.py
```

新训练拒绝使用已经存在的输出目录。需要更改 batch size、轮数、学习率等配置时，在首次运行前修改入口函数中的配置；不同模型比较时须保持总轮数和数据条件一致。

每个完整轮次完成后保存检查点。终端中断后，将入口函数中的 `resume` 改为 `True` 再运行：

```text
python training/train_baseline.py
```

恢复直接使用保存的实验配置，中断的未完成轮次会重跑，每个运行目录同时只启动一个训练进程。