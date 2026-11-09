# 心电信号抗噪分类

本项目研究如何结合原始心电波形与导数峰值编码（DP），识别噪声干扰下的心电记录。现阶段可以下载数据、按患者划分训练集和测试集，并用真实记录运行一次训练检查。

首次使用请从 [project/README.md](project/README.md) 开始。安装适合自己系统的 PyTorch 后，Windows 和 Linux 用户可以使用相同的命令准备数据和运行检查。

## 许可证

本项目的软件代码采用 GNU General Public License 第 3 版或任何后续版本（`GPL-3.0-or-later`）授权，完整条款见 [LICENSE](LICENSE)。

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

PTB-XL 数据采用 CC BY 4.0，NSTDB 数据采用 Open Data Commons Attribution License v1.0。第三方数据与依赖按各自的许可证授权，数据来源和引用要求见 [project/README.md](project/README.md)。
