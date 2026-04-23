# ICV LiDAR Tools

`ICV LiDAR Tools` 是一个面向 LiDAR/视频融合实验的数据处理工具集，主要用于：

- 解析 `pcap` 协议数据并导出结构化点云
- 将 `parquet` 点云转换为 `csv`
- 对点云进行单帧/首场景可视化
- 将 LiDAR 点投影到视频帧上进行融合检查
- 对时间戳数据做最近邻对齐

项目以 Python 为主，代码分为两部分：

- `src/icv_lidar_tools/`：可安装的库源码
- `examples/`：各种脚本、调试脚本和实验脚本

---

## 项目结构

当前与功能相关的主要文件如下：

- `pyproject.toml`
- `src/icv_lidar_tools/__init__.py`
- `src/icv_lidar_tools/models.py`
- `src/icv_lidar_tools/time_sync.py`
- `src/icv_lidar_tools/convert_par_to_csv.py`
- `src/icv_lidar_tools/visualize_single.py`
- `src/icv_lidar_tools/lidar/__init__.py`
- `src/icv_lidar_tools/lidar/pcap_parser_protocol.py`
- `examples/vl_projection.py`
- `examples/vl_projection_pyqt_tuner.py`
- `examples/video_rename.py`
- `examples/query.py`
- `examples/pcap_export_protocol_single.py`
- `examples/pcap_export_protocol_all.py`
- `examples/visualization_bin_points.py`
- `examples/demo_protocol_full_fields_export.py`


---

## 环境要求

- Python `3.10+`
- Windows 10 / Windows 11

`pyproject.toml` 中声明的基础依赖包括：

- `numpy`
- `pandas`
- `pyarrow`
- `openpyxl`
- `matplotlib`
- `scapy`

部分脚本还会用到以下可选依赖：

- `open3d`：点云可视化
- `opencv-python`：视频读取与投影叠加
- `PyQt` / `PySide`：交互式调参界面
- `cupy`：部分解析流程中可能用于 GPU 加速

建议安装方式：

```bash
pip install -e .
```

如果你要运行可视化或视频投影脚本，再补充安装：

```bash
pip install open3d opencv-python
```

---

## 核心功能说明

### 1. `src/icv_lidar_tools/time_sync.py`

用于做时间戳对齐。

核心类：

- `TimeAligner`

常见用途：

- GNSS 与点云对齐
- 视频帧与点云对齐
- 多源传感器时间同步

---

### 2. `src/icv_lidar_tools/convert_par_to_csv.py`

用于将 `parquet` 转成 `csv`，支持单文件和目录批量处理。

主要函数：

- `parquet_to_csv(parquet_path, csv_path, chunk_size=None)`
- `convert_parquet_dir_to_csv(input_dir, output_dir, chunk_size=None, recursive=False)`

命令行参数：

- `--input` / `-i`：单文件模式输入
- `--input-dir`：目录模式输入
- `--output` / `-o`：单文件模式输出
- `--output-dir`：目录模式输出
- `--chunk-size`：分块读取大小
- `--recursive`：是否递归搜索子目录

示例：

```bash
python -m icv_lidar_tools.convert_par_to_csv --input xxx.parquet --output xxx.csv
```

目录批量转换：

```bash
python -m icv_lidar_tools.convert_par_to_csv --input-dir G:/data/parquet_out --output-dir G:/data/csv_out --recursive
```

---

### 3. `src/icv_lidar_tools/visualize_single.py`

用于可视化 `parquet` 点云，默认显示首个时间片/首个场景。

支持的颜色模式：

- `intensity`
- `height`
- `distance`

主要入口：

- `visualize_points(parquet_path, color_mode='intensity')`

适合用于：

- 快速检查点云文件是否正确
- 验证点云坐标、强度和高度分布

> 该脚本依赖 `open3d`。

---

### 4. `src/icv_lidar_tools/lidar/pcap_parser_protocol.py`

用于解析 LiDAR `pcap` 协议数据，并导出结构化点云。

主要类：

- `DecodedMsopPacket`
- `ParseConfig`
- `MsopDecoder`
- `ProtocolPcapParser`

主要能力：

- 解析 UDP 协议包
- 提取 `packet_timestamp`、`azimuth_deg`、`ring`、`distance_m`、`intensity`、`x/y/z` 等字段
- 支持导出 `parquet` / `bin`
- 支持批量解析

适用场景：

- 从原始 `pcap` 中提取点云
- 调试协议字段
- 为后续投影、可视化和转换做数据准备


---

### 5. `src/icv_lidar_tools/models.py`

定义基础数据结构。

常见模型：

- `LidarPoint`

用途：

- 表示解析后的点云点
- 便于数据导出、测试和中间处理

---

## 示例脚本说明

### `examples/vl_projection.py`

LiDAR 点云投影到视频帧的主脚本之一。

特点：

- 读取视频和点云 `parquet`
- 在视频帧上叠加投影点
- 支持时间窗口筛选
- 支持图像边界过滤、边缘掩码过滤、颜色一致性过滤、深度时序过滤
- 支持按深度或强度着色
- 支持自动匹配目录中的视频和点云文件

示例：

```bash
python examples/vl_projection.py --video G:/data/video/a.mp4 --parquet G:/data/parquet_out/a.parquet
```

如果使用自动匹配：

```bash
python examples/vl_projection.py --match-time 1644
```

---

### `examples/vl_projection_pyqt_tuner.py`

基于 PyQt 的投影调参工具。

适合：

- 交互式调节外参
- 调节时间偏移
- 测试不同过滤参数
- 现场快速观察投影效果

---


### `examples/video_rename.py`

用于批量重命名视频文件，方便按照时间戳或采集编号统一管理数据。

---

### `examples/query.py`

用于快速查询或调试某些点云、视频或协议字段。

---

### `examples/pcap_export_protocol_single.py`

单个 `pcap` 导出示例，适合单文件调试。

---

### `examples/pcap_export_protocol_all.py`

批量导出多个 `pcap` 文件的示例。

---

### `examples/visualization_bin_points.py`

用于可视化 `bin` 格式点云。

适合：

- 检查导出的二进制点云是否正确
- 查看是否符合预期格式

---

### `examples/demo_protocol_full_fields_export.py`

演示导出完整协议字段，便于查看解析结果是否完整。

---

## 常见工作流程

### 1. 从 `pcap` 到点云文件

1. 使用 `pcap_parser_protocol.py` 或示例脚本解析 `pcap`
2. 导出为 `parquet` 或 `bin`
3. 用 `visualize_single.py` 检查点云
4. 用 `vl_projection.py` 将点云投影到视频中

---

### 2. `parquet` 转 `csv`

单文件：

```bash
python -m icv_lidar_tools.convert_par_to_csv --input xxx.parquet --output xxx.csv
```

目录批量：

```bash
python -m icv_lidar_tools.convert_par_to_csv --input-dir G:/data/parquet_out --output-dir G:/data/csv_out --recursive
```

---

### 3. 点云可视化

```bash
python src/icv_lidar_tools/visualize_single.py
```

或者在代码中调用：

```python
from icv_lidar_tools.visualize_single import visualize_points

visualize_points("your.parquet", color_mode="intensity")
```

---

### 4. LiDAR 与视频投影融合

```bash
python examples/vl_projection.py --video your.mp4 --parquet your.parquet
```

---

## 数据字段说明

### 点云常见字段

项目中常见点云字段包括：

- `timestamp_ns`
- `x`
- `y`
- `z`
- `intensity`
- `ring`

### 协议导出字段

在 `pcap_parser_protocol.py` 的导出结果中，还可能包含：

- `packet_timestamp`
- `src_port`
- `dst_port`
- `packet_index`
- `block_index`
- `block_flag_hex`
- `azimuth_raw`
- `azimuth_deg`
- `distance_raw`
- `distance_m`
- `vertical_angle_deg`



---

## 推荐运行环境

- Python 3.10+
- Windows 10 / Windows 11
- `numpy`
- `pandas`
- `pyarrow`
- `scapy`
- `open3d`
- `opencv-python`

---

## 总结

这个项目目前可以概括为四类能力：

- **解析**：从 `pcap` 解码 LiDAR 原始数据
- **转换**：导出为 `parquet`、`bin`、`csv`
- **对齐**：进行时间戳同步与匹配
- **可视化/融合**：查看点云并投影到视频帧上

