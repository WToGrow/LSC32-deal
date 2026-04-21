# ICV LiDAR Tools 项目说明

## 1. 项目简介

本项目是一个围绕 LiDAR 点云、PCAP 协议解析、Parquet 导出、点云可视化以及 LiDAR-视频投影融合的 Python 工具集合。

项目代码主要分为两部分：

- `src/icv_lidar_tools/`：可安装的工具包源码，提供解析、对齐、可视化等基础能力
- `examples/`：各类脚本示例、调试脚本、实验脚本和投影脚本

适用场景：

- 解析雷达 `pcap` 数据并导出为 `parquet` / `bin`
- 将点云转换为 CSV
- 按时间戳对齐 GNSS / 其他传感器数据
- 可视化单帧或首个场景点云
- 将 LiDAR 点投影到视频帧上进行融合查看

---

## 2. 目录总览

仓库扫描到的主要文件如下（不含 `.idea/` 等 IDE 配置目录）：

- `pyproject.toml`
- `src/icv_lidar_tools/__init__.py`
- `src/icv_lidar_tools/models.py`
- `src/icv_lidar_tools/time_sync.py`
- `src/icv_lidar_tools/convert_par_to_csv.py`
- `src/icv_lidar_tools/visualize_single.py`
- `src/icv_lidar_tools/lidar/__init__.py`
- `src/icv_lidar_tools/lidar/pcap_parser_protocol.py`
- `examples/vl_projection.py`
- `examples/vl_projection_s.py`
- `examples/vl_projection acc.py`
- `examples/vl_projection_pyqt_tuner.py`
- `examples/vl_first_frame_check.py`
- `examples/video_rename.py`
- `examples/query.py`
- `examples/pcap_export_protocol_single.py`
- `examples/visualization_bin_points.py`
- `examples/demo_protocol_full_fields_export.py`

---

## 3. 项目依赖与安装

### 3.1 Python 版本

项目在 `pyproject.toml` 中要求：

- `Python >= 3.10`

### 3.2 主要依赖

- `numpy`
- `pandas`
- `pyarrow`
- `openpyxl`
- `matplotlib`
- `scapy`

其中部分脚本还会用到：

- `opencv-python`（视频叠加投影）
- `open3d`（点云可视化）

> 注：`open3d` 和 `opencv-python` 目前未写在 `pyproject.toml` 的基础依赖里，但从源码看某些示例脚本是需要它们的。

### 3.3 安装方式

在项目根目录执行：

```bash
pip install -e .
```

如果你需要运行可视化或视频投影脚本，建议额外安装：

```bash
pip install opencv-python open3d
```

---

## 4. 核心源码说明

### 4.1 `src/icv_lidar_tools/__init__.py`

作用：

- 定义包入口
- 暴露可直接导入的类

当前导出：

- `TimeAligner`

文件内容很短，主要用于包级别引用。

---

### 4.2 `src/icv_lidar_tools/models.py`

作用：

- 定义项目中的数据结构

主要内容：

- `LidarPoint`

字段：

- `timestamp`: 时间戳
- `x`, `y`, `z`: 三维坐标
- `intensity`: 反射强度
- `ring`: 线束编号

用途：

- 作为 PCAP 解析后的点云点数据模型
- 方便后续导出、组装、调试

---

### 4.3 `src/icv_lidar_tools/time_sync.py`

作用：

- 做时间戳对齐

核心类：

- `TimeAligner`

核心方法：

- `align_nearest(left_df, right_df, left_ts_col='timestamp', right_ts_col='timestamp', tolerance_ms=50)`

功能：

- 将两个 DataFrame 按时间戳最近邻方式对齐
- 内部使用 `pandas.merge_asof`
- 适合 GNSS、IMU、视频帧时间与点云时间的近似匹配

---

### 4.4 `src/icv_lidar_tools/convert_par_to_csv.py`

作用：

- 将 `parquet` 文件转换为 `csv`
- 支持单文件模式和目录批量模式

主要函数：

- `parquet_to_csv(parquet_path, csv_path, chunk_size=None)`
- `convert_parquet_dir_to_csv(input_dir, output_dir, chunk_size=None, recursive=False)`

命令行参数：

- `--input / -i`：单个 parquet 文件
- `--input-dir`：目录模式
- `--output / -o`：单文件输出路径
- `--output-dir`：目录模式输出目录
- `--chunk-size`：分块读取大小
- `--recursive`：递归搜索子目录

用途：

- 将点云数据转换成便于 Excel 或其他工具处理的 CSV 格式

---

### 4.5 `src/icv_lidar_tools/visualize_single.py`

作用：

- 使用 `open3d` 可视化点云
- 默认只显示首个场景/首个时间片的数据
- 支持按不同模式着色

主要能力：

- 自动检测时间列
- 读取首个时间片点云
- 支持颜色模式：
  - `intensity`
  - `height`
  - `distance`

核心函数：

- `_bright_colormap`
- `_normalize_with_percentile`
- `_detect_time_column`
- `_apply_color_mode`
- `visualize_points(parquet_path, color_mode='intensity')`

用途：

- 快速检查点云数据质量
- 验证强度/高度/距离颜色映射效果

注意：

- 该脚本依赖 `open3d`

---

### 4.6 `src/icv_lidar_tools/lidar/pcap_parser_protocol.py`

作用：

- 解析 LiDAR `pcap` 协议数据
- 将 UDP 包中的点云结构解码为结构化数据
- 支持导出 `parquet` / `bin`

核心类：

- `DecodedMsopPacket`
- `ParseConfig`
- `MsopDecoder`
- `ProtocolPcapParser`

重要特征：

- 面向 LeiShen C32 MSOP 协议
- 提取字段：
  - `packet_timestamp`
  - `azimuth_deg`
  - `ring`
  - `distance_m`
  - `intensity`
  - `x, y, z`
- 支持批量输出
- 支持 CUDA 开关（如果环境中安装了 `cupy`）

用途：

- 从原始 `pcap` 中提取点云并标准化存储

---

## 5. examples 目录脚本说明

### 5.1 `examples/vl_projection.py`

作用：

- LiDAR 点云投影到视频帧上
- 读取视频与点云 `parquet` 后，在视频中叠加投影点

特点：

- 支持按时间窗口抽取点云
- 支持图像边界过滤
- 支持边缘掩码过滤
- 支持颜色一致性过滤
- 支持深度时序过滤
- 支持按深度或强度着色（你目前正在扩展）
- 支持默认目录自动寻找视频和 parquet 文件

主要入口：

- `run_projection(config)`
- `main()`

常见命令示例：

```bash
python examples/vl_projection.py --video G:/data/video/a.mp4 --parquet G:/data/parquet_out/a.parquet
```

如果启用自动匹配：

```bash
python examples/vl_projection.py --match-time 1644
```

> 注：当前脚本中你已经做过多次实验性修改，建议后续再整理成稳定版参数接口。

---

### 5.2 `examples/vl_projection_s.py`

作用：

- 另一个投影脚本的变体
- 通常用于不同投影策略、不同过滤逻辑或实验参数

建议：

- 作为 `vl_projection.py` 的对照实验版本
- 如果要正式维护，建议整理成统一参数风格

---

### 5.3 `examples/vl_projection acc.py`

作用：

- 投影脚本的加速/实验版本
- 文件名中的 `acc` 通常表示 accelerated 或 accuracy-related experiment

建议：

- 适合保留为实验备份
- 若功能稳定，可考虑重命名为更规范的文件名

---

### 5.4 `examples/vl_projection_pyqt_tuner.py`

作用：

- 基于 PyQt 的投影调参工具
- 从文件名看，应该用于交互式调整投影参数、筛选参数或可视化参数

特点：

- GUI 交互式调整
- 适合现场调试相机外参、时间偏移、过滤阈值等

注意：

- 这类脚本通常依赖 `PyQt` / `PySide` 相关库

---

### 5.5 `examples/vl_first_frame_check.py`

作用：

- 检查视频或点云数据的首帧/首个场景是否正确

用途：

- 用于同步校验
- 判断数据是否从正确时间开始

---

### 5.6 `examples/video_rename.py`

作用：

- 批量重命名视频文件

用途：

- 配合时间戳命名规范整理数据集
- 方便后续自动匹配视频与点云

---

### 5.7 `examples/query.py`

作用：

- 看名字像是数据查询或快速验证脚本
- 可能用于测试点云、视频或协议数据中的某些字段

建议：

- 若需要更明确，可补充脚本头部注释或使用说明

---

### 5.8 `examples/pcap_export_protocol_single.py`

作用：

- 单个 `pcap` 文件导出脚本
- 可能与 `ProtocolPcapParser` 配套使用

用途：

- 将协议包转换为标准点云文件
- 适合单文件调试

---

### 5.9 `examples/visualization_bin_points.py`

作用：

- 可视化 `bin` 格式点云

用途：

- 用于检查导出的 `bin` 数据是否正确
- 常见于 KITTI 风格或类似格式的数据展示

---

### 5.10 `examples/demo_protocol_full_fields_export.py`

作用：

- 演示导出完整协议字段
- 可能用于把 UDP/协议解析结果导出成包含全部字段的表格

用途：

- 调试协议解析结果
- 检查字段完整性

---

## 6. 典型工作流程

### 6.1 从 PCAP 到点云文件

1. 使用 `pcap_parser_protocol.py` 或示例脚本解析 `pcap`
2. 导出 `parquet` 或 `bin`
3. 用 `visualize_single.py` 检查点云
4. 用 `vl_projection.py` 将点云投影到视频中

---

### 6.2 Parquet 转 CSV

```bash
python -m icv_lidar_tools.convert_par_to_csv --input xxx.parquet --output xxx.csv
```

目录批量转换：

```bash
python -m icv_lidar_tools.convert_par_to_csv --input-dir G:/data/parquet_out --output-dir G:/data/csv_out --recursive
```

---

### 6.3 点云可视化

```bash
python src/icv_lidar_tools/visualize_single.py
```

或者在代码里调用：

```python
from icv_lidar_tools.visualize_single import visualize_points
visualize_points("your.parquet", color_mode="intensity")
```

---

### 6.4 LiDAR 与视频投影融合

```bash
python examples/vl_projection.py --video your.mp4 --parquet your.parquet
```

如果使用自动匹配目录：

```bash
python examples/vl_projection.py --match-time 1644
```

---

## 7. 数据字段说明

### 7.1 点云常见字段

在项目中，点云/协议解析常见字段包括：

- `timestamp_ns`
- `x`
- `y`
- `z`
- `intensity`
- `ring`

### 7.2 协议导出完整字段

在 `pcap_parser_protocol.py` 中，导出结果还可能包含：

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

## 8. 使用注意事项

1. `vl_projection.py` 目前是实验性较强的脚本，里面包含多个过滤环节和历史修补逻辑。
2. `open3d`、`opencv-python` 等依赖需要手动安装。
3. 如果数据时间戳格式不统一，`TimeAligner` 和投影脚本的匹配结果可能受影响。
4. `pcap_parser_protocol.py` 目前偏向特定协议格式，使用前要确认你的雷达设备协议一致。
5. `examples/` 下的脚本大多是调试/实验用途，正式使用建议整理参数接口。

---

## 9. 建议的运行环境

- Python 3.10+
- Windows 10 / Windows 11
- `numpy`
- `pandas`
- `pyarrow`
- `opencv-python`
- `open3d`
- `scapy`

---

## 10. 总结

这个项目的核心能力可以概括为：

- **解析**：从 `pcap` 解出 LiDAR 原始数据
- **转换**：导出 `parquet`、`bin`、`csv`
- **对齐**：按时间戳对齐多源数据
- **可视化**：查看点云单帧或首个场景
- **融合**：把 LiDAR 点投影到视频帧上


