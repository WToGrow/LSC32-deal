# LSC32-deal

面向 Java Web 对接的激光雷达处理项目说明。

## 项目定位

本项目主要提供两类能力：

1. `pcap_parser_protocol.py`
   - 解析雷达 `pcap` 数据
   - 导出点云结果（如 `parquet`）
   - 适合 Java Web 后端异步调用

2. `vl_projection.py`
   - 读取视频和点云数据
   - 完成 LiDAR 点云到图像的投影叠加
   - 输出结果视频，支持自定义保存路径

## 目录概览

- `LiDAR/src/icv_lidar_tools/`
  - 核心库代码
- `LiDAR/examples/`
  - 示例脚本和可直接调用的处理入口
- `LiDAR/outputs/`
  - 输出结果目录（如果有）

## Java Web 对接建议

建议 Java Web 侧只调用两个 Python 能力：

### 1. pcap 解析

输入：

- `pcapPath`
- `outPath` 或 `outStem`
- `fmt`
- `formats`
- `binMode`
- `batchSize`
- `payloadByteorder`
- `debugStats`
- `useCuda`

输出：

- `parquet` 文件路径
- 或 `parquet + bin` 文件路径集合

### 2. 视频投影

输入：

- `videoPath`
- `parquetPath`
- `outputDir`
- `outputVideoPath`
- `timeWindowMs`
- `videoStartNs`
- `maxFrames`
- `drawOverlay`
- `mirrorX`
- `mirrorY`
- `mirrorZ`
- `rLidarToCam`
- `tWorldToCam`

输出：

- 叠加投影后的结果视频路径

## Python 侧推荐接口

### 解析接口

- `ProtocolPcapParser.parse_pcap(config)`

### 投影接口

- `run_projection(config)`

其中 `config` 建议使用 `dataclass` 封装，便于 Java Web 传参映射。

## 默认行为

### 点云解析
- 默认按分块读取
- 默认导出 `parquet`

### 视频投影
- 默认使用固定相机内参和畸变参数
- 默认允许镜像设置
- 默认输出为 `mp4`
- 支持 Java Web 指定自定义结果保存路径

## 输入输出示例

### pcap 解析 JSON 示例

```json
{
  "pcapPath": "D:/data/input.pcap",
  "outPath": "D:/data/output.parquet",
  "fmt": "parquet",
  "batchSize": 500000,
  "payloadByteorder": "little",
  "debugStats": false,
  "useCuda": false
}
```

### 投影 JSON 示例

```json
{
  "videoPath": "D:/data/input.mp4",
  "parquetPath": "D:/data/points.parquet",
  "outputDir": "D:/data/out",
  "outputVideoPath": "D:/data/out/result.mp4",
  "timeWindowMs": 150.0,
  "videoStartNs": 1710000000000000000,
  "maxFrames": 300,
  "drawOverlay": true,
  "mirrorX": true,
  "mirrorY": false,
  "mirrorZ": false,
  "rLidarToCam": [[...], [...], [...]],
  "tWorldToCam": [[...], [...], [...]]
}
```

## Java Web 交付建议

建议最终交付给 Java Web 的内容包括：

- Python 可执行脚本
- 接口参数说明
- 输出文件说明
- 示例 JSON
- 测试样本数据

## 备注

如果后续要做成真正的 Web 服务，建议再补一层：

- FastAPI / Flask HTTP 接口
- Java 通过 HTTP 调用 Python 服务
- Python 返回任务状态和结果路径
