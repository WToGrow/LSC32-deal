from pathlib import Path
import re

from icv_lidar_tools.lidar import ProtocolPcapParser


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    # pcap_path = root / "source data" / "lidar_162920.pcap"
    pcap_path = Path("G:/data/2025-01-12/lidar/lidar_162920.pcap")
    # out_dir = Path("G:/data/parquet")
    out_dir = Path("G:/data/parquet_v3")

    out_dir.mkdir(parents=True, exist_ok=True)


    if not pcap_path.exists():
        raise FileNotFoundError(f"pcap file not found: {pcap_path}")

    filename = pcap_path.stem  # 获取无后缀文件名: lidar_162920
    match = re.search(r'(\d+)', filename)
    suffix = match.group(1) if match else "unknown"

    parser = ProtocolPcapParser(payload_byteorder="little", debug_stats=True, use_cuda=True)

    parquet_out = out_dir / f"lidar_points_protocol_{suffix}.parquet"
    # bin_out = out_dir / f"lidar_points_protocol_{suffix}.bin"

    parser.export_parse_points(pcap_path, parquet_out, fmt="parquet")

    # df = parser.parse_points_dataframe(pcap_path)
    # parser.export_bin_from_points_df(df, bin_out, mode="kitti")

    print(f"pcap protocol export done: {parquet_out}")
    # print(f"pcap protocol export done: {bin_out}")


if __name__ == "__main__":
    main()
