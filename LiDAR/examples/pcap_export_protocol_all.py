from __future__ import annotations

from pathlib import Path
import re

from icv_lidar_tools.lidar import ProtocolPcapParser


def process_all_pcaps(
    input_dir: Path,
    output_dir: Path,
    payload_byteorder: str = "little",
    debug_stats: bool = False,
    use_cuda: bool = False,
) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    pcap_files = sorted(input_dir.glob("*.pcap"))
    if not pcap_files:
        print(f"未找到 pcap 文件: {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"共找到 {len(pcap_files)} 个 pcap 文件，开始处理...")

    parser = ProtocolPcapParser(payload_byteorder=payload_byteorder, debug_stats=debug_stats, use_cuda=use_cuda)

    for idx, pcap_path in enumerate(pcap_files, start=1):
        filename = pcap_path.stem
        match = re.search(r"(\d+)", filename)
        suffix = match.group(1) if match else "unknown"

        print(f"\n[{idx}/{len(pcap_files)}] 处理: {pcap_path.name}")

        parquet_out = output_dir / f"lidar_points_protocol_{suffix}.parquet"
        parser.export_parse_points(pcap_path, parquet_out, fmt="parquet")

        print(f"  输出 parquet: {parquet_out}")

    print("\n全部处理完成。")


if __name__ == "__main__":
    input_dir = Path(r"G:\data\2025-01-12\lidar")
    output_dir = Path(r"G:\data\2025-01-12\parquet_out_v2")

    process_all_pcaps(
        input_dir=input_dir,
        output_dir=output_dir,
        payload_byteorder="little",
        debug_stats=True,
        use_cuda=False,
    )
