from pathlib import Path

from icv_lidar_tools.lidar import ProtocolPcapParser


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    pcap_path = root / "source data" / "lidar_162920.pcap"
    out_stem = Path("G:/data/lidar_full_fields")

    if not pcap_path.exists():
        raise FileNotFoundError(f"pcap not found: {pcap_path}")

    parser = ProtocolPcapParser(payload_byteorder="little", debug_stats=True, use_cuda=False)
    outputs = parser.export_full_fields(
        pcap_path=pcap_path,
        out_stem=out_stem,
        formats=("parquet", "bin"),
        bin_mode="kitti",  # or "full_float64"
    )

    for fmt, path in outputs.items():
        print(f"full fields export done ({fmt}): {path}")


if __name__ == "__main__":
    main()
