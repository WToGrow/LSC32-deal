from pathlib import Path

from icv_lidar_tools.exporters import Exporter
from icv_lidar_tools.gnss import GnssParser
from icv_lidar_tools.visualization_gnss import plot_gnss_track


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    gnss_file = root / "source data" / "parsed_gnss_data.txt"
    out_dir = root / "outputs"

    records = GnssParser.parse_file(gnss_file)
    df = GnssParser.to_dataframe(records)

    Exporter.to_csv(df, out_dir / "gnss.csv")
    Exporter.to_excel(df, out_dir / "gnss.xlsx")
    plot_gnss_track(df, out_dir / "gnss_track.png")

    print(f"parsed {len(df)} gnss records")


if __name__ == "__main__":
    main()
