from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parquet_to_csv(parquet_path: Path, csv_path: Path, chunk_size: int | None = None) -> None:
    """
    将单个 parquet 文件转换为 csv 文件。

    参数:
        parquet_path: 输入 parquet 文件路径
        csv_path: 输出 csv 文件路径
        chunk_size: 分块大小。为 None 时一次性读取；为整数时按分块写出，适合大文件
    """
    parquet_path = Path(parquet_path)
    csv_path = Path(csv_path)

    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet 文件不存在: {parquet_path}")

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if chunk_size is None:
        # 小文件：直接一次性读取
        df = pd.read_parquet(parquet_path)
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"转换完成: {parquet_path} -> {csv_path}")
        print(f"总行数: {len(df)}")
        return

    # 大文件：分块读取，避免内存占用过高
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet_path)
    first_write = True
    total_rows = 0

    for batch in pf.iter_batches(batch_size=chunk_size):
        df = batch.to_pandas()
        total_rows += len(df)

        df.to_csv(
            csv_path,
            mode="w" if first_write else "a",
            index=False,
            header=first_write,
            encoding="utf-8-sig",
        )
        first_write = False

    print(f"转换完成: {parquet_path} -> {csv_path}")
    print(f"总行数: {total_rows}")


def convert_parquet_dir_to_csv(
    input_dir: Path,
    output_dir: Path,
    chunk_size: int | None = None,
    recursive: bool = False,
) -> None:
    """批量转换目录下的所有 parquet 文件为 csv。"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"输入路径不是目录: {input_dir}")

    pattern = "**/*.parquet" if recursive else "*.parquet"
    parquet_files = sorted(input_dir.glob(pattern))
    if not parquet_files:
        print(f"未找到 parquet 文件: {input_dir} ({pattern})")
        return

    print(f"找到 {len(parquet_files)} 个 parquet 文件，开始转换...")
    for idx, parquet_path in enumerate(parquet_files, start=1):
        rel_path = parquet_path.relative_to(input_dir)
        csv_path = (output_dir / rel_path).with_suffix(".csv")
        print(f"[{idx}/{len(parquet_files)}] {parquet_path.name}")
        parquet_to_csv(parquet_path, csv_path, chunk_size)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="将 parquet 文件转换为 csv 文件")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", "-i", type=Path, help="输入 parquet 文件路径")
    group.add_argument("--input-dir", type=Path, help="输入 parquet 目录路径")

    parser.add_argument("--output", "-o", type=Path, help="输出 csv 文件路径（单文件模式）")
    parser.add_argument("--output-dir", type=Path, help="输出 csv 目录路径（目录模式）")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="分块大小，适合大文件；不填则一次性读取",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="目录模式下是否递归搜索子目录中的 parquet 文件",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.input is not None:
        if args.output is None:
            raise ValueError("单文件模式下必须指定 --output")
        parquet_to_csv(args.input, args.output, args.chunk_size)
        return

    if args.output_dir is None:
        raise ValueError("目录模式下必须指定 --output-dir")

    convert_parquet_dir_to_csv(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        recursive=args.recursive,
    )


if __name__ == "__main__":
    main()