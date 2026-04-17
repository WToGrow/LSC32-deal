from pathlib import Path
from datetime import datetime, timedelta

# 视频目录
VIDEO_DIR = Path(r"G:\data\2025-01-12\video")

# 支持的视频后缀
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v", ".ts"}


def get_modify_time(path: Path) -> datetime:
    """
    获取文件修改时间。
    在 Windows 上，st_mtime 表示最后修改时间。
    """
    ts = path.stat().st_mtime
    return datetime.fromtimestamp(ts)


def build_new_name(path: Path) -> str:
    """
    规则：
    - 取文件修改时间作为前缀，格式：YYYYMMDD_HHMMSS
    - 保留原文件名第一个下划线后的部分
    例如：
        162906_右后_000.mp4 -> 20250112_162906_右后_000.mp4
    """
    modify_time = get_modify_time(path) - timedelta(minutes=5)
    time_prefix = modify_time.strftime("%Y%m%d_%H%M%S")

    stem = path.stem
    suffix = path.suffix

    # 保留第二个下划线后的部分
    parts = stem.split("_", 2)
    if len(parts) >= 3:
        tail = parts[2]
        new_stem = f"{time_prefix}_{tail}"
    elif len(parts) == 2:
        # 如果只有一个下划线，就保留第一个下划线后的部分
        tail = parts[1]
        new_stem = f"{time_prefix}_{tail}"
    else:
        # 如果原文件名没有下划线，就直接加上时间前缀
        new_stem = f"{time_prefix}_{stem}"

    return new_stem + suffix


def main():
    if not VIDEO_DIR.exists():
        print(f"目录不存在：{VIDEO_DIR}")
        return

    for file_path in VIDEO_DIR.iterdir():
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in VIDEO_EXTS:
            continue

        new_name = build_new_name(file_path)
        new_path = file_path.with_name(new_name)

        # 如果重名，自动加序号避免覆盖
        if new_path.exists():
            base = new_path.stem
            ext = new_path.suffix
            i = 1
            while True:
                candidate = file_path.with_name(f"{base}_{i}{ext}")
                if not candidate.exists():
                    new_path = candidate
                    break
                i += 1

        print(f"{file_path.name} -> {new_path.name}")
        file_path.rename(new_path)


if __name__ == "__main__":
    main()