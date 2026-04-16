import pandas as pd
import numpy as np

# 你的 parquet 文件路径
df = pd.read_parquet("G:/data/fused_lidar_gnss.parquet")

# 打印前 5 行
print("==== parquet ====")
print("==== 前5行数据 ====")
print(df.head())

# 查看总点数
print(f"\n总点数：{len(df)}")

# 查看字段信息
print("\n==== 字段 ====")
print(df.columns.tolist())

# 查看 XYZ 范围
print("\n==== XYZ 范围 ====")
print(df[["x", "y", "z"]].describe())



# 读取 bin
points = np.fromfile("G:/data/lidar_points_protocol.bin", dtype=np.float32).reshape(-1, 4)
print("\n")
print("==== parquet ====")
print("形状 (N, 4)：", points.shape)
print("前5个点：")
print(points[:5])

print("\nX 范围：", points[:,0].min(), points[:,0].max())
print("Y 范围：", points[:,1].min(), points[:,1].max())
print("Z 范围：", points[:,2].min(), points[:,2].max())