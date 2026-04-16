import open3d as o3d
import numpy as np

# 读取原始点云
points = np.fromfile("G:/data/lidar_points_protocol.bin", dtype=np.float32).reshape(-1, 4)
xyz = points[:, :3]  # 取 XYZ

# 创建点云对象
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)

# ===================== 降采样：体素网格滤波，让点更稀疏 =====================
# voxel_size 越小保留点越多，越大越稀疏  0.1 = 10cm
pcd = pcd.voxel_down_sample(voxel_size=0.5)  
print("降采样完成")

# ===================== 去噪：统计滤波，移除飞点/噪点 =====================
# nb_neighbors：邻域点数   std_ratio：滤波强度（越大保留越多）
pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
print("去噪完成")

# 显示
o3d.visualization.draw_geometries([pcd])