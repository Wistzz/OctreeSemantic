import time
import torch
import numpy as np
import open3d as o3d
import open3d as o3d
import numpy as np

import numpy as np
import numba

@numba.jit(nopython=True)
def numba_fps(points, ratio):
    num_points = points.shape[0]
    num_samples = int(num_points * ratio)
    sample_indices = np.zeros(num_samples, dtype=np.int32)
    distance = np.full(num_points, np.inf)
    # 随机选择第一个点
    first_index = np.random.randint(0, num_points)
    sample_indices[0] = first_index

    for i in range(1, num_samples):
        last_sample = points[sample_indices[i - 1]]
        dist = np.sum((points - last_sample) ** 2, axis=1)
        distance = np.minimum(distance, dist)
        sample_indices[i] = np.argmax(distance)

    return sample_indices

def open3d_fps(points, ratio):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    num_points = len(pcd.points)
    num_samples = int(num_points * ratio)
    sample_indices = pcd.farthest_point_down_sample(num_samples)
    return np.asarray(sample_indices.points)

def torch_fps(points, ratio):
    num_points = points.size(0)
    num_samples = int(num_points * ratio)
    sample_indices = torch.zeros(num_samples, dtype=torch.long, device=points.device)
    distance = torch.full((num_points,), float('inf'), device=points.device)
    first_index = torch.randint(0, num_points, (1,), device=points.device)
    sample_indices[0] = first_index

    for i in range(1, num_samples):
        last_sample = points[sample_indices[i - 1]]
        dist = torch.sum((points - last_sample) ** 2, dim=1)
        distance = torch.min(distance, dist)
        sample_indices[i] = torch.argmax(distance)

    return sample_indices
# 生成示例点云数据
points = np.random.rand(10000, 3)
torch_points = torch.from_numpy(points).float()
ratio = 0.2

# 测试 torch_cluster.fps
from torch_cluster import fps
start_time = time.time()
torch_cluster_indices = fps(torch_points, ratio=ratio)
torch_cluster_time = time.time() - start_time

# 测试 numba_fps
start_time = time.time()
numba_indices = numba_fps(points, ratio)
numba_time = time.time() - start_time

# 测试 torch_fps
start_time = time.time()
torch_indices = torch_fps(torch_points, ratio)
torch_time = time.time() - start_time

# 测试 open3d_fps
start_time = time.time()
open3d_points = open3d_fps(points, ratio)
open3d_time = time.time() - start_time

print(f"torch_cluster.fps 时间: {torch_cluster_time} 秒")
print(f"numba_fps 时间: {numba_time} 秒")
print(f"torch_fps 时间: {torch_time} 秒")
print(f"open3d_fps 时间: {open3d_time} 秒")