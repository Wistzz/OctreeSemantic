import torch
import numpy as np
# import faiss
from torch_scatter import scatter_mean


class ClassicKMeans():
    def __init__(self, num_clusters=100, num_iters=100, dim=9, batch_size=1000):
        self.num_clusters = num_clusters  # 聚类的数量
        self.num_iters = num_iters  # K-Means 迭代次数
        self.vec_dim = dim  # 特征维度
        self.centers = torch.empty(0)  # 聚类中心，形状为 [num_clusters, dim]
        self.cls_ids = torch.empty(0)  # 每个样本的聚类 ID，形状为 [num_pts]
        self.batch_size = batch_size

    def get_dist(self, x, y, mode='sq_euclidean'):
        """计算 x 中所有向量与 y 中所有向量之间的距离。

        x: (m, dim)
        y: (n, dim)
        dist: (m, n)
        """
        if mode == 'sq_euclidean':
            num_batches = int(np.ceil(x.shape[0] / self.batch_size))
            dist_list = []
            for i in range(num_batches):
                start = i * self.batch_size
                end = min((i + 1) * self.batch_size, x.shape[0])
                batch_x = x[start:end]
                with torch.no_grad():
                    batch_dist = torch.cdist(batch_x, y)
                dist_list.append(batch_dist)
            dist = torch.cat(dist_list, dim=0)
        return dist

    def cluster_assign(self, feat):
        """执行 K-Means 聚类。

        feat: (num_pts, dim)
        """
        # 初始化聚类中心
        if len(self.centers) == 0:
            with torch.no_grad():
                self.centers = feat[torch.randperm(feat.shape[0])[:self.num_clusters], :]

        for iteration in range(self.num_iters):
            # 计算每个样本到所有聚类中心的距离
            with torch.no_grad():
                dist = self.get_dist(feat, self.centers)
            # 分配样本到最近的聚类中心
            self.cls_ids = torch.argmin(dist, dim=-1)

            # 更新聚类中心
            for i in range(self.num_clusters):
                cluster_points = feat[self.cls_ids == i]
                if len(cluster_points) > 0:
                    with torch.no_grad():
                        self.centers[i] = torch.mean(cluster_points, dim=0)

    def forward(self, gaussian, pos_weight=1.0):
        # 合并特征和坐标信息
        scale = pos_weight
        with torch.no_grad():
            xyz_feat = gaussian._xyz * scale
            feat = torch.cat((gaussian._ins_feat, xyz_feat), dim=1)  # [N, 9]

        # 执行 K-Means 聚类
        self.cluster_assign(feat)







import torch
import numpy as np
from sklearn.cluster import DBSCAN

class DBSCAN_Clustering:
    def __init__(self, batch_size=1000):
        self.centers = torch.empty(0)
        self.cls_ids = torch.empty(0)
        self.eps = 0.5
        self.min_samples = 500
        self.batch_size = batch_size

    def get_dist(self, x, y, mode='sq_euclidean'):
        """计算 x 中所有向量与 y 中所有向量之间的距离。

        x: (m, dim)
        y: (n, dim)
        dist: (m, n)
        """
        if mode == 'sq_euclidean':
            num_batches = int(np.ceil(x.shape[0] / self.batch_size))
            dist_list = []
            for i in range(num_batches):
                start = i * self.batch_size
                end = min((i + 1) * self.batch_size, x.shape[0])
                batch_x = x[start:end]
                with torch.no_grad():
                    batch_dist = torch.cdist(batch_x, y)
                dist_list.append(batch_dist)
            dist = torch.cat(dist_list, dim=0)
        return dist

    def cluster_assign(self, feat):
        """执行 DBSCAN 聚类。

        feat: (num_pts, dim)
        """
        # DBSCAN聚类
        feat_np = feat.cpu().numpy()
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)  # 参数需调整
        labels = dbscan.fit_predict(feat_np)
        
        # 处理噪声点（labels为-1）
        noise_mask = labels == -1
        if np.any(noise_mask):
            labels[noise_mask] = -1  # 噪声点保持-1
        
        # 计算聚类中心
        unique_labels = np.unique(labels[labels != -1])
        centers = []
        for label in unique_labels:
            cluster_points = feat_np[labels == label]
            center = np.mean(cluster_points, axis=0)
            centers.append(center)
        
        # 更新属性
        if centers:
            self.centers = torch.tensor(centers, device=feat.device)
            self.cls_ids = torch.tensor(labels, device=feat.device, dtype=torch.long)
        else:
            self.centers = torch.empty(0, device=feat.device)
            self.cls_ids = torch.full((feat.size(0),), -1, device=feat.device, dtype=torch.long)

    def forward(self, gaussian, pos_weight=1.0):
        # 合并特征和坐标信息
        scale = pos_weight
        with torch.no_grad():
            xyz_feat = gaussian._xyz * scale
            # xyz_norm = (xyz - xyz.mean(dim=0)) / (xyz.std(dim=0) + 1e-6)  # 归一化位置
            # feat_norm = (feat - feat.mean(dim=0)) / (feat.std(dim=0) + 1e-6)  # 归一化特征
            feat = torch.cat((gaussian._ins_feat, xyz_feat), dim=1)  # [N, 9]

        # 执行 DBSCAN 聚类
        self.cluster_assign(feat)


from hdbscan import HDBSCAN  # 需要安装 hdbscan: pip install hdbscan

class HDBSCAN_Clustering:
    def __init__(self, batch_size=1000):
        self.centers = torch.empty(0)  # 存储聚类中心
        self.cls_ids = torch.empty(0)  # 存储样本标签
        self.min_cluster_size = 500  # HDBSCAN 的最小簇大小，类似于 min_samples
        self.min_samples = 50        # 用于软聚类的参数，控制噪声点
        self.batch_size = batch_size

    def get_dist(self, x, y, mode='sq_euclidean'):
        """计算 x 中所有向量与 y 中所有向量之间的距离。

        x: (m, dim)
        y: (n, dim)
        dist: (m, n)
        """
        if mode == 'sq_euclidean':
            num_batches = int(np.ceil(x.shape[0] / self.batch_size))
            dist_list = []
            for i in range(num_batches):
                start = i * self.batch_size
                end = min((i + 1) * self.batch_size, x.shape[0])
                batch_x = x[start:end]
                with torch.no_grad():
                    batch_dist = torch.cdist(batch_x, y)
                dist_list.append(batch_dist)
            dist = torch.cat(dist_list, dim=0)
        return dist

    def cluster_assign(self, feat):
        """执行 HDBSCAN 聚类。

        feat: (num_pts, dim)
        """
        # 将特征转换为 numpy 数组以供 HDBSCAN 使用
        feat_np = feat.cpu().numpy()
        
        # HDBSCAN 聚类
        hdbscan = HDBSCAN(min_cluster_size=self.min_cluster_size, 
                         min_samples=self.min_samples, 
                         cluster_selection_method='eom')  # 'eom' 表示 Extracting Optimal Clusters
        labels = hdbscan.fit_predict(feat_np)
        
        # 处理噪声点（labels 为 -1）
        noise_mask = labels == -1
        if np.any(noise_mask):
            labels[noise_mask] = -1  # 噪声点保持 -1
        
        # 计算聚类中心
        unique_labels = np.unique(labels[labels != -1])
        centers = []
        for label in unique_labels:
            cluster_points = feat_np[labels == label]
            center = np.mean(cluster_points, axis=0)
            centers.append(center)
        
        # 更新属性
        if centers:
            centers_np = np.array(centers)
            labels_np = np.array(labels)
            # 然后将 numpy.ndarray 转换为 PyTorch 张量
            self.centers = torch.tensor(centers_np, device=feat.device)
            self.cls_ids = torch.tensor(labels_np, device=feat.device, dtype=torch.long)
        else:
            self.centers = torch.empty(0, device=feat.device)
            self.cls_ids = torch.full((feat.size(0),), -1, device=feat.device, dtype=torch.long)

    def forward(self, gaussian, pos_weight=1.0):
        # 合并特征和坐标信息
        scale = pos_weight
        with torch.no_grad():
            xyz_feat = gaussian._xyz * scale
            feat = torch.cat((gaussian._ins_feat, xyz_feat), dim=1)  # [N, 9]

        # 执行 HDBSCAN 聚类
        self.cluster_assign(feat)

# import torch
# import torch.nn.functional as F
# from torch_cluster import knn
# import torch_scatter
# import heapq


# class EnhancedCluster:
#     def __init__(self, num_clusters=64, k_edges=16, semantic_ratio=0.7, chunk_size=512, pos_weight=0.5):
#         self.num_clusters = num_clusters
#         self.k_edges = k_edges
#         self.semantic_ratio = semantic_ratio
#         self.k_sem = max(2, int(k_edges * semantic_ratio))
#         self.k_geo = k_edges - self.k_sem
#         self.chunk_size = chunk_size
#         self.pos_weight = pos_weight
#         self.centers = None
#         self.cls_ids = None

#     def _kmeans_chunked(self, xyz, sem_feat, num_clusters=5000, max_iters=10):
#         """分块 K-Means 粗聚类，结合语义和坐标信息"""
#         num_points = sem_feat.size(0)
#         # 随机初始化中心点
#         center_indices = torch.randperm(num_points)[:num_clusters]
#         centers_xyz = xyz[center_indices]
#         centers_sem = sem_feat[center_indices]

#         for iter in range(max_iters):
#             labels = []
#             for i in range(0, num_points, self.chunk_size):
#                 chunk_xyz = xyz[i:i + self.chunk_size]
#                 chunk_sem = sem_feat[i:i + self.chunk_size]

#                 # 计算语义距离
#                 sem_dists = torch.cdist(chunk_sem, centers_sem)
#                 # 计算坐标距离
#                 geo_dists = torch.cdist(chunk_xyz, centers_xyz)

#                 # 结合语义和坐标距离
#                 dists = self.pos_weight * geo_dists + (1 - self.pos_weight) * sem_dists

#                 chunk_labels = torch.argmin(dists, dim=1)
#                 labels.append(chunk_labels)
#             labels = torch.cat(labels)

#             # 更新中心点
#             new_centers_xyz = torch_scatter.scatter_mean(xyz, labels, dim=0, dim_size=num_clusters)
#             new_centers_sem = torch_scatter.scatter_mean(sem_feat, labels, dim=0, dim_size=num_clusters)

#             centers_xyz = new_centers_xyz
#             centers_sem = new_centers_sem

#         return labels, centers_xyz, centers_sem

#     def _build_similarity_graph(self, centers_xyz, centers_sem):
#         """构建相似度图，结合语义和坐标信息"""
#         num_clusters = centers_sem.size(0)
#         sem_sim_matrix = F.cosine_similarity(centers_sem.unsqueeze(1), centers_sem.unsqueeze(0), dim=2)
#         geo_dists = torch.cdist(centers_xyz, centers_xyz)
#         geo_sim_matrix = 1 / (1 + geo_dists)  # 将距离转换为相似度

#         # 结合语义和坐标相似度
#         sim_matrix = self.pos_weight * geo_sim_matrix + (1 - self.pos_weight) * sem_sim_matrix

#         # 构建优先队列
#         priority_queue = []
#         for i in range(num_clusters):
#             for j in range(i + 1, num_clusters):
#                 heapq.heappush(priority_queue, (-sim_matrix[i, j].item(), (i, j)))
#         return priority_queue

#     def _merge_clusters(self, priority_queue, labels, xyz, sem_feat, num_clusters=64):
#         """合并聚类结果"""
#         current_clusters = len(torch.unique(labels))
#         while current_clusters > num_clusters:
#             # 取出相似度最高的节点对
#             _, (src, dst) = heapq.heappop(priority_queue)
#             # 合并两个聚类
#             labels[labels == dst] = src
#             current_clusters = len(torch.unique(labels))

#             # 检查标签范围
#             min_label = labels.min().item()
#             max_label = labels.max().item()
#             if min_label < 0 or max_label >= num_clusters:
#                 print(f"Labels out of range: min={min_label}, max={max_label}")
#                 labels[labels < 0] = 0
#                 labels[labels >= num_clusters] = num_clusters - 1

#             # 更新优先队列
#             new_priority_queue = []
#             for sim, (i, j) in priority_queue:
#                 if i != dst and j != dst:
#                     heapq.heappush(new_priority_queue, (sim, (i, j)))
#             priority_queue = new_priority_queue

#         # 重新计算合并后的中心点
#         new_centers_xyz = torch_scatter.scatter_mean(xyz, labels, dim=0, dim_size=num_clusters)
#         new_centers_sem = torch_scatter.scatter_mean(sem_feat, labels, dim=0, dim_size=num_clusters)

#         # 检查是否有空聚类
#         cluster_counts = torch.bincount(labels, minlength=num_clusters)
#         valid_clusters = cluster_counts > 0
#         new_centers_xyz = new_centers_xyz[valid_clusters]
#         new_centers_sem = new_centers_sem[valid_clusters]

#         return labels, new_centers_xyz, new_centers_sem

#     def forward(self, gaussian_data):
#         # 分离输入数据
#         xyz = gaussian_data._xyz.detach()
#         sem_feat = gaussian_data._ins_feat.detach()

#         if len(xyz) != len(sem_feat):
#             print("Data length mismatch: xyz and sem_feat have different lengths.")
#             return None

#         # 第一阶段：粗聚类
#         coarse_labels, centers_xyz, centers_sem = self._kmeans_chunked(xyz, sem_feat)

#         # 第二阶段：构建相似度图
#         priority_queue = self._build_similarity_graph(centers_xyz, centers_sem)

#         # 第三阶段：合并聚类结果
#         final_labels, final_centers_xyz, final_centers_sem = self._merge_clusters(priority_queue, coarse_labels, xyz,
#                                                                                    sem_feat, self.num_clusters)

#         # 存储最终的中心和样本标签
#         self.centers = final_centers_xyz#, final_centers_sem)
#         self.cls_ids = final_labels
#         return final_labels






















