import torch
import numpy as np
from sklearn.cluster import DBSCAN

class DBSCANCluster():
    def __init__(self, eps=0.3, min_samples=10, dim=9):
        self.eps = eps  # DBSCAN 的邻域半径
        self.min_samples = min_samples  # DBSCAN 的最小样本数
        self.vec_dim = dim  # 特征维度
        self.centers = torch.empty(0)  # 聚类中心，形状为 [num_clusters, dim]
        self.cls_ids = torch.empty(0)  # 每个样本的聚类 ID，形状为 [num_pts]

    def get_dist(self, x, y, mode='sq_euclidean'):
        """计算 x 中所有向量与 y 中所有向量之间的距离。

        x: (m, dim)
        y: (n, dim)
        dist: (m, n)
        """
        if mode == 'sq_euclidean':
            dist = torch.cdist(x.unsqueeze(0).detach(), y.unsqueeze(0).detach())[0]
        return dist

    def cluster_assign(self, feat):
        """执行 DBSCAN 聚类。

        feat: (num_pts, dim)
        """
        # 转换为 numpy 数组以使用 sklearn 的 DBSCAN
        feat_np = feat.detach().cpu().numpy()

        # 创建 DBSCAN 模型并进行聚类
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(feat_np)
        self.cls_ids = torch.tensor(db.labels_, dtype=torch.long, device=feat.device)

        # 计算聚类中心
        unique_labels = torch.unique(self.cls_ids)
        num_clusters = len(unique_labels)
        self.centers = torch.zeros((num_clusters, self.vec_dim), device=feat.device)
        for i, label in enumerate(unique_labels):
            cluster_points = feat[self.cls_ids == label]
            if len(cluster_points) > 0:
                self.centers[i] = torch.mean(cluster_points, dim=0)

    def forward(self, gaussian, pos_weight=1.0):
        # 合并特征和坐标信息
        scale = pos_weight
        xyz_feat = gaussian._xyz.detach() * scale
        feat = torch.cat((gaussian._ins_feat, xyz_feat), dim=1)  # [N, 9]
        # feat = gaussian._ins_feat

        # 执行 DBSCAN 聚类
        self.cluster_assign(feat)