import torch
import numpy as np

class XMeans:
    def __init__(self, k_init=2, k_max=100, num_iters=1000, dim=9):
        """
        初始化 X-Means 算法
        :param k_init: 初始簇数量
        :param k_max: 最大簇数量
        :param num_iters: 每次 K-Means 迭代次数
        :param dim: 特征维度
        """
        self.k_init = k_init  # 初始聚类数量
        self.k_max = k_max    # 最大聚类数量
        self.num_iters = num_iters  # K-Means 迭代次数
        self.vec_dim = dim    # 特征维度
        self.centers = torch.empty(0)  # 聚类中心，形状为 [num_clusters, dim]
        self.cls_ids = torch.empty(0)  # 每个样本的聚类 ID，形状为 [num_pts]
        self.k_final = None   # 最终的簇数量

    def get_dist(self, x, y, mode='sq_euclidean'):
        """计算 x 中所有向量与 y 中所有向量之间的距离。

        x: (m, dim)
        y: (n, dim)
        dist: (m, n)
        """
        if mode == 'sq_euclidean':
            dist = torch.cdist(x.unsqueeze(0).detach(), y.unsqueeze(0).detach())[0]
        return dist

    def _kmeans_step(self, feat, centers):
        """执行一次 K-Means 迭代并返回标签和中心"""
        for _ in range(self.num_iters):
            dist = self.get_dist(feat, centers)
            cls_ids = torch.argmin(dist, dim=-1)
            new_centers = torch.zeros_like(centers)
            for i in range(centers.shape[0]):
                cluster_points = feat[cls_ids == i]
                if len(cluster_points) > 0:
                    new_centers[i] = torch.mean(cluster_points, dim=0)
                else:
                    new_centers[i] = centers[i]  # 如果簇为空，保持不变
            centers = new_centers
        dist = self.get_dist(feat, centers)
        cls_ids = torch.argmin(dist, dim=-1)
        return cls_ids, centers

    def _compute_bic(self, feat, cls_ids, centers):
        """计算贝叶斯信息准则 (BIC)"""
        n_samples, n_features = feat.shape
        n_clusters = centers.shape[0]

        # 计算簇内方差
        variance = 0
        for i in range(n_clusters):
            cluster_points = feat[cls_ids == i]
            if len(cluster_points) > 0:
                distances = torch.sum((cluster_points - centers[i]) ** 2)
                variance += distances
        variance /= (n_samples - n_clusters) if n_samples > n_clusters else 1e-6  # 避免除以 0

        # BIC 计算
        log_likelihood = -0.5 * n_samples * (n_features * np.log(2 * np.pi) + n_features * torch.log(variance + 1e-6) + 1)
        n_params = n_clusters * (n_features + 1)  # 每个簇有中心 (n_features) 和方差 (1)
        bic = -2 * log_likelihood + n_params * np.log(n_samples)
        return bic

    def cluster_assign(self, feat):
        """执行 X-Means 聚类。

        feat: (num_pts, dim)
        """
        # 初始化聚类中心
        if len(self.centers) == 0:
            perm = torch.randperm(feat.shape[0])
            self.centers = feat[perm[:self.k_init], :]

        current_k = self.k_init
        cls_ids, centers = self._kmeans_step(feat, self.centers)

        while current_k < self.k_max:
            new_centers = []
            new_cls_ids = cls_ids.clone()
            split_occurred = False

            # 对每个簇进行分裂测试
            for cluster_idx in range(current_k):
                cluster_points = feat[cls_ids == cluster_idx]
                if len(cluster_points) < 2:  # 如果簇内点太少，跳过
                    new_centers.append(centers[cluster_idx])
                    continue

                # 在当前簇上运行 k=2 的 K-Means
                init_centers = cluster_points[torch.randperm(len(cluster_points))[:2]]
                sub_cls_ids, sub_centers = self._kmeans_step(cluster_points, init_centers)

                # 计算 BIC
                bic_parent = self._compute_bic(cluster_points, torch.zeros(len(cluster_points), device=feat.device), centers[cluster_idx].unsqueeze(0))
                bic_children = self._compute_bic(cluster_points, sub_cls_ids, sub_centers)

                # 如果分裂后的 BIC 更优，则分裂簇
                if bic_children < bic_parent:
                    split_occurred = True
                    new_centers.extend(sub_centers)
                    # 更新全局标签
                    mask = cls_ids == cluster_idx
                    new_cls_ids[mask] = sub_cls_ids + current_k
                    current_k += 1
                else:
                    new_centers.append(centers[cluster_idx])

            if not split_occurred:
                break  # 如果没有分裂，停止

            # 更新全局中心和标签
            centers = torch.stack(new_centers)
            cls_ids = new_cls_ids
            current_k = len(centers)

            # 如果还需要继续，重新运行 K-Means
            if current_k < self.k_max:
                cls_ids, centers = self._kmeans_step(feat, centers)

        self.centers = centers
        self.cls_ids = cls_ids
        self.k_final = current_k

    def forward(self, gaussian, pos_weight=1.0):
        # 合并特征和坐标信息
        scale = pos_weight
        xyz_feat = gaussian._xyz.detach() * scale
        feat = torch.cat((gaussian._ins_feat, xyz_feat), dim=1)  # [N, 9]

        # 执行 X-Means 聚类
        self.cluster_assign(feat)