#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import math
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
# from simple_knn._C import distCUDA2   # no need
from scipy.spatial import KDTree        # modify
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation, knn, reduce
from torch_scatter import scatter_max
from einops import repeat
import time

def sigmoid(x):  
    return 1 / (1 + np.exp(-x))  

def distCUDA2(points):
    '''
    https://github.com/graphdeco-inria/gaussian-splatting/issues/292
    '''
    points_np = points.detach().cpu().float().numpy()
    dists, inds = KDTree(points_np).query(points_np, k=4)
    meanDists = (dists[:, 1:] ** 2).mean(1)

    return torch.tensor(meanDists, dtype=points.dtype, device=points.device)

class GaussianModel:

    def setup_functions(self):
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize
        
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, opt=None, **model_kwargs):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self.update_depth = opt.update_depth
        self.update_hierachy_factor = opt.update_hierachy_factor
        self.update_init_factor = opt.update_init_factor
        self.voxel_size = opt.voxel_size

        for key, value in model_kwargs.items():
            setattr(self, key, value)
        
        self._anchor = torch.empty(0)
        self._offset = torch.empty(0)
        self._level = torch.empty(0)
        self._extra_level = torch.empty(0)
        self._anchor_feat = torch.empty(0)
        self._anchor_feat_semantic =  torch.empty(0)
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.opacity_accum = torch.empty(0)
        self.anchor_demon = torch.empty(0)
        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)      
        self._ins_feat = torch.empty(0)     # Continuous instance features before quantization
        self._ins_feat_q = torch.empty(0)   # Discrete instance features after quantization
        self.iClusterSubNum = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        # self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        self.dist_ratio=0.999
        self.init_level=-1
        self.levels=-1
        self.fork=2
        self.extend=1.1
        self.base_layer=11
        self.padding=0.0
        self.progressive=True
        self.visible_threshold=0.9
        self.dist2level='round'
        self.n_offsets=10
        self.feat_dim=32
        self.view_dim=3
        self.semantic_dim=6
        self.appearance_dim=0
        self.use_feat_bank=False
        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(self.view_dim, self.feat_dim),
                nn.ReLU(True),
                nn.Linear(self.feat_dim, 3),
                nn.Softmax(dim=1)
            ).cuda()
            
        self.mlp_opacity = nn.Sequential(
            nn.Linear(self.feat_dim+self.view_dim, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, self.n_offsets),
            nn.Sigmoid()
            # nn.Tanh()
        ).cuda()
        
        self.mlp_cov = nn.Sequential(
            nn.Linear(self.feat_dim+self.view_dim, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, 7*self.n_offsets),
        ).cuda()
    
        self.mlp_color = nn.Sequential(
            nn.Linear(self.feat_dim+self.view_dim, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, 3*self.n_offsets),
            nn.Sigmoid()
        ).cuda()

        self.mlp_semantic = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, self.semantic_dim*self.n_offsets),
            # nn.Tanh()
            # nn.Sigmoid()
        ).cuda()

    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()
        self.mlp_semantic.eval()
        if self.use_feat_bank:
            self.mlp_feature_bank.eval()
        if self.appearance_dim > 0:
            self.embedding_appearance.eval()

    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color.train()
        self.mlp_semantic.train()
        if self.use_feat_bank:                   
            self.mlp_feature_bank.train()
        if self.appearance_dim > 0:
            self.embedding_appearance.train()

    def capture(self):
        param_dict = {}
        param_dict['optimizer'] = self.optimizer.state_dict()
        param_dict['opacity_mlp'] = self.mlp_opacity.state_dict()
        param_dict['cov_mlp'] = self.mlp_cov.state_dict()
        param_dict['color_mlp'] = self.mlp_color.state_dict()
        param_dict['semantic_mlp'] = self.mlp_semantic.state_dict()
        if self.use_feat_bank:
            param_dict['feature_bank_mlp'] = self.mlp_feature_bank.state_dict()
        if self.appearance_dim > 0:
            param_dict['appearance'] = self.embedding_appearance.state_dict()
        return (
            self.active_sh_degree,
            # self._xyz,
            self._anchor,
            self._offset,
            self._level,
            self._extra_level,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._ins_feat,     # Continuous instance features before quantization
            self._ins_feat_q,   # Discrete instance features after quantization
            self.max_radii2D,
            # self.xyz_gradient_accum,
            # self.denom,
            # self.opacity_accum, 
            self.anchor_demon,
            # self.offset_gradient_accum,
            # self.offset_denom,
            self.optimizer.state_dict(),
            param_dict,
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        # self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self._anchor,
        self._offset,
        self._level,
        self._extra_level,
        self.opacity_accum, 
        self.anchor_demon,
        self.offset_gradient_accum,
        self.offset_denom,
        self._ins_feat,     # Continuous instance features before quantization
        self._ins_feat_q,   # Discrete instance features after quantization
        self.max_radii2D, 
        # xyz_gradient_accum, 
        denom,
        opt_dict, 
        param_dict,
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        # self.xyz_gradient_accum = xyz_gradient_accum
        # self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        self.mlp_opacity.load_state_dict(param_dict['opacity_mlp'])
        self.mlp_cov.load_state_dict(param_dict['cov_mlp'])
        self.mlp_color.load_state_dict(param_dict['color_mlp'])
        if self.use_feat_bank:
            self.mlp_feature_bank.load_state_dict(param_dict['feature_bank_mlp'])
        if self.appearance_dim > 0:
            self.embedding_appearance.load_state_dict(param_dict['appearance'])

    @property
    def get_anchor(self):
        return self._anchor

    @property
    def get_level(self):
        return self._level
    
    @property
    def get_extra_level(self):
        return self._extra_level

    @property
    def get_anchor_feat(self):
        return self._anchor_feat
    
    @property
    def get_anchor_feat_semantic(self):
        return self._anchor_feat_semantic
    
    @property
    def get_offset(self):
        return self._offset

    @property
    def get_scaling(self):
        return 1.0*self.scaling_activation(self._scaling)
    
    @property
    def get_scaling_origin(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_rotation_matrix(self):
        return build_rotation(self._rotation)
    
    @property
    def get_eigenvector(self):
        scales = self.get_scaling_origin
        N = scales.shape[0]
        idx = torch.min(scales, dim=1)[1]
        normals = self.get_rotation_matrix[np.arange(N), :, idx]
        normals = torch.nn.functional.normalize(normals, dim=1)
        return normals
    
    # @property
    # def get_xyz(self):
    #     return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity   

    @property
    def get_cov_mlp(self):
        return self.mlp_cov
    
    @property
    def get_color_mlp(self):
        return self.mlp_color
    
    @property
    def get_semantic_mlp(self):
        return self.mlp_semantic

    @property
    def get_featurebank_mlp(self):
        return self.mlp_feature_bank

    # NOTE: get instance feature
    # @property
    def get_ins_feat(self, origin=False):
        if len(self._ins_feat_q) == 0 or origin:
            ins_feat = self._ins_feat
        else:
            ins_feat = self._ins_feat_q
        ins_feat = torch.nn.functional.normalize(ins_feat, dim=1)
        return ins_feat
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def set_coarse_interval(self, opt):
        self.coarse_intervals = []
        num_level = self.levels - 1 - self.init_level
        if num_level > 0:
            q = 1/opt.coarse_factor
            a1 = opt.coarse_iter*(1-q)/(1-q**num_level)
            temp_interval = 0
            for i in range(num_level):
                interval = a1 * q ** i + temp_interval
                temp_interval = interval
                self.coarse_intervals.append(interval)

    def set_level(self, points, cameras, scales):
        all_dist = torch.tensor([]).cuda()
        self.cam_infos = torch.empty(0, 4).float().cuda()
        for scale in scales:
            for cam in cameras[scale]:
                cam_center = cam.camera_center
                cam_info = torch.tensor([cam_center[0], cam_center[1], cam_center[2], scale]).float().cuda()
                self.cam_infos = torch.cat((self.cam_infos, cam_info.unsqueeze(dim=0)), dim=0)
                dist = torch.sqrt(torch.sum((points - cam_center)**2, dim=1))
                dist_max = torch.quantile(dist, self.dist_ratio)
                dist_min = torch.quantile(dist, 1 - self.dist_ratio)
                new_dist = torch.tensor([dist_min, dist_max]).float().cuda()
                new_dist = new_dist * scale
                all_dist = torch.cat((all_dist, new_dist), dim=0)
        dist_max = torch.quantile(all_dist, self.dist_ratio)
        dist_min = torch.quantile(all_dist, 1 - self.dist_ratio)
        self.standard_dist = dist_max
        if self.levels == -1:
            self.levels = torch.round(torch.log2(dist_max/dist_min)/math.log2(self.fork)).int().item() + 1
        if self.init_level == -1:
            self.init_level = int(self.levels/2)

    def map_to_int_level(self, pred_level, cur_level):
        if self.dist2level=='floor':
            int_level = torch.floor(pred_level).int()
            int_level = torch.clamp(int_level, min=0, max=cur_level)
        elif self.dist2level=='round':
            int_level = torch.round(pred_level).int()
            int_level = torch.clamp(int_level, min=0, max=cur_level)
        elif self.dist2level=='ceil':
            int_level = torch.ceil(pred_level).int()
            int_level = torch.clamp(int_level, min=0, max=cur_level)
        elif self.dist2level=='progressive':
            pred_level = torch.clamp(pred_level+1.0, min=0.9999, max=cur_level + 0.9999)
            int_level = torch.floor(pred_level).int()
            self._prog_ratio = torch.frac(pred_level).unsqueeze(dim=1)
            self.transition_mask = (self._level.squeeze(dim=1) == int_level)
        else:
            raise ValueError(f"Unknown dist2level: {self.dist2level}")
        
        return int_level
    
    def octree_sample(self, data):
        torch.cuda.synchronize(); t0 = time.time()
        self.positions = torch.empty(0, 3).float().cuda()
        self._level = torch.empty(0).int().cuda() 
        for cur_level in range(self.levels):
            cur_size = self.voxel_size/(float(self.fork) ** cur_level)
            new_positions = torch.unique(torch.round((data - self.init_pos) / cur_size), dim=0) * cur_size + self.init_pos
            new_positions += self.padding * cur_size
            new_level = torch.ones(new_positions.shape[0], dtype=torch.int, device="cuda") * cur_level
            self.positions = torch.concat((self.positions, new_positions), dim=0)
            self._level = torch.concat((self._level, new_level), dim=0)
        torch.cuda.synchronize(); t1 = time.time()
        time_diff = t1 - t0
        print(f"Building octree time: {int(time_diff // 60)} min {time_diff % 60} sec")


    # def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, *args):
    #     points = torch.tensor(pcd.points, dtype=torch.float, device="cuda")
    #     # colors = torch.tensor(pcd.colors, dtype=torch.float, device="cuda")
    #     self.set_level(points, *args)
    #     self.spatial_lr_scale = spatial_lr_scale
    #     # fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
    #     # fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
    #     box_min = torch.min(points)*self.extend
    #     box_max = torch.max(points)*self.extend
    #     box_d = box_max - box_min
    #     # print(box_d)
    #     if self.base_layer < 0:
    #         default_voxel_size = 0.02
    #         self.base_layer = torch.round(torch.log2(box_d/default_voxel_size)).int().item()-(self.levels//2)+1
    #     self.voxel_size = box_d/(float(self.fork) ** self.base_layer)
    #     self.init_pos = torch.tensor([box_min, box_min, box_min]).float().cuda()
    #     self.octree_sample(points)

    #     if self.visible_threshold < 0:
    #         self.visible_threshold = 0.0
    #         self.positions, self._level, self.visible_threshold, _ = self.weed_out(self.positions, self._level)
    #     self.positions, self._level, _, _ = self.weed_out(self.positions, self._level)

    #     fused_point_cloud = self.positions
    #     offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3)).float().cuda()
    #     anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()
    #     anchors_feat_semantic = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()

    #     # features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda() # [N, 3, 16]
    #     # features[:, :3, 0 ] = fused_color
    #     # features[:, 3:, 1:] = 0.0

    #     print("Number of points at initialisation : ", fused_point_cloud.shape[0])

    #     dist2 = (knn(fused_point_cloud, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    #     scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 6)
    #     rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
    #     rots[:, 0] = 1
    #     # dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
    #     # scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
    #     # rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
    #     # rots[:, 0] = 1
    #     opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

    #     # modify -----
    #     self._anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))
    #     self._offset = nn.Parameter(offsets.requires_grad_(True)) 
    #     self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
    #     self._anchor_feat_semantic = nn.Parameter(anchors_feat_semantic.requires_grad_(True))
    #     # ins_feat = torch.rand((fused_point_cloud.shape[0], 6), dtype=torch.float, device="cuda")

    #     # self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
    #     # self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
    #     # self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
    #     self._scaling = nn.Parameter(scales.requires_grad_(True))
    #     self._rotation = nn.Parameter(rots.requires_grad_(True))
    #     self._opacity = nn.Parameter(opacities.requires_grad_(True))
    #     # modify -----
    #     # self._ins_feat = nn.Parameter(ins_feat.requires_grad_(True))
    #     # self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
    #     self._level = self._level.unsqueeze(dim=1)
    #     self._extra_level = torch.zeros(self._anchor.shape[0], dtype=torch.float, device="cuda")
    #     self._anchor_mask = torch.ones(self._anchor.shape[0], dtype=torch.bool, device="cuda")

    def voxelize_sample(self, data=None, voxel_size=0.01):
        np.random.shuffle(data)
        data = np.unique(np.round(data/voxel_size), axis=0)*voxel_size
        
        return data
    
    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, *args):
        self.spatial_lr_scale = spatial_lr_scale
        self.ratio=1
        points = pcd.points[::self.ratio]

        if self.voxel_size <= 0:
            init_points = torch.tensor(points).float().cuda()
            init_dist = distCUDA2(init_points).float().cuda()
            median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0]*0.5))
            self.voxel_size = median_dist.item()
            del init_dist
            del init_points
            torch.cuda.empty_cache()

        print(f'Initial voxel_size: {self.voxel_size}')
        
        
        points = self.voxelize_sample(points, voxel_size=self.voxel_size)
        fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
        offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3)).float().cuda()
        anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()
        anchors_feat_semantic = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()
        
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud).float().cuda(), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 6)
        
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        self._anchor_feat_semantic = nn.Parameter(anchors_feat_semantic.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")

    def weed_out(self, gaussian_positions, gaussian_levels):
        visible_count = torch.zeros(gaussian_positions.shape[0], dtype=torch.int, device="cuda")
        for cam in self.cam_infos:
            cam_center, scale = cam[:3], cam[3]
            dist = torch.sqrt(torch.sum((gaussian_positions - cam_center)**2, dim=1)) * scale
            pred_level = torch.log2(self.standard_dist/dist)/math.log2(self.fork)   
            int_level = self.map_to_int_level(pred_level, self.levels - 1)
            visible_count += (gaussian_levels <= int_level).int()
        visible_count = visible_count/len(self.cam_infos)
        weed_mask = (visible_count > self.visible_threshold)
        mean_visible = torch.mean(visible_count)
        return gaussian_positions[weed_mask], gaussian_levels[weed_mask], mean_visible, weed_mask

    def set_anchor_mask(self, cam_center, iteration, resolution_scale):
        dist = torch.sqrt(torch.sum((self._anchor - cam_center)**2, dim=1)) * resolution_scale
        pred_level = torch.log2(self.standard_dist/dist)/math.log2(self.fork) + self._extra_level
        if self.progressive:
            coarse_index = np.searchsorted(self.coarse_intervals, iteration) + 1 + self.init_level
        else:
            coarse_index = self.levels

        int_level = self.map_to_int_level(pred_level, coarse_index - 1)
        self._anchor_mask = (self._level.squeeze(dim=1) <= int_level)    

    def set_anchor_mask_perlevel(self, cam_center, resolution_scale, cur_level):
        dist = torch.sqrt(torch.sum((self.get_anchor - cam_center)**2, dim=1)) * resolution_scale
        pred_level = torch.log2(self.standard_dist/dist)/math.log2(self.fork) + self._extra_level
        int_level = self.map_to_int_level(pred_level, cur_level)
        self._anchor_mask = (self._level.squeeze(dim=1) <= int_level)

    def training_setup(self, training_args):
        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")
        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")
        self.percent_dense = training_args.percent_dense
        # self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        l = [
            {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
            {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
            {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
            {'params': [self._anchor_feat_semantic], 'lr': training_args.feature_lr, "name": "anchor_feat_semantic"},
            # {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            # {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            # {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
            {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
            {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
            {'params': self.mlp_semantic.parameters(), 'lr': training_args.mlp_semantic_lr_init, "name": "mlp_semantic"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            # {'params': [self._ins_feat], 'lr': training_args.ins_feat_lr, "name": "ins_feat"}  # modify -----
        ]
        if self.appearance_dim > 0:
            l.append({'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"})
        if self.use_feat_bank:
            l.append({'params': self.mlp_feature_bank.parameters(), 'lr': training_args.mlp_featurebank_lr_init, "name": "mlp_featurebank"})
        # note: Freeze the position of the initial point, do not densify. for ScanNet 3DGS pre-train stage
        if training_args.frozen_init_pts:
            self._xyz = self._xyz.detach()

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        # self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
        #                                             lr_final=training_args.position_lr_final*self.spatial_lr_scale,
        #                                             lr_delay_mult=training_args.position_lr_delay_mult,
        #                                             max_steps=training_args.position_lr_max_steps)
        self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.offset_lr_delay_mult,
                                                    max_steps=training_args.offset_lr_max_steps)
        
        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                    lr_final=training_args.mlp_opacity_lr_final,
                                                    lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                    max_steps=training_args.mlp_opacity_lr_max_steps)
        
        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                    lr_final=training_args.mlp_cov_lr_final,
                                                    lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                    max_steps=training_args.mlp_cov_lr_max_steps)
        
        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                    lr_final=training_args.mlp_color_lr_final,
                                                    lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                    max_steps=training_args.mlp_color_lr_max_steps)
        
        self.mlp_semantic_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_semantic_lr_init,
                                                    lr_final=training_args.mlp_semantic_lr_final,
                                                    lr_delay_mult=training_args.mlp_semantic_lr_delay_mult,
                                                    max_steps=training_args.mlp_semantic_lr_max_steps)
        
        if self.use_feat_bank:
            self.mlp_featurebank_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_featurebank_lr_init,
                                                        lr_final=training_args.mlp_featurebank_lr_final,
                                                        lr_delay_mult=training_args.mlp_featurebank_lr_delay_mult,
                                                        max_steps=training_args.mlp_featurebank_lr_max_steps)
        if self.appearance_dim > 0:
            self.appearance_scheduler_args = get_expon_lr_func(lr_init=training_args.appearance_lr_init,
                                                        lr_final=training_args.appearance_lr_final,
                                                        lr_delay_mult=training_args.appearance_lr_delay_mult,
                                                        max_steps=training_args.appearance_lr_max_steps)

    def update_learning_rate(self, iteration, root_start, leaf_start):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            # if param_group["name"] == "xyz":
            #     lr = self.xyz_scheduler_args(iteration)
            #     param_group['lr'] = lr
            #     # return lr
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr
            # if param_group["name"] == "ins_feat":
            #     if iteration > root_start and iteration <= leaf_start:      # TODO: update lr
            #         param_group['lr'] = param_group['lr'] * 0 + 0.0001
            #     else:
            #         param_group['lr'] = param_group['lr'] * 0 + 0.001
            if param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_color":
                lr = self.mlp_color_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group['name'] =='mlp_semantic':
                lr = self.mlp_semantic_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_feat_bank and param_group["name"] == "mlp_featurebank":
                lr = self.mlp_featurebank_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.appearance_dim > 0 and param_group["name"] == "embedding_appearance":
                lr = self.appearance_scheduler_args(iteration)
                param_group['lr'] = lr
            # if iteration % 1000 == 0:
            #     self.oneupSHdegree()

    # def construct_list_of_attributes(self):
    #     l = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'ins_feat_r', 'ins_feat_g', 'ins_feat_b', \
    #         'ins_feat_r2', 'ins_feat_g2', 'ins_feat_b2']
    #     l.append('level')
    #     l.append('extra_level')
    #     for i in range(self._offset.shape[1]*self._offset.shape[2]):
    #         l.append('f_offset_{}'.format(i))
    #     # All channels except the 3 DC
    #     for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
    #         l.append('f_dc_{}'.format(i))
    #     for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
    #         l.append('f_rest_{}'.format(i))
    #     l.append('opacity')
    #     for i in range(self._scaling.shape[1]):
    #         l.append('scale_{}'.format(i))
    #     for i in range(self._rotation.shape[1]):
    #         l.append('rot_{}'.format(i))
    #     return l
    def construct_list_of_attributes(self):
        l = []
        l.append('x')
        l.append('y')
        l.append('z')
        # l.append('level')
        # l.append('extra_level')
        for i in range(self._offset.shape[1]*self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
        for i in range(self._anchor_feat.shape[1]):
            l.append('f_anchor_feat_{}'.format(i))
        for i in range(self._anchor_feat_semantic.shape[1]):
            l.append('f_semantic_feat_{}'.format(i))
        # for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
        #     l.append('f_dc_{}'.format(i))
        # for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
        #     l.append('f_rest_{}'.format(i))
        # l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        # for i in range(self._ins_feat_q.shape[1]):
        #     l.append('ins_feat_{}'.format(i))
        # l+=['ins_feat_r', 'ins_feat_g', 'ins_feat_b', \
        #     'ins_feat_r2', 'ins_feat_g2', 'ins_feat_b2']
        return l
    

    def save_ply(self, path, iteration):
        mkdir_p(os.path.dirname(path))

        if self.progressive:
            coarse_index = np.searchsorted(self.coarse_intervals, iteration) + 1 + self.init_level
        else:
            coarse_index = self.levels

        # level_mask = (self._level <= coarse_index-1).squeeze(-1)
        level_mask = torch.ones(self._anchor.shape[0]).bool().cuda()
        anchor = self._anchor[level_mask].detach().cpu().numpy()
        # levels = self._level[level_mask].detach().cpu().numpy()
        # extra_levels = self._extra_level.unsqueeze(dim=1)[level_mask].detach().cpu().numpy()
        anchor_feats = self._anchor_feat[level_mask].detach().cpu().numpy()
        anchor_feats_semantic = self._anchor_feat_semantic[level_mask].detach().cpu().numpy()
        offsets = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous()[level_mask].cpu().numpy()
        # f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous()[level_mask].cpu().numpy()
        # f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous()[level_mask].cpu().numpy()
        # opacities = self._opacity[level_mask].detach().cpu().numpy()
        scale = self._scaling[level_mask].detach().cpu().numpy()
        rotation = self._rotation[level_mask].detach().cpu().numpy()
        # ins_feat = self._ins_feat_q[level_mask].detach().cpu().numpy()
        # ins_feat = self._ins_feat[level_mask].detach().cpu().numpy()
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]


        # vis_color = (self._ins_feat + 1) / 2 * 255
        # r, g, b = vis_color[:, 0].reshape(-1, 1), vis_color[:, 1].reshape(-1, 1), vis_color[:, 2].reshape(-1, 1)
        # ignored_ind = sigmoid(opacities) < 0.1
        # r[ignored_ind], g[ignored_ind], b[ignored_ind] = 128, 128, 128
        # r = r[level_mask].detach().cpu().numpy()
        # g = g[level_mask].detach().cpu().numpy()
        # b = b[level_mask].detach().cpu().numpy()
        # dtype_full = dtype_full + [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]  # modify

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate((anchor, offsets, anchor_feats, anchor_feats_semantic, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')

        plydata = PlyData([el])
        # , obj_info=[
        #     'standard_dist {:.6f}'.format(self.standard_dist),
        #     'levels {:.6f}'.format(self.levels),
        #     ])
        plydata.write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)
        infos = plydata.obj_info
        for info in infos:
            var_name = info.split(' ')[0]
            self.__dict__[var_name] = float(info.split(' ')[1])
        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        # levels = np.asarray(plydata.elements[0]["level"])[... ,np.newaxis].astype(np.int16)
        # extra_levels = np.asarray(plydata.elements[0]["extra_level"])[... ,np.newaxis].astype(np.float32)
        # offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        # offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        # offsets = np.zeros((anchor.shape[0], len(offset_names)))
        # for idx, attr_name in enumerate(offset_names):
        #     offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        # offsets = offsets.reshape((offsets.shape[0], 3, -1))
        # ins_feat = np.stack((np.asarray(plydata.elements[0]["ins_feat_r"]),
        #                 np.asarray(plydata.elements[0]["ins_feat_g"]),
        #                 np.asarray(plydata.elements[0]["ins_feat_b"]),
        #                 np.asarray(plydata.elements[0]["ins_feat_r2"]),
        #                 np.asarray(plydata.elements[0]["ins_feat_g2"]),
        #                 np.asarray(plydata.elements[0]["ins_feat_b2"])),  axis=1)
        # opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        # if not opacities.flags['C_CONTIGUOUS']:
        #     opacities = np.ascontiguousarray(opacities)

        # features_dc = np.zeros((anchor.shape[0], 3, 1))
        # features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        # features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        # features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        # extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        # extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        # assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        # features_extra = np.zeros((anchor.shape[0], len(extra_f_names)))
        # for idx, attr_name in enumerate(extra_f_names):
        #     features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        # features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        
        anchor_feat_semantic_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_semantic_feat")]
        anchor_feat_semantic_names = sorted(anchor_feat_semantic_names, key = lambda x: int(x.split('_')[-1]))
        anchor_feats_semantic = np.zeros((anchor.shape[0], len(anchor_feat_semantic_names)))
        for idx, attr_name in enumerate(anchor_feat_semantic_names):
            anchor_feats_semantic[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        # ins_feat_q_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("ins_feat_")]
        # ins_feat_q_names = sorted(ins_feat_q_names, key = lambda x: int(x.split('_')[-1]))
        # ins_feat_q = np.zeros((anchor.shape[0], len(ins_feat_q_names)))
        # for idx, attr_name in enumerate(ins_feat_q_names):
        #     ins_feat_q[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))
        

        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))
        self._anchor_feat_semantic = nn.Parameter(torch.tensor(anchor_feats_semantic, dtype=torch.float, device="cuda").requires_grad_(True))
        # self._level = torch.tensor(levels, dtype=torch.int, device="cuda")
        # self._extra_level = torch.tensor(extra_levels, dtype=torch.float, device="cuda").squeeze(dim=1)
        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        # self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        # self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        # self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._anchor_mask = torch.ones(self._anchor.shape[0], dtype=torch.bool, device="cuda")
        # self._ins_feat_q = nn.Parameter(torch.tenssor(ins_feat_q, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
        # self.levels = round(self.levels)
        # if self.init_level == -1:
        #     self.init_level = int(self.levels/2)

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'embedding' in group['name']:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            
        return optimizable_tensors
        # optimizable_tensors = {}
        # for group in self.optimizer.param_groups:
        #     stored_state = self.optimizer.state.get(group['params'][0], None)
        #     if stored_state is not None:
        #         stored_state["exp_avg"] = stored_state["exp_avg"][mask]
        #         stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

        #         del self.optimizer.state[group['params'][0]]
        #         group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
        #         self.optimizer.state[group['params'][0]] = stored_state

        #         optimizable_tensors[group["name"]] = group["params"][0]
        #     else:
        #         group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
        #         optimizable_tensors[group["name"]] = group["params"][0]
        # return optimizable_tensors
    # def _prune_anchor_optimizer(self, mask):
    #     optimizable_tensors = {}
    #     for group in self.optimizer.param_groups:
    #         if  'mlp' in group['name'] or \
    #             'conv' in group['name'] or \
    #             'embedding' in group['name']:
    #             continue

    #         stored_state = self.optimizer.state.get(group['params'][0], None)
    #         if stored_state is not None:
    #             stored_state["exp_avg"] = stored_state["exp_avg"][mask]
    #             stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

    #             del self.optimizer.state[group['params'][0]]
    #             group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
    #             self.optimizer.state[group['params'][0]] = stored_state
    #             if group['name'] == "scaling":
    #                 scales = group["params"][0]
    #                 temp = scales[:,3:]
    #                 temp[temp>0.05] = 0.05
    #                 group["params"][0][:,3:] = temp
    #             optimizable_tensors[group["name"]] = group["params"][0]
    #         else:
    #             group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
    #             if group['name'] == "scaling":
    #                 scales = group["params"][0]
    #                 temp = scales[:,3:]
    #                 temp[temp>0.05] = 0.05
    #                 group["params"][0][:,3:] = temp
    #             optimizable_tensors[group["name"]] = group["params"][0]

    # def prune_points(self, mask):
    #     valid_points_mask = ~mask
    #     optimizable_tensors = self._prune_optimizer(valid_points_mask)

    #     # self._xyz = optimizable_tensors["xyz"]
    #     self._anchor = optimizable_tensors["anchor"]
    #     self._offset = optimizable_tensors["offset"]
    #     self._features_dc = optimizable_tensors["f_dc"]
    #     self._features_rest = optimizable_tensors["f_rest"]
    #     self._opacity = optimizable_tensors["opacity"]
    #     self._scaling = optimizable_tensors["scaling"]
    #     self._rotation = optimizable_tensors["rotation"]
    #     self._ins_feat = optimizable_tensors["ins_feat"]
    #     self._level = self._level[valid_points_mask]
    #     self._extra_level = self._extra_level[valid_points_mask]

    #     # self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

    #     # self.denom = self.denom[valid_points_mask]
    #     self.max_radii2D = self.max_radii2D[valid_points_mask]
    def prune_anchor(self,mask):
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)

        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._anchor_feat_semantic = optimizable_tensors["anchor_feat_semantic"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

    def _prune_anchor_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'embedding' in group['name']:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            
            
        return optimizable_tensors

    def anchor_growing(self, grads, threshold, offset_mask):
        ## 
        init_length = self.get_anchor.shape[0]*self.n_offsets
        for i in range(self.update_depth):
            # update threshold
            cur_threshold = threshold*((self.update_hierachy_factor//2)**i)
            # mask from grad threshold
            candidate_mask = (grads >= cur_threshold)
            candidate_mask = torch.logical_and(candidate_mask, offset_mask)
            
            # random pick
            rand_mask = torch.rand_like(candidate_mask.float())>(0.5**(i+1))
            rand_mask = rand_mask.cuda()
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)
            
            length_inc = self.get_anchor.shape[0]*self.n_offsets - init_length
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)

            all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:,:3].unsqueeze(dim=1)
            
            # assert self.update_init_factor // (self.update_hierachy_factor**i) > 0
            # size_factor = min(self.update_init_factor // (self.update_hierachy_factor**i), 1)
            size_factor = self.update_init_factor // (self.update_hierachy_factor**i)
            cur_size = self.voxel_size*size_factor
            
            grid_coords = torch.round(self.get_anchor / cur_size).int()

            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round(selected_xyz / cur_size).int()

            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)


            ## split data for reducing peak memory calling
            use_chunk = True
            if use_chunk:
                chunk_size = 4096
                max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
                remove_duplicates_list = []
                for i in range(max_iters):
                    cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i*chunk_size:(i+1)*chunk_size, :]).all(-1).any(-1).view(-1)
                    remove_duplicates_list.append(cur_remove_duplicates)
                
                remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
            else:
                remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)

            remove_duplicates = ~remove_duplicates
            candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size

            
            if candidate_anchor.shape[0] > 0:
                new_scaling = torch.ones_like(candidate_anchor).repeat([1,2]).float().cuda()*cur_size # *0.05
                new_scaling = torch.log(new_scaling)
                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device=candidate_anchor.device).float()
                new_rotation[:,0] = 1.0

                new_opacities = inverse_sigmoid(0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda"))

                new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_mask]

                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]

                new_feat_semantic = self._anchor_feat_semantic.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_mask]

                new_feat_semantic = scatter_max(new_feat_semantic, inverse_indices.unsqueeze(1).expand(-1, new_feat_semantic.size(1)), dim=0)[0][remove_duplicates]

                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1,self.n_offsets,1]).float().cuda()

                d = {
                    "anchor": candidate_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "anchor_feat_semantic": new_feat_semantic,
                    "offset": new_offsets,
                    "opacity": new_opacities,
                }
                

                temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                torch.cuda.empty_cache()
                
                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._anchor_feat_semantic = optimizable_tensors["anchor_feat_semantic"]
                self._offset = optimizable_tensors["offset"]
                self._opacity = optimizable_tensors["opacity"]
    
    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'embedding' in group['name']:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
    
    def get_remove_duplicates(self, grid_coords, selected_grid_coords_unique, use_chunk = True):
        if use_chunk:
            chunk_size = 4096
            max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
            remove_duplicates_list = []
            for i in range(max_iters):
                cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i*chunk_size:(i+1)*chunk_size, :]).all(-1).any(-1).view(-1)
                remove_duplicates_list.append(cur_remove_duplicates)
            remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
        else:
            remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)
        return remove_duplicates
    
    def adjust_anchor(self, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, min_opacity=0.005):
        # # adding anchors
        grads = self.offset_gradient_accum / self.offset_denom # [N*k, 1]
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > check_interval*success_threshold*0.5).squeeze(dim=1)
        
        self.anchor_growing(grads_norm, grad_threshold, offset_mask)
        
        # update offset_denom
        self.offset_denom[offset_mask] = 0
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)
        
        # # prune anchors
        prune_mask = (self.opacity_accum < min_opacity*self.anchor_demon).squeeze(dim=1)
        anchors_mask = (self.anchor_demon > check_interval*success_threshold).squeeze(dim=1) # [N, 1]
        prune_mask = torch.logical_and(prune_mask, anchors_mask) # [N] 
        
        # update offset_denom
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum
        
        # update opacity accum 
        if anchors_mask.sum()>0:
            self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
        
        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon

        if prune_mask.shape[0]>0:
            self.prune_anchor(prune_mask)
        
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")

    def save_mlp_checkpoints(self, path):#split or unite
        mkdir_p(os.path.dirname(path))
        self.eval()
        opacity_mlp = torch.jit.trace(self.mlp_opacity, (torch.rand(1, self.feat_dim+self.view_dim).cuda()))
        opacity_mlp.save(os.path.join(path, 'opacity_mlp.pt'))
        cov_mlp = torch.jit.trace(self.mlp_cov, (torch.rand(1, self.feat_dim+self.view_dim).cuda()))
        cov_mlp.save(os.path.join(path, 'cov_mlp.pt'))
        color_mlp = torch.jit.trace(self.mlp_color, (torch.rand(1, self.feat_dim+self.view_dim+self.appearance_dim).cuda()))
        color_mlp.save(os.path.join(path, 'color_mlp.pt'))
        semantic_mlp = torch.jit.trace(self.mlp_semantic, (torch.rand(1, self.feat_dim).cuda()))
        semantic_mlp.save(os.path.join(path, 'semantic_mlp.pt'))
        if self.use_feat_bank:
            feature_bank_mlp = torch.jit.trace(self.mlp_feature_bank, (torch.rand(1, self.view_dim).cuda()))
            feature_bank_mlp.save(os.path.join(path, 'feature_bank_mlp.pt'))
        if self.appearance_dim > 0:
            emd = torch.jit.trace(self.embedding_appearance, (torch.zeros((1,), dtype=torch.long).cuda()))
            emd.save(os.path.join(path, 'embedding_appearance.pt'))
        self.train()

    def load_mlp_checkpoints(self, path):
        self.mlp_opacity = torch.jit.load(os.path.join(path, 'opacity_mlp.pt')).cuda()
        self.mlp_cov = torch.jit.load(os.path.join(path, 'cov_mlp.pt')).cuda()
        self.mlp_color = torch.jit.load(os.path.join(path, 'color_mlp.pt')).cuda()
        self.mlp_semantic = torch.jit.load(os.path.join(path, 'semantic_mlp.pt')).cuda()
        if self.use_feat_bank:
            self.mlp_feature_bank = torch.jit.load(os.path.join(path, 'feature_bank_mlp.pt')).cuda()
        if self.appearance_dim > 0:
            self.embedding_appearance = torch.jit.load(os.path.join(path, 'embedding_appearance.pt')).cuda()


    # def generate_neural_gaussians(self, viewpoint_camera, visible_mask=None, ape_code=-1, ins_feat_q=False):
    #     # view frustum filtering for acceleration    
    #     if visible_mask is None:
    #         visible_mask = torch.ones(self.get_anchor.shape[0], dtype=torch.bool, device = self.get_anchor.device)
    #     anchor = self.get_anchor[visible_mask]
    #     # anchor_full = self.get_anchor
    #     feat = self.get_anchor_feat[visible_mask]
    #     if ins_feat_q == True:
    #         feat_semantic = self._ins_feat_q[visible_mask]
    #     else:
    #         feat_semantic = self.get_anchor_feat_semantic[visible_mask]
    #     # feat_semantic = self.get_anchor_feat_semantic
    #     grid_offsets = self.get_offset[visible_mask]
    #     grid_scaling = self.get_scaling[visible_mask]
    #     # get view properties for anchor
    #     ob_view = anchor - viewpoint_camera.camera_center
    #     # dist
    #     ob_dist = ob_view.norm(dim=1, keepdim=True)
    #     # view
    #     ob_view = ob_view / ob_dist

    #     ## view-adaptive feature
    #     if self.use_feat_bank:
    #         bank_weight = self.get_featurebank_mlp(ob_view).unsqueeze(dim=1) # [n, 1, 3]

    #         ## multi-resolution feat
    #         feat = feat.unsqueeze(dim=-1)
    #         feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
    #             feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
    #             feat[:,::1, :1]*bank_weight[:,:,2:]
    #         feat = feat.squeeze(dim=-1) # [n, c]

    #     cat_local_view = torch.cat([feat, ob_view], dim=1) # [N, c+3]
    #     cat_local_view_semantic = torch.cat([feat_semantic, ob_view], dim=1)

    #     if self.appearance_dim > 0:
    #         if ape_code < 0:
    #             camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * viewpoint_camera.uid
    #             appearance = self.get_appearance(camera_indicies)
    #         else:
    #             camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * ape_code[0]
    #             appearance = self.get_appearance(camera_indicies)
                
    #     # get offset's opacity
    #     neural_opacity = self.get_opacity_mlp(cat_local_view) # [N, k]
    #     if self.dist2level=="progressive":
    #         prog = self._prog_ratio[visible_mask]
    #         transition_mask = self.transition_mask[visible_mask]
    #         prog[~transition_mask] = 1.0
    #         neural_opacity = neural_opacity * prog
        
    #     # opacity mask generation
    #     neural_opacity = neural_opacity.reshape([-1, 1])
    #     mask = (neural_opacity>0.0)
    #     mask = mask.view(-1)

    #     # select opacity 
    #     # opacity = neural_opacity.clone()
    #     # opacity[~mask] = 0.0
    #     opacity = neural_opacity[mask]

    #     # get offset's color
    #     if self.appearance_dim > 0:
    #         color = self.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
    #     else:
    #         color = self.get_color_mlp(cat_local_view)
    #     color = color.reshape([anchor.shape[0]*self.n_offsets, 3])# [mask]

    #     # get offset's cov
    #     scale_rot = self.get_cov_mlp(cat_local_view)
    #     scale_rot = scale_rot.reshape([anchor.shape[0]*self.n_offsets, 7]) # [mask]
        
    #     # get offset's semantic
    #     semantic = self.get_semantic_mlp(feat_semantic)#(cat_local_view_semantic)
    #     semantic = semantic.reshape([anchor.shape[0]*self.n_offsets, self.semantic_dim])
    #     # offsets
    #     offsets = grid_offsets.view([-1, 3]) # [mask]
        
    #     # combine for parallel masking
    #     concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    #     concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=self.n_offsets)
    #     concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, semantic, offsets], dim=-1)
    #     masked = concatenated_all[mask]
    #     scaling_repeat, repeat_anchor, color, scale_rot, semantic, offsets = masked.split([6, 3, 3, 7, self.semantic_dim, 3], dim=-1)
        
    #     # post-process cov
    #     scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3]) # * (1+torch.sigmoid(repeat_dist))
    #     rot = self.rotation_activation(scale_rot[:,3:7])
        
    #     # post-process offsets to get centers for gaussians
    #     offsets = offsets * scaling_repeat[:,:3]
    #     xyz = repeat_anchor + offsets 
    #     return xyz, color, opacity, scaling, rot, semantic, None, mask, neural_opacity

    def generate_neural_gaussians(self, viewpoint_camera, visible_mask=None, ape_code=-1, ins_feat_q=False):
        ## view frustum filtering for acceleration    
        if visible_mask is None:
            visible_mask = torch.ones(self.get_anchor.shape[0], dtype=torch.bool, device = self.get_anchor.device)
        
        feat = self._anchor_feat[visible_mask]
        # feat_semantic = self.get_anchor_feat_semantic[visible_mask]
        anchor = self.get_anchor[visible_mask]
        grid_offsets = self._offset[visible_mask]
        grid_scaling = self.get_scaling[visible_mask]


        ## get view properties for anchor
        ob_view = anchor - viewpoint_camera.camera_center
        # dist
        ob_dist = ob_view.norm(dim=1, keepdim=True)
        # view
        ob_view = ob_view / ob_dist

        ## view-adaptive feature
        if self.use_feat_bank:
            cat_view = torch.cat([ob_view, ob_dist], dim=1)
            
            bank_weight = self.get_featurebank_mlp(cat_view).unsqueeze(dim=1) # [n, 1, 3]

            ## multi-resolution feat
            feat = feat.unsqueeze(dim=-1)
            feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
                feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
                feat[:,::1, :1]*bank_weight[:,:,2:]
            feat = feat.squeeze(dim=-1) # [n, c]


        cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1) # [N, c+3+1]
        cat_local_view_wodist = torch.cat([feat, ob_view], dim=1) # [N, c+3]
        if self.appearance_dim > 0:
            camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * viewpoint_camera.uid
            # camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * 10
            appearance = self.get_appearance(camera_indicies)

        # # get offset's opacity
        # if self.add_opacity_dist:
        #     neural_opacity = self.get_opacity_mlp(cat_local_view) # [N, k]
        # else:
        neural_opacity = self.get_opacity_mlp(cat_local_view_wodist)

        # opacity mask generation
        neural_opacity = neural_opacity.reshape([-1, 1])
        mask = (neural_opacity>0.0)
        mask = mask.view(-1)

        # select opacity 
        opacity = neural_opacity[mask]

        # get offset's color
        if self.appearance_dim > 0:
            if self.add_color_dist:
                color = self.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
            else:
                color = self.get_color_mlp(torch.cat([cat_local_view_wodist, appearance], dim=1))
        else:
            # if self.add_color_dist:
            #     color = self.get_color_mlp(cat_local_view)
            # else:
            color = self.get_color_mlp(cat_local_view_wodist)
        color = color.reshape([anchor.shape[0]*self.n_offsets, 3])# [mask]

        # get offset's cov
        # if self.add_cov_dist:
        #     scale_rot = self.get_cov_mlp(cat_local_view)
        # else:
        scale_rot = self.get_cov_mlp(cat_local_view_wodist)
        scale_rot = scale_rot.reshape([anchor.shape[0]*self.n_offsets, 7]) # [mask]
        
        # offsets
        offsets = grid_offsets.view([-1, 3]) # [mask]
        
        # semantic
        if ins_feat_q == True:
            semantic = self._ins_feat_q[torch.repeat_interleave(visible_mask, 10)]
        else:
            feat_semantic = self.get_anchor_feat_semantic[visible_mask]
            semantic = self.get_semantic_mlp(feat_semantic)
            semantic = semantic.reshape([anchor.shape[0]*self.n_offsets, self.semantic_dim])
        semantic = torch.nn.functional.normalize(semantic, dim=1)
        # combine for parallel masking
        concatenated = torch.cat([grid_scaling, anchor], dim=-1)
        concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=self.n_offsets)
        concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets, semantic], dim=-1)
        masked = concatenated_all[mask]
        scaling_repeat, repeat_anchor, color, scale_rot, offsets, semantic = masked.split([6, 3, 3, 7, 3, self.semantic_dim], dim=-1)
        
        # post-process cov
        scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3]) # * (1+torch.sigmoid(repeat_dist))
        rot = self.rotation_activation(scale_rot[:,3:7])
        
        # post-process offsets to get centers for gaussians
        offsets = offsets * scaling_repeat[:,:3]
        xyz = repeat_anchor + offsets

        # # neural_xyz and neural_semantic for kmeans
        # neural_masked = concatenated_all
        # neural_scaling_repeat, neural_repeat_anchor, neural_color, neural_scale_rot, neural_offsets, neural_semantic = neural_masked.split([6, 3, 3, 7, 3, self.semantic_dim], dim=-1)
        # neural_offsets = neural_offsets * neural_scaling_repeat[:,:3]
        # neural_xyz = neural_repeat_anchor + neural_offsets

        return xyz, color, opacity, scaling, rot, semantic, None, mask, neural_opacity#, neural_xyz, neural_semantic
    
    def training_statis(self, viewspace_point_tensor, opacity, update_filter, offset_selection_mask, anchor_visible_mask):
        # update opacity stats
        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity<0] = 0
        
        temp_opacity = temp_opacity.view([-1, self.n_offsets])
        self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)
        
        # update anchor visiting statis
        self.anchor_demon[anchor_visible_mask] += 1

        # update neural gaussian statis
        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_visible_mask] = offset_selection_mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter
        
        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.offset_gradient_accum[combined_mask] += grad_norm
        self.offset_denom[combined_mask] += 1

    # def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, \
    #                             new_scaling, new_rotation, new_ins_feat):
    #     d = {"xyz": new_xyz,
    #     "f_dc": new_features_dc,
    #     "f_rest": new_features_rest,
    #     "opacity": new_opacities,
    #     "scaling" : new_scaling,
    #     "rotation" : new_rotation,
    #     "ins_feat": new_ins_feat}

    #     optimizable_tensors = self.cat_tensors_to_optimizer(d)
    #     self._xyz = optimizable_tensors["xyz"]
    #     self._features_dc = optimizable_tensors["f_dc"]
    #     self._features_rest = optimizable_tensors["f_rest"]
    #     self._opacity = optimizable_tensors["opacity"]
    #     self._scaling = optimizable_tensors["scaling"]
    #     self._rotation = optimizable_tensors["rotation"]
    #     self._ins_feat = optimizable_tensors["ins_feat"]

    #     self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
    #     self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
    #     self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    # def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
    #     n_init_points = self.get_xyz.shape[0]
    #     # Extract points that satisfy the gradient condition
    #     padded_grad = torch.zeros((n_init_points), device="cuda")
    #     padded_grad[:grads.shape[0]] = grads.squeeze()
    #     selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
    #     selected_pts_mask = torch.logical_and(selected_pts_mask,
    #                                           torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

    #     stds = self.get_scaling[selected_pts_mask].repeat(N,1)
    #     means =torch.zeros((stds.size(0), 3),device="cuda")
    #     samples = torch.normal(mean=means, std=stds)
    #     rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
    #     new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
    #     new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
    #     new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
    #     new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
    #     new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
    #     new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
    #     new_ins_feat = self._ins_feat[selected_pts_mask].repeat(N,1)

    #     self.densification_postfix(new_xyz, new_features_dc, new_features_rest, \
    #         new_opacity, new_scaling, new_rotation, new_ins_feat)

    #     prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
    #     self.prune_points(prune_filter)

    # def densify_and_clone(self, grads, grad_threshold, scene_extent):
    #     # Extract points that satisfy the gradient condition
    #     selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
    #     selected_pts_mask = torch.logical_and(selected_pts_mask,
    #                                           torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
    #     new_xyz = self._xyz[selected_pts_mask]
    #     new_features_dc = self._features_dc[selected_pts_mask]
    #     new_features_rest = self._features_rest[selected_pts_mask]
    #     new_opacities = self._opacity[selected_pts_mask]
    #     new_scaling = self._scaling[selected_pts_mask]
    #     new_rotation = self._rotation[selected_pts_mask]
    #     new_ins_feat = self._ins_feat[selected_pts_mask]

    #     self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, \
    #         new_scaling, new_rotation, new_ins_feat)

    # def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
    #     grads = self.xyz_gradient_accum / self.denom
    #     grads[grads.isnan()] = 0.0

    #     self.densify_and_clone(grads, max_grad, extent)
    #     self.densify_and_split(grads, max_grad, extent)

    #     prune_mask = (self.get_opacity < min_opacity).squeeze()
    #     if max_screen_size:
    #         big_points_vs = self.max_radii2D > max_screen_size
    #         big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
    #         prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
    #     self.prune_points(prune_mask)

    #     torch.cuda.empty_cache()

    # def add_densification_stats(self, viewspace_point_tensor, update_filter):
    #     self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
    #     self.denom[update_filter] += 1

from torch.autograd import Function

# 自定义Function类用于处理掩码和梯度相关操作
class MaskedGradientFunction(Function):
    @staticmethod
    def forward(ctx, input, mask):
        ctx.mask = mask
        ctx.selected_indices = torch.where(mask)[0]  # 记录下掩码为True对应的索引
        ctx.input_shape = input.shape  # 保存input的形状
        return input[mask]  # 返回掩码筛选后的数据用于后续正常前向传播

    @staticmethod
    def backward(ctx, grad_output):
        grad_full = torch.zeros_like(grad_output.new_empty((ctx.input_shape[0], *grad_output.size()[1:]))).to(grad_output.device)
        grad_full[ctx.selected_indices] = grad_output  # 根据记录的索引把梯度填充到对应位置
        return grad_full, None  # 第二个返回值None对应forward中除input外的其他参数（这里只有mask，不需要梯度回传）