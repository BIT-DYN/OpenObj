import random
import numpy as np
import torch
from tqdm import tqdm
import trainer
import open3d
import trimesh
import scipy
from bidict import bidict
import copy
import os
import render_rays
import matplotlib.pyplot as plt
import open3d as o3d 

import utils
from collections import Counter




class sceneObject:
    """
    这个是场景物体的类，用于存储物体的关键帧数据等
    object instance mapping,
    updating keyframes, get training samples, optimizing MLP map
    """

    def __init__(self, cfg, obj_id, rgb:torch.tensor, depth:torch.tensor, mask:torch.tensor, bbox_2d:torch.tensor, 
                 t_wc:torch.tensor, live_frame_id, clip_feat = None, caption_feat = None) -> None:
        self.do_bg = cfg.do_bg
        self.obj_id = obj_id
        self.data_device = cfg.data_device
        self.training_device = cfg.training_device
        self.part_mode = cfg.part_mode
        self.stride = cfg.stride

        assert rgb.shape[:2] == depth.shape
        assert rgb.shape[:2] == mask.shape
        assert bbox_2d.shape == (4,)
        assert t_wc.shape == (4, 4,)
        # 给背景赋予不同的参数，包括网络结构和关键帧数量
        if self.do_bg and self.obj_id == 0: # do seperate bg
            self.obj_scale = cfg.bg_scale
            self.hidden_feature_size = cfg.hidden_feature_size_bg
            self.n_bins_cam2surface = cfg.n_bins_cam2surface_bg
            self.keyframe_step = cfg.keyframe_step_bg
        else:
            self.obj_scale = cfg.obj_scale
            self.hidden_feature_size = cfg.hidden_feature_size
            self.n_bins_cam2surface = cfg.n_bins_cam2surface
            self.keyframe_step = cfg.keyframe_step

        self.frames_width = rgb.shape[0]
        self.frames_height = rgb.shape[1]

        self.min_bound = cfg.min_depth
        self.max_bound = cfg.max_depth
        self.n_bins = cfg.n_bins
        self.n_unidir_funcs = cfg.n_unidir_funcs

        self.surface_eps = cfg.surface_eps
        self.stop_eps = cfg.stop_eps

        self.n_keyframes = 1  # Number of keyframes
        self.kf_pointer = None
        self.keyframe_buffer_size = cfg.keyframe_buffer_size
        # 把当前帧加入进来作为关键帧
        self.kf_id_dict = bidict({live_frame_id:0})
        self.kf_buffer_full = False
        # 现在有多少关键帧进来了啦
        self.frame_cnt = 0  # number of frames taken in
        self.lastest_kf_queue = []
        
        # 存储物体整体的clip和caption特征
        self.feat_cnt = 1  
        self.clip_feat = clip_feat
        self.caption_feat = caption_feat
        
        self.eps_fine_vis = cfg.eps_fine_vis
        self.n_bins_fine_vis = cfg.n_bins_fine_vis

        # 在这些关键帧中的bbox
        self.bbox = torch.empty(  # obj bounding bounding box in the frame
            self.keyframe_buffer_size,
            4,
            device=self.data_device)  # [u low, u high, v low, v high]
        # 把第一个bbox加进来
        self.bbox[0] = bbox_2d

        # RGB + pixel state batch
        self.rgb_idx = slice(0, 3)
        self.state_idx = slice(3, 4)
        # rgb的切片
        self.rgbs_batch = torch.empty(self.keyframe_buffer_size,
                                      self.frames_width,
                                      self.frames_height,
                                      4,
                                      dtype=torch.uint8,
                                      device=self.data_device)
        # b部件级特征
        if self.part_mode:
            # if part_feat is None:
            #     print("No way!!! part_feat is lost!!!")
            # 这个是partfeat降采样的大小
            self.part_down = cfg.part_down
            # 只保存编号，不然太大了，用到那个图像的特征就保存一下id，时刻紧跟rgbs的节奏
            self.use_frame = np.zeros(self.keyframe_buffer_size)
            self.use_frame[0] = live_frame_id
            # self.partfeat_batch = torch.empty(self.keyframe_buffer_size,
            #                             self.frames_width//self.part_down,
            #                             self.frames_height//self.part_down,
            #                             part_feat.shape[-1],
            #                             device=self.data_device)
            # self.partfeat_batch[0] = part_feat

        # Pixel states:
        self.other_obj = 0  # pixel doesn't belong to obj
        self.this_obj = 1  # pixel belong to obj 
        self.unknown_obj = 2  # pixel state is unknown
        
        # 这个物体的语义标签
        
        self.semantic_id = None

        # Initialize first frame rgb and pixel state
        # 第一个rgb切片，前三位是rgb值，最后一位是状态，是否输入这个物体
        self.rgbs_batch[0, :, :, self.rgb_idx] = rgb
        self.rgbs_batch[0, :, :, self.state_idx] = mask[..., None]
        

        self.depth_batch = torch.empty(self.keyframe_buffer_size,
                                       self.frames_width,
                                       self.frames_height,
                                       dtype=torch.float32,
                                       device=self.data_device)

        # 把当前帧的深入加入进来
        self.depth_batch[0] = depth
        self.t_wc_batch = torch.empty(
            self.keyframe_buffer_size, 4, 4,
            dtype=torch.float32,
            device=self.data_device)  # world to camera transform

        # Initialize first frame's world2cam transform
        # 当前帧的位姿也加入进来
        self.t_wc_batch[0] = t_wc

        # 构建一个神经隐式场
        trainer_cfg = copy.deepcopy(cfg)
        trainer_cfg.obj_id = self.obj_id
        trainer_cfg.hidden_feature_size = self.hidden_feature_size
        trainer_cfg.obj_scale = self.obj_scale
        self.trainer = trainer.Trainer(trainer_cfg)

        # 3D boundary
        self.bbox_final = False
        self.serialized_bbox = None
        self.bbox3d = None
        self.bbox3dour = None
        self.pc = []
        self.obj_center = torch.tensor(0.0)


    # @profile
    def append_keyframe(self, rgb:torch.tensor, depth:torch.tensor, mask:torch.tensor, bbox_2d:torch.tensor, t_wc:torch.tensor, 
                        frame_id:np.uint8=1,clip_feat=None, caption_feat=None):
        '''
        把一个新的帧引入进来
        '''
        assert rgb.shape[:2] == depth.shape
        assert rgb.shape[:2] == mask.shape
        assert bbox_2d.shape == (4,)
        assert t_wc.shape == (4, 4,)
        assert self.n_keyframes <= self.keyframe_buffer_size - 1
        assert rgb.dtype == torch.uint8
        assert mask.dtype == torch.uint8
        assert depth.dtype == torch.float32

        # every kf_step choose one kf 根据步长判断是否需要为改物体创建一个新的关键帧，其实是在判断上一帧是否为关键
        is_kf = (self.frame_cnt % self.keyframe_step == 0) or self.n_keyframes == 1
        # print("---------------------")
        # print("self.kf_id_dict ", self.kf_id_dict)
        # print("live frame id ", frame_id)
        # print("n_frames ", self.n_keyframes)
        # 如果关键帧满了，也要把最后一个位置让给当前帧
        if self.n_keyframes == self.keyframe_buffer_size - 1:  # kf buffer full, need to prune
            self.kf_buffer_full = True
            if self.kf_pointer is None:
                self.kf_pointer = self.n_keyframes
            # 把当前帧设置为关键帧列表中空闲的那个
            self.rgbs_batch[self.kf_pointer, :, :, self.rgb_idx] = rgb
            self.rgbs_batch[self.kf_pointer, :, :, self.state_idx] = mask[..., None]
            self.depth_batch[self.kf_pointer, ...] = depth
            self.t_wc_batch[self.kf_pointer, ...] = t_wc
            self.bbox[self.kf_pointer, ...] = bbox_2d
            self.kf_id_dict.inv[self.kf_pointer] = frame_id
            if self.part_mode:
                self.use_frame[self.kf_pointer] = frame_id
            #     if part_feat is None:
            #         print("No way!!! part_feat is lost!!!")
            #     self.partfeat_batch[self.kf_pointer] = part_feat
                
            if is_kf:
                # 如果当前帧还是关键帧的话，更要把这个加进来，并继续删一个备后面用
                self.lastest_kf_queue.append(self.kf_pointer)
                pruned_frame_id, pruned_kf_id = self.prune_keyframe()
                self.kf_pointer = pruned_kf_id
                print("pruned kf id ", self.kf_pointer)
        # 如果关键帧没有满
        else:
            if not is_kf:   # not kf, replace
                # 如果上一帧不是关键帧，不用加进来，只拿去训练就行，随时准备备替换
                self.rgbs_batch[self.n_keyframes-1, :, :, self.rgb_idx] = rgb
                self.rgbs_batch[self.n_keyframes-1, :, :, self.state_idx] = mask[..., None]
                if self.part_mode:
                    self.use_frame[self.n_keyframes-1] = frame_id
                #     if part_feat is None:
                #         print("No way!!! part_feat is lost!!!")
                #     self.partfeat_batch[self.n_keyframes-1] = part_feat
                self.depth_batch[self.n_keyframes-1, ...] = depth
                self.t_wc_batch[self.n_keyframes-1, ...] = t_wc
                self.bbox[self.n_keyframes-1, ...] = bbox_2d
                self.kf_id_dict.inv[self.n_keyframes-1] = frame_id
            else:   # is kf, add new kf
                # 如果上一帧是关键帧，保留，随意加到最后
                self.kf_id_dict[frame_id] = self.n_keyframes
                self.rgbs_batch[self.n_keyframes, :, :, self.rgb_idx] = rgb
                self.rgbs_batch[self.n_keyframes, :, :, self.state_idx] = mask[..., None]
                if self.part_mode:
                    self.use_frame[self.n_keyframes] = frame_id
                #     if part_feat is None:
                #         print("No way!!! part_feat is lost!!!")
                #     self.partfeat_batch[self.n_keyframes] = part_feat
                self.depth_batch[self.n_keyframes, ...] = depth
                self.t_wc_batch[self.n_keyframes, ...] = t_wc
                self.bbox[self.n_keyframes, ...] = bbox_2d
                self.lastest_kf_queue.append(self.n_keyframes)
                self.n_keyframes += 1
        # print("self.kf_id_dic ", self.kf_id_dict)
        self.frame_cnt += 1
        # 堆叠clip和caption特征
        if clip_feat is not None:
            # 这样计算应该不需要归一化了
            self.clip_feat = np.vstack((self.clip_feat,clip_feat))
            self.caption_feat = np.vstack((self.caption_feat,caption_feat))
            self.feat_cnt += 1
            
        if len(self.lastest_kf_queue) > 2:  # keep latest two frames
            self.lastest_kf_queue = self.lastest_kf_queue[-2:]

    def prune_keyframe(self):
        '''
        随机选一个关键帧删除
        '''
        key, value = random.choice(list(self.kf_id_dict.items())[:-2])  # do not prune latest two frames
        return key, value

    def pcd_denoise_dbscan(self, pcd: o3d.geometry.PointCloud, eps=0.02, min_points=10) -> o3d.geometry.PointCloud:
        '''
        过滤杂点
        '''
        pcd_clusters = pcd.cluster_dbscan(
            eps=eps,
            min_points=min_points,
        )
        obj_points = np.asarray(pcd.points)
        pcd_clusters = np.array(pcd_clusters)
        counter = Counter(pcd_clusters)
        if counter and (-1 in counter):
            del counter[-1]
        if counter:
            most_common_label, _ = counter.most_common(1)[0]
            largest_mask = pcd_clusters == most_common_label
            largest_cluster_points = obj_points[largest_mask]
            if len(largest_cluster_points) < 5:
                return pcd
            largest_cluster_pcd = o3d.geometry.PointCloud()
            largest_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)
            pcd = largest_cluster_pcd
        return pcd
    
    
    def set_semantic(self, semantic_id):
        self.semantic_id = semantic_id

    def get_bound(self, intrinsic_open3d, final=False):
        '''
        根据历史所有关键帧的mask和深度depth,得到这个物体的3d边界
        '''
        # 只计算一次即可啊
        if self.bbox_final is True:
            # bbox3d是open3d格式，bbox是我们自己的格式
            return self.bbox3d, self.bbox3dour
        
        # print("Getting bound...")
        self.bbox_final = final
        # get 3D boundary from posed depth img   
        pcs = open3d.geometry.PointCloud()
        # print("kf id dict ", self.kf_id_dict)
        # 先把所有的点云累加起来，计算量太大了
        for kf_id in range(self.n_keyframes):
            mask = self.rgbs_batch[kf_id, :, :, self.state_idx].squeeze() == self.this_obj
            depth = self.depth_batch[kf_id].cpu().clone()
            twc = self.t_wc_batch[kf_id].cpu().numpy()
            depth[~mask] = 0
            depth = depth.permute(1,0).numpy().astype(np.float32)
            T_CW = np.linalg.inv(twc)
            pc = open3d.geometry.PointCloud.create_from_depth_image(depth=open3d.geometry.Image(np.asarray(depth, order="C")), intrinsic=intrinsic_open3d, extrinsic=T_CW)
            # 不用过滤了，因为已经搞好了
            # if self.obj_id != 0:
            #     print(self.obj_id,kf_id)
            #     pc = self.pcd_denoise_dbscan(
            #         pc, 
            #         eps=0.05, 
            #         min_points=10
            #     )
            # pc = pc.voxel_down_sample(voxel_size=0.05)
            # self.pc += pc
            pcs += pc
        # bbox不用太严格，降采样一下
        pcs = pcs.voxel_down_sample(voxel_size=0.05)
        # 过滤一下杂点，处理背景，因为可能有很多东西组成
        # if self.obj_id != 0:
        #     pcs = self.pcd_denoise_dbscan(
        #         pcs, 
        #         eps=0.05, 
        #         min_points=10
        #     )
        # trimesh has a better minimal bbox implementation than open3d
        # 在用trimesh计算最小边界
        # open3d.visualization.draw_geometries([pcs])
        try:
            transform, extents = trimesh.bounds.oriented_bounds(np.array(pcs.points))  # pc
            transform = np.linalg.inv(transform)
        except scipy.spatial._qhull.QhullError:
            print("too few pcs obj ")
            return None, None
        # 最少10cm，两边各往外扩展10%
        for i in range(extents.shape[0]):
            # extents[i] = extents[i]*1.2
            extents[i] = np.maximum(extents[i], 0.10)  # at least rendering 10cm
        bbox = utils.BoundingBox()
        bbox.center = transform[:3, 3]
        bbox.R = transform[:3, :3]
        bbox.extent = extents
        # 计算包围盒的半边长
        half_extent = bbox.extent / 2
        # 定义包围盒的八个顶点相对于中心的偏移量
        corners_offsets = np.array([
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1]
        ])
        # 计算包围盒的八个顶点坐标
        corners = np.dot(corners_offsets * half_extent, bbox.R.T) + bbox.center
        # 将计算得到的顶点坐标赋给points3d
        bbox.points3d = corners
        self.bbox3dour = bbox
        # open3d类型
        bbox3d = open3d.geometry.OrientedBoundingBox(bbox.center, bbox.R, bbox.extent)
        # 物体最小也得是5cm
        min_extent = 0.05
        bbox3d.extent = np.maximum(min_extent, bbox3d.extent)
        bbox3d.color = (255,0,0)
        self.bbox3d = bbox3d
        # 转为自己的格式
        self.pc = []
        # open3d.visualization.draw_geometries([bbox3d, pcs])
        # print("obj ", self.obj_id)
        # print("bound ", bbox3d)
        # print("kf id dict ", self.kf_id_dict)
        # print("Getted bound!!!")
        return bbox3d, bbox
    
    
    



    def get_training_samples(self, n_frames, n_samples, cached_rays_dir, global_partfeat):
        # 从现有的数据中，得到训练数据，采样n_frames帧，每帧n_samples个点
        # Sample pixels
        # 先随机采样n_frames - 2次
        if self.n_keyframes > 2: # make sure latest 2 frames are sampled    
            keyframe_ids = torch.randint(low=0,
                                         high=self.n_keyframes,
                                         size=(n_frames - 2,),
                                         dtype=torch.long,
                                         device=self.data_device)
            # if self.kf_buffer_full:
            # latest_frame_ids = list(self.kf_id_dict.values())[-2:]
            # 在确保最近的两帧被采样到了
            latest_frame_ids = self.lastest_kf_queue[-2:]
            keyframe_ids = torch.cat([keyframe_ids,
                                          torch.tensor(latest_frame_ids, device=keyframe_ids.device)])
            # print("latest_frame_ids", latest_frame_ids)
            # else:   # sample last 2 frames
            #     keyframe_ids = torch.cat([keyframe_ids,
            #                               torch.tensor([self.n_keyframes-2, self.n_keyframes-1], device=keyframe_ids.device)])
        else:
            keyframe_ids = torch.randint(low=0,
                                         high=self.n_keyframes,
                                         size=(n_frames,),
                                         dtype=torch.long,
                                         device=self.data_device)
        keyframe_ids = torch.unsqueeze(keyframe_ids, dim=-1)
        # 随机采样像素，现在的值是在0-1
        idx_w = torch.rand(n_frames, n_samples, device=self.data_device)
        idx_h = torch.rand(n_frames, n_samples, device=self.data_device)

        # resize到bbox的尺寸
        idx_w = idx_w * (self.bbox[keyframe_ids, 1] - self.bbox[keyframe_ids, 0]) + self.bbox[keyframe_ids, 0]
        idx_h = idx_h * (self.bbox[keyframe_ids, 3] - self.bbox[keyframe_ids, 2]) + self.bbox[keyframe_ids, 2]

        idx_w_long = idx_w.long()
        idx_h_long = idx_h.long()
        # 得到当前批次所有采样像素的rgb值和深度值，注意rgb里面包含mask判断是否属于该物体
        sampled_rgbs = self.rgbs_batch[keyframe_ids, idx_w_long, idx_h_long]
        sampled_depth = self.depth_batch[keyframe_ids, idx_w_long, idx_h_long]

        # 得到采样像素这些位置的光线方向
        sampled_ray_dirs = cached_rays_dir[idx_w_long, idx_h_long]

        # 得到采样的相机位姿
        sampled_twc = self.t_wc_batch[keyframe_ids[:, 0], :, :]
        # 从而把采样像素的方向转到全局坐标系
        origins, dirs_w = utils.origin_dirs_W(sampled_twc, sampled_ray_dirs)
        
        # 采样的部件级特征
        sampled_partfeat = None
        if self.part_mode:
            use_frame = torch.tensor(self.use_frame).to(self.data_device)
            sample_frameid = use_frame[keyframe_ids]
            sample_frameid = (sample_frameid/self.stride).long()
            idx_w = torch.floor(idx_w / self.part_down).long()
            idx_h = torch.floor(idx_h / self.part_down).long()
            # if (sample_frameid < 0).any() or (sample_frameid >= global_partfeat.shape[0]).any():
            #     print("sample_frameid 超出范围:")
            #     print("超出的索引值:", sample_frameid[(sample_frameid < 0) | (sample_frameid >= global_partfeat.shape[0])])
            # if (idx_w < 0).any() or (idx_w >= global_partfeat.shape[1]).any():
            #     print("idx_w 超出范围:")
            #     print("超出的索引值:", idx_w[(idx_w < 0) | (idx_w >= global_partfeat.shape[1])])
            # if (idx_h < 0).any() or (idx_h >= global_partfeat.shape[2]).any():
            #     print("idx_h 超出范围:")
            #     print("超出的索引值:", idx_h[(idx_h < 0) | (idx_h >= global_partfeat.shape[2])])
            sampled_partfeat = global_partfeat[sample_frameid, idx_w, idx_h]

        return self.sample_3d_points(sampled_rgbs, sampled_depth, origins, dirs_w, sampled_partfeat=sampled_partfeat)

    def sample_3d_points(self, sampled_rgbs, sampled_depth, origins, dirs_w, sampled_partfeat=None):
        """
        3D sampling strategy，N=n_bins_cam2surface，M=n_bins
        在原点origins，方向dirs_w的这些光线上采样，光线末端rgb为sampled_rgbs，长度为sampled_depth
        * For pixels with invalid depth:
            - N+M from minimum bound to max (stratified)
            # 如果深度值无效，则直接到max均匀采样
        * For pixels with valid depth:
            # 深度值有效，且属于该物体，N+M
            # Pixel belongs to this object
                - N from cam to surface (stratified)
                - M around surface (stratified/normal)
            # 深度值有效，但不属于该物体(sampled_rgbs的最后一位可以判断)，N+M个均匀采样
            # Pixel belongs that don't belong to this object
                - N from cam to surface (stratified)
                - M around surface (stratified)
            # 如果是未知状态，不知道属不属于该物体，啥也不做
            # Pixel with unknown state
                - Do nothing!
        """

        n_bins_cam2surface = self.n_bins_cam2surface
        n_bins = self.n_bins
        eps = self.surface_eps
        other_objs_max_eps = self.stop_eps #0.05  
        # print("max depth ", torch.max(sampled_depth))
        # 初始化一个采样点，大小为(N*n_rays, n_bins_cam2surface + n_bins)
        sampled_z = torch.zeros(
            sampled_rgbs.shape[0] * sampled_rgbs.shape[1],
            n_bins_cam2surface + n_bins,
            dtype=self.depth_batch.dtype,
            device=self.data_device)  # shape (N*n_rays, n_bins_cam2surface + n_bins)
        # 深度值比最小深度0还小，则是无效深度值
        invalid_depth_mask = (sampled_depth <= self.min_bound).view(-1)
        # 注意在读取数据时，已经不会有深度超过max_bound了
        # max_bound = self.max_bound
        # 最大边界是深度值的位置
        max_bound = torch.max(sampled_depth)
        # sampling for points with invalid depth
        invalid_depth_count = invalid_depth_mask.count_nonzero()
        # 如果有深度值无效的，从0到所有深度中的最大值，均匀采样
        if invalid_depth_count:
            sampled_z[invalid_depth_mask, :] = utils.stratified_bins(
                self.min_bound, max_bound,
                n_bins_cam2surface + n_bins, invalid_depth_count,
                device=self.data_device)

        # 下面采样有效深度值的像素
        valid_depth_mask = ~invalid_depth_mask
        valid_depth_count = valid_depth_mask.count_nonzero()
        # 对于有效深度值
        if valid_depth_count:
            # 现在下界和深度之间均匀采样n_bins_cam2surface个深度
            sampled_z[valid_depth_mask, :n_bins_cam2surface] = utils.stratified_bins(
                self.min_bound, sampled_depth.view(-1)[valid_depth_mask]-eps,
                n_bins_cam2surface, valid_depth_count, device=self.data_device)

            # 如果属于该物体，在深度值±0.1m附近高斯采样
            obj_mask = (sampled_rgbs[..., -1] == self.this_obj).view(-1) & valid_depth_mask 
            assert sampled_z.shape[0] == obj_mask.shape[0]
            obj_count = obj_mask.count_nonzero()
            if obj_count:
                sampling_method = "normal"  # stratified or normal
                if sampling_method == "stratified":
                    sampled_z[obj_mask, n_bins_cam2surface:] = utils.stratified_bins(
                        sampled_depth.view(-1)[obj_mask] - eps, sampled_depth.view(-1)[obj_mask] + eps,
                        n_bins, obj_count, device=self.data_device)
                elif sampling_method == "normal":
                    sampled_z[obj_mask, n_bins_cam2surface:] = utils.normal_bins_sampling(
                        sampled_depth.view(-1)[obj_mask],
                        n_bins,
                        obj_count,
                        delta=eps,
                        device=self.data_device)
                else:
                    raise (
                        f"sampling method not implemented {sampling_method}, \
                            stratified and normal sampling only currenty implemented."
                    )
            # 对于bbox的其他的，在表面附近均匀采样，奇怪为啥采样其他物体啊，并没有区分是否为该物体
            other_obj_mask = (sampled_rgbs[..., -1] != self.this_obj).view(-1) & valid_depth_mask
            other_objs_count = other_obj_mask.count_nonzero()
            if other_objs_count:
                sampled_z[other_obj_mask, n_bins_cam2surface:] = utils.stratified_bins(
                    sampled_depth.view(-1)[other_obj_mask] - eps,
                    sampled_depth.view(-1)[other_obj_mask] + other_objs_max_eps,
                    n_bins, other_objs_count, device=self.data_device)
        
        sampled_z = sampled_z.view(sampled_rgbs.shape[0],
                                   sampled_rgbs.shape[1],
                                   -1)  # view as (n_rays, n_samples, 10)
        # 成为真正的采样点
        input_pcs = origins[..., None, None, :] + (dirs_w[:, :, None, :] *
                                                   sampled_z[..., None])
        # obj_center=0
        input_pcs -= self.obj_center
        # 还要说吗那些属于该物体哦
        obj_labels = sampled_rgbs[..., -1].view(-1)
        return sampled_rgbs[..., :3], sampled_depth, valid_depth_mask, obj_labels, input_pcs, sampled_z, sampled_partfeat

    def save_checkpoints(self, path, epoch):
        '''
        保存当前物体的参数，包括网络权重，实例id，bbox和需要缩放的大小
        '''
        obj_id = self.obj_id
        # chechpoint_load_file = (path + "/obj_" + str(obj_id) + "_frame_" + str(epoch) + ".pth")
        chechpoint_load_file = (path + "/obj_" + str(obj_id) + ".pth")
        torch.save(
            {
                "epoch": epoch,
                "FC_state_dict": self.trainer.fc_occ_map.state_dict(),
                "PE_state_dict": self.trainer.pe.state_dict(),
                "obj_id": self.obj_id,
                "bbox": self.bbox3dour,
                "obj_scale": self.trainer.obj_scale,
                "clip_feat": self.clip_feat,
                "caption_feat": self.caption_feat,
                "semantic_id": self.semantic_id,
            },
            chechpoint_load_file,
        )
        # optimiser?

    def load_checkpoints(self, ckpt_file):
        '''
        可以加载保存的物体的参数，可用于可视化等
        '''
        checkpoint_load_file = (ckpt_file)
        if not os.path.exists(checkpoint_load_file):
            print("ckpt not exist ", checkpoint_load_file)
            return
        checkpoint = torch.load(checkpoint_load_file)
        self.trainer.fc_occ_map.load_state_dict(checkpoint["FC_state_dict"])
        self.trainer.pe.load_state_dict(checkpoint["PE_state_dict"])
        self.obj_id = checkpoint["obj_id"]
        self.bbox3dour = checkpoint["bbox"]
        self.trainer.obj_scale = checkpoint["obj_scale"]
        self.trainer.fc_occ_map.to(self.training_device)
        self.trainer.pe.to(self.training_device)
        if "clip_feat" not in checkpoint.keys():
            print("no clip_feat for this obj:", self.obj_id)
            return False
        self.clip_feat = checkpoint["clip_feat"]
        self.caption_feat = checkpoint["caption_feat"]
        self.semantic_id = checkpoint["semantic_id"]
        self.bbox_final = True
        return True
        
    def render_2D_syn(self, T_WC, intrinsic_open3d, cached_rays_dir, T_WO=None, chunk_size=1000, do_fine=True, obj_mask=None, render_part=False):
        '''
        给定一个T_WC, 得到某个物体mask的渲染
        '''
        # used in viz thread
        # render 2D view ----------------------------------------------------
        W = self.trainer.W_vis
        H = self.trainer.H_vis
        _, bbox = self.get_bound(intrinsic_open3d, final=True)
        
        
        if obj_mask is None:
            obj_mask = np.ones([W, H], dtype=np.bool)
        indices_h, indices_w = np.where(obj_mask)
        self.trainer.T_WC_gt = torch.from_numpy(T_WC).float().unsqueeze(0).repeat_interleave(len(indices_h), dim=0)
        indices_h = torch.from_numpy(indices_h)
        indices_w = torch.from_numpy(indices_w)
            
        self.trainer.dirs_C_gt = cached_rays_dir[indices_h, indices_w].to(indices_h.device)
        obj_hit, obj_near, obj_far = self.trainer.sample_points_bbox(bbox, do_eval=True)
        if obj_hit is None:
            # print("None hit")
            return None, None, None
        # 按照bbox命中程度设置mask
        obj_ma = obj_mask[obj_mask]
        obj_ma[~obj_hit] = False
        obj_mask[obj_mask] = obj_ma

        n_pts = self.trainer.input_pcs.shape[0]
        # print("rays ", n_pts)
        if n_pts <= 1:
            print("too few hits")
            return None, None, None
        n_chunks = int(np.ceil(n_pts / chunk_size))
        # self.trainer.input_pcs = self.trainer.input_pcs.to(self.device_vis)
        alpha = []
        color = []
        if render_part:
            partfeat = []
        # print("Calculating...")
        with torch.no_grad():
            for n in range(n_chunks):
                start = n * chunk_size
                end = start + chunk_size
                chunk = self.trainer.input_pcs[start:end, :].to(self.training_device)
                points_embedding = self.trainer.pe(chunk)
                alpha_chunk, color_chunk, partfeat_chunk = self.trainer.fc_occ_map(
                    points_embedding, do_color=True)    # self.trainer.dirs_W.to(self.device_vis)
                alpha.append(alpha_chunk.detach())
                color.append(color_chunk.detach())
                if render_part:
                    partfeat.append(partfeat_chunk)
        alpha = torch.cat(alpha, dim=0).squeeze(dim=-1)
        color = torch.cat(color, dim=0).squeeze(dim=-1)
        if render_part:
            partfeat = torch.cat(partfeat, dim=0).squeeze(dim=-1)
        z_vals = self.trainer.z_vals
        # print("Calculated!!!")
        occupancy = render_rays.occupancy_activation(alpha)
        termination = render_rays.occupancy_to_termination(occupancy).cpu()
        opacity = torch.sum(termination, dim=-1)
        opacity_mask = opacity < 0.9
        # plt.plot(termination[0])
        # plt.plot(occupancy[0].cpu().numpy())
        # plt.show()
        render_depth = render_rays.render(termination, z_vals).view(n_pts).detach().cpu().numpy()
        render_color = render_rays.render(termination[..., None], color.cpu(), dim=-2).view(n_pts,3).detach().cpu().numpy()
        render_color = (render_color * 255).astype(np.uint8)
        depth_mask = (render_depth < obj_near.numpy()) | (render_depth > obj_far.numpy()) | opacity_mask.numpy()
        render_depth = render_depth[~depth_mask]
        render_color = render_color[~depth_mask]
        
        render_partfeat = None
        if render_part:
            render_partfeat = render_rays.render(termination[..., None], partfeat.cpu(), dim=-2).view(n_pts,-1).detach().cpu().numpy()
            render_partfeat = render_partfeat[~depth_mask]
        # 再按照是否穿透设置mask
        obj_ma = obj_mask[obj_mask]
        obj_ma[depth_mask] = False
        obj_mask[obj_mask] = obj_ma
        
        return (obj_mask, render_depth, render_color, render_partfeat,)

class cameraInfo:

    def __init__(self, cfg) -> None:
        self.device = cfg.data_device
        self.width = cfg.W  # Frame width
        self.height = cfg.H  # Frame height

        self.fx = cfg.fx
        self.fy = cfg.fy
        self.cx = cfg.cx
        self.cy = cfg.cy

        self.rays_dir_cache = self.get_rays_dirs()

    def get_rays_dirs(self, depth_type="z"):
        '''
        根据相机的内参，得到每个像素对应的光线方向
        '''
        idx_w = torch.arange(end=self.width, device=self.device)
        idx_h = torch.arange(end=self.height, device=self.device)

        dirs = torch.ones((self.width, self.height, 3), device=self.device)

        dirs[:, :, 0] = ((idx_w - self.cx) / self.fx)[:, None]
        dirs[:, :, 1] = ((idx_h - self.cy) / self.fy)

        if depth_type == "euclidean":
            raise Exception(
                "Get camera rays directions with euclidean depth not yet implemented"
            )
            norm = torch.norm(dirs, dim=-1)
            dirs = dirs * (1. / norm)[:, :, :, None]

        return dirs
    
    


