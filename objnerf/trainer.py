import torch
import model
import embedding
import render_rays
import numpy as np
import vis
from tqdm import tqdm
import open3d as o3d 
import utils

class Trainer:
    def __init__(self, cfg):
        self.obj_id = cfg.obj_id
        self.device = cfg.training_device
        self.hidden_feature_size = cfg.hidden_feature_size #32 for obj  # 256 for iMAP, 128 for seperate bg
        self.clip_point_feature_size = cfg.clip_point_feature_size # 512
        self.obj_scale = cfg.obj_scale # 5 for bg, 10 for iMAP
        self.n_unidir_funcs = cfg.n_unidir_funcs
        # 总的位置编码被按照3:2大概得比例拆成了两部分
        self.emb_size1 = 21*(3+1)+3
        self.emb_size2 = 21*(5+1)+3 - self.emb_size1
        # 加载位置编码和整个MLP
        self.load_network()
        # 如果是背景，在构建mesh时收一点就行，其他收多点，因为当时建图时给他扩大了
        if self.obj_id == 0:
            self.bound_extent = 0.995
        else:
            self.bound_extent = 0.9
        # 一些可视化渲染有关的新加参数
        self.W_vis = cfg.W
        self.H_vis = cfg.H
        self.T_WC_gt = None
        self.dirs_C_gt = None
        self.input_pcs = None

    def load_network(self):
        self.fc_occ_map = model.OccupancyMap(
            self.emb_size1,
            self.emb_size2,
            hidden_size=self.hidden_feature_size,
            clip_size=self.clip_point_feature_size,
        )
        self.fc_occ_map.apply(model.init_weights).to(self.device)
        self.pe = embedding.UniDirsEmbed(max_deg=self.n_unidir_funcs, scale=self.obj_scale).to(self.device)

    def meshing(self, bound, obj_center, grid_dim=256, save_pcd=True, save_mesh=True, if_color=False, if_part=False):
        '''
        得到物体的mesh，open3d格式
        '''
        occ_range = [-1., 1.]
        range_dist = occ_range[1] - occ_range[0]
        scene_scale_np = bound.extent / (range_dist * self.bound_extent)
        scene_scale = torch.from_numpy(scene_scale_np).float().to(self.device)
        transform_np = np.eye(4, dtype=np.float32)
        transform_np[:3, 3] = bound.center
        transform_np[:3, :3] = bound.R
        # transform_np = np.linalg.inv(transform_np)  #
        transform = torch.from_numpy(transform_np).to(self.device)
        grid_pc = render_rays.make_3D_grid(occ_range=occ_range, dim=grid_dim, device=self.device,
                                           scale=scene_scale, transform=transform).view(-1, 3)
        grid_pc -= obj_center.to(grid_pc.device)
        ret = self.eval_points(grid_pc)
        if ret is None:
            return None, None
        occ, colors, _ = ret
        pcd = None
        mesh = None
        partfeat = None
        # 如果只是展示点云的话
        if save_pcd:
            mask_valid = occ > 0.5
            points = grid_pc[mask_valid]
            colors = colors[mask_valid]
            # 创建 Open3D 点云对象
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy())
            pcd.colors = o3d.utility.Vector3dVector(colors.cpu().numpy()) 
            # 点云降采样
            pcd.voxel_down_sample(voxel_size=0.02)
        elif save_mesh:
            mesh = vis.marching_cubes(occ.view(grid_dim, grid_dim, grid_dim).cpu().numpy())
            if mesh is None:
                print("marching cube failed")
                return None
            # Transform to [-1, 1] range
            mesh.apply_translation([-0.5, -0.5, -0.5])
            mesh.apply_scale(2)
            # Transform to scene coordinates
            mesh.apply_scale(scene_scale_np)
            mesh.apply_transform(transform_np)
            if if_color or if_part:
                vertices_pts = torch.from_numpy(np.array(mesh.vertices)).float().to(self.device)
                ret = self.eval_points(vertices_pts)
                if ret is None:
                    return None
                _, color, clip = ret
                if if_color:
                    mesh_color = color * 255
                    vertex_colors = mesh_color.detach().squeeze(0).cpu().numpy().astype(np.uint8)
                    mesh.visual.vertex_colors = vertex_colors
                if if_part:
                    partfeat = clip
        return pcd, mesh, partfeat

    def eval_points(self, points, chunk_size=300000): #100000
        '''
        在points的位置计算网络的输出，用于评估或可视化
        '''
        # 256^3 = 16777216，所以一个物体要168批次才能出来
        alpha, color, clip = [], [], []
        n_chunks = int(np.ceil(points.shape[0] / chunk_size))
        with torch.no_grad():
            for k in tqdm(range(n_chunks)): # 2s/it 1000000 pts
                chunk_idx = slice(k * chunk_size, (k + 1) * chunk_size)
                embedding_k = self.pe(points[chunk_idx, ...])
                alpha_k, color_k, clip_k = self.fc_occ_map(embedding_k)
                alpha.extend(alpha_k.detach().squeeze())
                color.extend(color_k.detach().squeeze())
                clip.extend(clip_k.detach().squeeze())
        alpha = torch.stack(alpha)
        color = torch.stack(color)
        clip = torch.stack(clip)
        # 从密度中得到占用概率
        occ = render_rays.occupancy_activation(alpha).detach()
        if occ.max() == 0:
            print("no occ")
            return None
        return (occ, color, clip)
    
    def sample_points_bbox(self, bbox, do_eval=True):
        '''
        在已知的bbox内进行点采样
        '''
        # sample points with known 3D bbox
        # todo coarse and fine
        # rays in world coordinate
        # origins, self.dirs_W = render_rays.origin_dirs_W(self.T_WC_gt, self.dirs_C_gt)
        origins, dirs_W = utils.origin_dirs_W(self.T_WC_gt, self.dirs_C_gt)
        origins = origins.view(-1, 3)
        dirs_W = dirs_W.view(-1, 3)
        # 背景多一些
        if self.obj_id == 0:
            n_bins = 60
        else:
            n_bins = 20
        if do_eval:
            n_bins = 150
        bbox = o3d.geometry.OrientedBoundingBox(bbox.center, bbox.R, bbox.extent)
        ray = dirs_W.numpy().copy()
        ray /= np.linalg.norm(ray, axis=-1, keepdims=True)
        # ray intersection with OBB
        # rotate
        T_WO = torch.eye(4)
        T_WO[:3,:3] = torch.from_numpy(bbox.R)
        T_WO[:3,3] = torch.from_numpy(bbox.center)
        T_WC = self.T_WC_gt[0]
        T_OW = torch.inverse(T_WO)
        T_OC = T_OW @ T_WC
        T_OC_gt = T_OC.float().unsqueeze(0).repeat_interleave(len(self.T_WC_gt), dim=0)
        origins_r, dirs_W_r = utils.origin_dirs_W(T_OC_gt, self.dirs_C_gt)
        ray_r = dirs_W_r.numpy().copy()
        bounds_r = np.concatenate([-bbox.extent.reshape(1,-1)/2.0, bbox.extent.reshape(1,-1)/2.0], axis=0).astype(dtype=np.float32)
        all_near, all_far, all_hit = utils.ray_box_intersection(origins_r, torch.from_numpy(ray_r),
                                                                torch.from_numpy(bounds_r[0]),
                                                                torch.from_numpy(bounds_r[1]))
        if torch.sum(all_hit) <= 1:
            return None, None, None
        all_near = torch.clip(all_near, 0).float()
        all_far = all_far.float() + 0.2 # if cam inside bound extend ray a bit
        hit_mask = all_hit == True
        n_rays = hit_mask[hit_mask==True].shape[0]
        self.dirs_W = dirs_W[hit_mask]
        self.origins = origins[hit_mask]
        self.z_vals_cat = utils.stratified_bins(all_near[hit_mask].reshape(-1, 1).squeeze(),
                                                      all_far[hit_mask].reshape(-1, 1).squeeze(), n_bins,
                                                      n_rays, device = "cpu")
        self.z_vals = 0.5 * (self.z_vals_cat[..., 1:] + self.z_vals_cat[..., :-1])
        self.input_pcs = self.origins[:, None, :] + (self.dirs_W[:, None, :] * self.z_vals[:, :, None])
        # self.input_pcs = self.origins[:, None, :] + (torch.from_numpy(ray[hit_mask][:, None, :]) * self.z_vals[:, :, None])

        # # vis
        # # check AABB ray intersection
        # # because # different ray angle corresponds to different unit length, ray will not stop at the bounds exactly
        # print("Vising")
        # pc = o3d.geometry.PointCloud()
        # # pc_far = origins[hit_mask] + ray[hit_mask]*all_far[hit_mask].numpy().reshape(-1,1)
        # # pc_near = origins[hit_mask] + ray[hit_mask]*all_near[hit_mask].numpy().reshape(-1,1)
        # # pc_show = np.concatenate([pc_near, pc_far])
        # pc_show = self.input_pcs.numpy().reshape(-1,3)
        # pc.points = o3d.utility.Vector3dVector(pc_show)
        # pc.voxel_down_sample(voxel_size=0.03)
        # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        #     size=2, origin=origins[0])
        # # axis_bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(bbox.get_box_points())
        # # o3d.visualization.draw_geometries([pc, bbox, mesh_frame, axis_bbox])    # pc should fill the bbox
        # o3d.visualization.draw_geometries([pc, bbox, mesh_frame])    
        # print("Vised")
        return hit_mask, all_near[hit_mask], all_far[hit_mask]
    












