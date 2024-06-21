import cv2
import imgviz
import numpy as np
import torch
from functorch import combine_state_for_ensemble
import open3d
import queue
import copy
import torch.utils.dlpack
from time import perf_counter_ns
from sklearn.cluster import DBSCAN

class performance_measure:
    '''
    可用于计算耗时
    '''
    def __init__(self, name) -> None:
        self.name = name

    def __enter__(self):
        self.start_time = perf_counter_ns()

    def __exit__(self, type, value, tb):
        self.end_time = perf_counter_ns()
        self.exec_time = self.end_time - self.start_time

        print(f"{self.name} excution time: {(self.exec_time)/1000000:.2f} ms")
        
        
class BoundingBox():
    def __init__(self):
        super(BoundingBox, self).__init__()
        self.extent = None
        self.R = None
        self.center = None
        self.points3d = None    # (8,3)

def bbox_open3d2bbox(bbox_o3d):
    '''
    将open3d的bbox类型转为我们自己的类型
    '''
    bbox = BoundingBox()
    bbox.extent = bbox_o3d.extent
    bbox.R = bbox_o3d.R
    bbox.center = bbox_o3d.center
    return bbox

def bbox_bbox2open3d(bbox):
    '''
    将我们自己的bbox类型转为open3d的类型
    '''
    bbox_o3d = open3d.geometry.OrientedBoundingBox(bbox.center, bbox.R, bbox.extent)
    return bbox_o3d

def update_vmap(models, optimiser):
    '''
    根据优化器，优化模型参数
    '''
    fmodel, params, buffers = combine_state_for_ensemble(models)
    [p.requires_grad_() for p in params]
    optimiser.add_param_group({"params": params})  # imap b l
    return (fmodel, params, buffers)

def enlarge_bbox(bbox, scale, w, h):
    '''
    扩大一些2d的bbox，看到一些周围的物体
    '''
    assert scale >= 0
    # print(bbox)
    min_x, min_y, max_x, max_y = bbox
    margin_x = int(0.5 * scale * (max_x - min_x))
    margin_y = int(0.5 * scale * (max_y - min_y))
    if margin_y == 0 or margin_x == 0:
        return None
    # assert margin_x != 0
    # assert margin_y != 0
    min_x -= margin_x
    max_x += margin_x
    min_y -= margin_y
    max_y += margin_y

    min_x = np.clip(min_x, 0, w-1)
    min_y = np.clip(min_y, 0, h-1)
    max_x = np.clip(max_x, 0, w-1)
    max_y = np.clip(max_y, 0, h-1)

    bbox_enlarged = [int(min_x), int(min_y), int(max_x), int(max_y)]
    return bbox_enlarged

def get_bbox2d(obj_mask, bbox_scale=1.0):
    '''
    根据物体的mask，得到2d的bbox
    '''
    contours, hierarchy = cv2.findContours(obj_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[
                          -2:]
    # # Find the index of the largest contour
    # areas = [cv2.contourArea(c) for c in contours]
    # max_index = np.argmax(areas)
    # cnt = contours[max_index]
    # Concatenate all contours
    if len(contours) == 0:
        return None
    cnt = np.concatenate(contours)
    x, y, w, h = cv2.boundingRect(cnt) 
    # x, y, w, h = cv2.boundingRect(contours)
    bbox_enlarged = enlarge_bbox([x, y, x + w, y + h], scale=bbox_scale, w=obj_mask.shape[1], h=obj_mask.shape[0])
    return bbox_enlarged

def get_bbox2d_batch(img):
    '''
    输入的是图像尺寸，计算True像素点的bbox
    '''
    b,h,w = img.shape[:3]
    rows = torch.any(img, axis=2)
    cols = torch.any(img, axis=1)
    rmins = torch.argmax(rows.float(), dim=1)
    rmaxs = h - torch.argmax(rows.float().flip(dims=[1]), dim=1)
    cmins = torch.argmax(cols.float(), dim=1)
    cmaxs = w - torch.argmax(cols.float().flip(dims=[1]), dim=1)

    return rmins, rmaxs, cmins, cmaxs


# for association/tracking
class InstData:
    # 实例类型，用于物体融合跟踪
    def __init__(self):
        super(InstData, self).__init__()
        self.bbox3D = None
        self.inst_id = None     # instance
        self.class_id = None    # semantic
        self.pc_sample = None
        self.merge_cnt = 0  # merge times counting
        self.cmp_cnt = 0



def get_majority_cluster_mean(vectors, eps, min_samples):
    '''
    对特征进行聚类，只选取比较多的哪一类
    '''
    # 使用DBSCAN算法
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    # print("The all num is ",vectors.shape[0])
    cluster_labels = dbscan.fit_predict(vectors)
    # 统计各个类别的数量
    unique_labels, label_counts = np.unique(cluster_labels, return_counts=True)
    # print("The max num is ",label_counts[np.argmax(label_counts)])
    # 找到数量最多的类别
    majority_label = unique_labels[np.argmax(label_counts)]
    # 找到数量最多的类别对应的向量
    majority_vectors = vectors[cluster_labels == majority_label]
    # 计算数量最多的类别的均值
    majority_mean = np.mean(majority_vectors, axis=0)
    return majority_mean




def box_filter(masks, classes, depth, inst_dict, intrinsic_open3d, T_CW, min_pixels=500, voxel_size=0.01):
    '''
    过滤掉三维中太小的物体，然后得到，最终的标准化inst，这个只需要对scannet处理
    '''
    bbox3d_scale = 1.0  # 1.05
    inst_data = np.zeros_like(depth, dtype=np.int)
    for i in range(len(masks)):
        diff_mask = None
        inst_mask = masks[i]
        inst_id = classes[i]
        if inst_id == 0:
            continue
        inst_depth = np.copy(depth)
        inst_depth[~inst_mask] = 0.  # inst_mask
        # proj_time = time.time()
        # 把非背景的物体，根据相机内参投影为点云
        inst_pc = unproject_pointcloud(inst_depth, intrinsic_open3d, T_CW)
        # print("proj time ", time.time()-proj_time)
        # 如果点数太少，就直接设置为背景
        if len(inst_pc.points) <= 10:  # too small
            inst_data[inst_mask] = 0  # set to background
            continue
        # 如果它被判断属于当前已经构建过的某个物体，其实很准确
        if inst_id in inst_dict.keys():
            candidate_inst = inst_dict[inst_id]
            # iou_time = time.time()
            # 看看那些点在已有的bbox中
            IoU, indices = check_inside_ratio(inst_pc, candidate_inst.bbox3D)
            # print("iou time ", time.time()-iou_time)
            # if indices empty
            candidate_inst.cmp_cnt += 1
            # 只要有一个点在已有bbox，就加进去？不太对劲，看来是id完全就是准确的呗
            if len(indices) >= 1:
                candidate_inst.pc += inst_pc.select_by_index(indices)  # only merge pcs inside scale*bbox
                valid_depth_mask = np.zeros_like(inst_depth, dtype=np.bool)
                valid_pc_mask = valid_depth_mask[inst_depth!=0]
                valid_pc_mask[indices] = True
                valid_depth_mask[inst_depth != 0] = valid_pc_mask
                # valid_mask为True的是深度值不是0，并且在bbox里面的
                valid_mask = valid_depth_mask
                diff_mask = np.zeros_like(inst_mask)
                # uv_opencv, _ = cv2.projectPoints(np.array(inst_pc.select_by_index(indices).points), T_CW[:3, :3],
                #                                  T_CW[:3, 3], intrinsic_open3d.intrinsic_matrix[:3, :3], None)
                # uv = np.round(uv_opencv).squeeze().astype(int)
                # u = uv[:, 0].reshape(-1, 1)
                # v = uv[:, 1].reshape(-1, 1)
                # vu = np.concatenate([v, u], axis=-1)
                # valid_mask = np.zeros_like(inst_mask)
                # valid_mask[tuple(vu.T)] = True
                # # cv2.imshow("valid", (inst_depth!=0).astype(np.uint8)*255)
                # # cv2.waitKey(1)
                # diff_mask为具有深度值但不在bbox里面的部分
                diff_mask[(inst_depth != 0) & (~valid_mask)] = True
                # cv2.imshow("diff_mask", diff_mask.astype(np.uint8) * 255)
                # cv2.waitKey(1)
            else:   # merge all for scannet
                # print("too few pcs obj ", inst_id)
                inst_data[inst_mask] = -1
                continue
            # downsample_time = time.time()
            # adapt_voxel_size = np.maximum(np.max(candidate_inst.bbox3D.extent)/100, 0.1)
            candidate_inst.pc = candidate_inst.pc.voxel_down_sample(voxel_size) # adapt_voxel_size
            # candidate_inst.pc = candidate_inst.pc.farthest_point_down_sample(500)
            # candidate_inst.pc = candidate_inst.pc.random_down_sample(np.minimum(len(candidate_inst.pc.points)/500.,1))
            # print("downsample time ", time.time() - downsample_time)  # 0.03s even
            # bbox_time = time.time()
            try:
                candidate_inst.bbox3D = open3d.geometry.OrientedBoundingBox.create_from_points(candidate_inst.pc.points)
            except RuntimeError:
                # print("too few pcs obj ", inst_id)
                inst_data[inst_mask] = -1
                continue
            # enlarge
            candidate_inst.bbox3D.scale(bbox3d_scale, candidate_inst.bbox3D.get_center())
        # 否则就是一个新的实例了
        else:   # new inst
            # init new inst and new sem
            new_inst = InstData()
            # 有实例id
            new_inst.inst_id = inst_id
            # 腐蚀操作，因为mask边界处数值差？
            smaller_mask = cv2.erode(inst_mask.astype(np.uint8), np.ones((5, 5)), iterations=3).astype(bool)
            # 对于太小的物体不要，用的1500个
            if np.sum(smaller_mask) < min_pixels:
                # print("too few pcs obj ", inst_id)
                inst_data[inst_mask] = 0
                continue
            inst_depth_small = depth.copy()
            inst_depth_small[~smaller_mask] = 0
            # 把腐蚀过的物体投影到点云
            inst_pc_small = unproject_pointcloud(inst_depth_small, intrinsic_open3d, T_CW)
            new_inst.pc = inst_pc_small
            new_inst.pc = new_inst.pc.voxel_down_sample(voxel_size)
            # 获得这个新实例的bbox
            try:
                inst_bbox3D = open3d.geometry.OrientedBoundingBox.create_from_points(new_inst.pc.points)
            except RuntimeError:
                # print("too few pcs obj ", inst_id)
                inst_data[inst_mask] = 0
                continue
            # 把bbox放大一些，得到一个新的
            inst_bbox3D.scale(bbox3d_scale, inst_bbox3D.get_center())
            new_inst.bbox3D = inst_bbox3D
            # 初始化一个新的实例，地图实例
            inst_dict.update({inst_id: new_inst})

        # 更新inst_data
        inst_data[inst_mask] = inst_id
        if diff_mask is not None:
            inst_data[diff_mask] = -1  # unsure area

    return inst_data

def load_matrix_from_txt(path, shape=(4, 4)):
    '''
    scannet加载内参用的
    '''
    with open(path) as f:
        txt = f.readlines()
    txt = ''.join(txt).replace('\n', ' ')
    matrix = [float(v) for v in txt.split()]
    return np.array(matrix).reshape(shape)


def unproject_pointcloud(depth, intrinsic_open3d, T_CW):
    '''
    按照深度图把物体投影为点云
    '''
    # depth, mask, intrinsic, extrinsic -> point clouds
    pc_sample = open3d.geometry.PointCloud.create_from_depth_image(depth=open3d.geometry.Image(depth),
                                                                   intrinsic=intrinsic_open3d,
                                                                   extrinsic=T_CW,
                                                                   depth_scale=1.0,
                                                                   project_valid_depth_only=True)
    return pc_sample

def check_inside_ratio(pc, bbox3D):
    '''
    根据点云和现有的bbox，检查他们的重叠度
    '''
    #  pc, bbox3d -> inside ratio
    indices = bbox3D.get_point_indices_within_bounding_box(pc.points)
    assert len(pc.points) > 0
    ratio = len(indices) / len(pc.points)
    # print("ratio ", ratio)
    return ratio, indices



def ray_box_intersection(origins, directions, bounds_min, bounds_max):
    tmin = (bounds_min - origins) / directions
    tmax = (bounds_max - origins) / directions
    t1 = torch.min(tmin, tmax)
    t2 = torch.max(tmin, tmax)
    near = torch.amax(t1, dim=1)
    far = torch.amin(t2, dim=1)
    hit = near <= far
    front = far>0
    hit = hit & front
    return near, far, hit




def origin_dirs_W(T_WC, dirs_C):
    '''
    把光线转到世界坐标系
    '''
    assert T_WC.shape[0] == dirs_C.shape[0]
    assert T_WC.shape[1:] == (4, 4)
    # assert dirs_C.shape[2] == 3
    if dirs_C.shape[1] == 3:
        dirs_W = torch.matmul(T_WC[:, :3, :3], dirs_C.unsqueeze(-1)).squeeze(-1)
    else:
        dirs_W = (T_WC[:, None, :3, :3] @ dirs_C[..., None]).squeeze()
    origins = T_WC[:, :3, -1]
    return origins, dirs_W




# @torch.jit.script
def stratified_bins(min_depth, max_depth, n_bins, n_rays, type=torch.float32, device = "cuda:0"):
    '''
    在光线上均匀采样
    '''
    # type: (Tensor, Tensor, int, int) -> Tensor

    bin_limits_scale = torch.linspace(0, 1, n_bins+1, dtype=type, device=device)

    if not torch.is_tensor(min_depth):
        min_depth = torch.ones(n_rays, dtype=type, device=device) * min_depth
    
    if not torch.is_tensor(max_depth):
        max_depth = torch.ones(n_rays, dtype=type, device=device) * max_depth
    
    depth_range = max_depth - min_depth
  
    lower_limits_scale = depth_range[..., None] * bin_limits_scale + min_depth[..., None]
    lower_limits_scale = lower_limits_scale[:, :-1]
    
    # print(lower_limits_scale.shape)
    # print(n_rays)
    # print(n_bins)

    assert lower_limits_scale.shape == (n_rays, n_bins)

    bin_length_scale = depth_range / n_bins
    # print(n_rays)
    # print(n_bins)
    # print(bin_length_scale)
    increments_scale = torch.rand(
        n_rays, n_bins, device=device,
        dtype=torch.float32) * bin_length_scale[..., None]

    z_vals_scale = lower_limits_scale + increments_scale

    assert z_vals_scale.shape == (n_rays, n_bins)

    return z_vals_scale

# @torch.jit.script
def normal_bins_sampling(depth, n_bins, n_rays, delta, device = "cuda:0"):
    '''
    表面附近高斯采样
    '''
    # type: (Tensor, int, int, float) -> Tensor

    # device = "cpu"
    # bins = torch.normal(0.0, delta / 3., size=[n_rays, n_bins], devi
        # self.keyframes_batch = torch.empty(self.n_keyframes,ce=device).sort().values
    bins = torch.empty(n_rays, n_bins, dtype=torch.float32, device=device).normal_(mean=0.,std=delta / 3.).sort().values
    bins = torch.clip(bins, -delta, delta)
    z_vals = depth[:, None] + bins

    assert z_vals.shape == (n_rays, n_bins)

    return z_vals


def track_instance(masks, classes, depth, inst_list, sem_dict, intrinsic_open3d, T_CW, IoU_thresh=0.5, voxel_size=0.1,
                   min_pixels=2000, erode=True, clip_features=None, class_names=None):
    device = masks.device
    inst_data_dict = {}
    inst_data_dict.update({0: torch.zeros(depth.shape, dtype=torch.int, device=device)})
    inst_ids = []
    bbox3d_scale = 1.0  
    min_extent = 0.05
    depth = torch.from_numpy(depth).to(device)
    for i in range(len(masks)):
        inst_data = torch.zeros(depth.shape, dtype=torch.int, device=device)
        smaller_mask = cv2.erode(masks[i].detach().cpu().numpy().astype(np.uint8), np.ones((5, 5)), iterations=3).astype(bool)
        inst_depth_small = depth.detach().cpu().numpy()
        inst_depth_small[~smaller_mask] = 0
        inst_pc_small = unproject_pointcloud(inst_depth_small, intrinsic_open3d, T_CW)
        diff_mask = None
        if np.sum(smaller_mask) <= min_pixels:  # too small    20  400 
            inst_data[masks[i]] = 0  # set to background
            continue
        inst_pc_voxel = inst_pc_small.voxel_down_sample(voxel_size)
        if len(inst_pc_voxel.points) <= 10:  # too small    20  400 
            inst_data[masks[i]] = 0  # set to background
            continue
        is_merged = False
        inst_id = None
        inst_mask = masks[i] #smaller_mask #masks[i] 
        inst_class = classes[i]
        inst_depth = depth.detach().cpu().numpy()
        inst_depth[~masks[i].detach().cpu().numpy()] = 0.  # inst_mask
        inst_pc = unproject_pointcloud(inst_depth, intrinsic_open3d, T_CW)
        sem_inst_list = []
        if clip_features is not None: # check similar sems based on clip feature distance
            sem_thr = 200 #300. for table #320.  # 260.
            for sem_exist in sem_dict.keys():
                if torch.abs(clip_features[class_names[inst_class]] - clip_features[class_names[sem_exist]]).sum() < sem_thr:
                    sem_inst_list.extend(sem_dict[sem_exist])
        else:   # no clip features, only do strictly sem check
            if inst_class in sem_dict.keys():
                sem_inst_list.extend(sem_dict[inst_class])

        for candidate_inst in sem_inst_list:
    # if True:  # only consider 3D bbox, merge them if they are spatial together
            IoU, indices = check_inside_ratio(inst_pc, candidate_inst.bbox3D)
            candidate_inst.cmp_cnt += 1
            if IoU > IoU_thresh:
                # merge inst to candidate
                is_merged = True
                candidate_inst.merge_cnt += 1
                candidate_inst.pc += inst_pc.select_by_index(indices)
                # inst_uv = inst_pc.select_by_index(indices).project_to_depth_image(masks[i].shape[1], masks[i].shape[0], intrinsic_open3d, T_CW, depth_scale=1.0, depth_max=10.0)
                # # inst_uv = torch.utils.dlpack.from_dlpack(uv_opencv.as_tensor().to_dlpack())
                # valid_mask = inst_uv.squeeze() > 0.  # shape --> H, W
                # diff_mask = (inst_depth > 0.) & (~valid_mask)
                diff_mask = torch.zeros_like(inst_mask)
                uv_opencv, _ = cv2.projectPoints(np.array(inst_pc.select_by_index(indices).points), T_CW[:3, :3],
                                                 T_CW[:3, 3], intrinsic_open3d.intrinsic_matrix[:3,:3], None)
                uv = np.round(uv_opencv).squeeze().astype(int)
                u = uv[:, 0].reshape(-1, 1)
                v = uv[:, 1].reshape(-1, 1)
                vu = np.concatenate([v, u], axis=-1)
                valid_mask = np.zeros(inst_mask.shape, dtype=np.bool)
                valid_mask[tuple(vu.T)] = True
                diff_mask[(inst_depth!=0) & (~valid_mask)] = True
                # downsample pcs
                candidate_inst.pc = candidate_inst.pc.voxel_down_sample(voxel_size)
                # candidate_inst.pc.random_down_sample(np.minimum(500//len(candidate_inst.pc.points),1))
                candidate_inst.bbox3D = open3d.geometry.OrientedBoundingBox.create_from_points(candidate_inst.pc.points)
                # enlarge
                candidate_inst.bbox3D.scale(bbox3d_scale, candidate_inst.bbox3D.get_center())
                candidate_inst.bbox3D.extent = np.maximum(candidate_inst.bbox3D.extent, min_extent) # at least bigger than min_extent
                inst_id = candidate_inst.inst_id
                break
            # if candidate_inst.cmp_cnt >= 20 and candidate_inst.merge_cnt <= 5:
            #     sem_inst_list.remove(candidate_inst)

        if not is_merged:
            # init new inst and new sem
            new_inst = InstData()
            new_inst.inst_id = len(inst_list) + 1
            new_inst.class_id = inst_class

            new_inst.pc = inst_pc_small
            new_inst.pc = new_inst.pc.voxel_down_sample(voxel_size)
            inst_bbox3D = open3d.geometry.OrientedBoundingBox.create_from_points(new_inst.pc.points)
            # scale up
            inst_bbox3D.scale(bbox3d_scale, inst_bbox3D.get_center())
            inst_bbox3D.extent = np.maximum(inst_bbox3D.extent, min_extent)
            new_inst.bbox3D = inst_bbox3D
            inst_list.append(new_inst)
            inst_id = new_inst.inst_id
            # update sem_dict
            if inst_class in sem_dict.keys():
                sem_dict[inst_class].append(new_inst)   # append new inst to exist sem
            else:
                sem_dict.update({inst_class: [new_inst]})   # init new sem
        # update inst_data
        inst_data[inst_mask] = inst_id
        if diff_mask is not None:
            inst_data[diff_mask] = -1   # unsure area
        if inst_id not in inst_ids:
            inst_data_dict.update({inst_id: inst_data})
        else:
            continue
            # idx = inst_ids.index(inst_id)
            # inst_data_list[idx] = inst_data_list[idx] & torch.from_numpy(inst_data) # merge them? 
    # return inst_data
    mask_bg = torch.stack(list(inst_data_dict.values())).sum(0) != 0
    inst_data_dict.update({0: mask_bg.int()})
    return inst_data_dict
