import pickle
import sys
import argparse
import glob
import os
import yaml
import cv2
import numpy as np
from tqdm import tqdm, trange
import open3d as o3d
import copy
import time
import torch
from collections import Counter
from scipy.sparse import coo_matrix
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from networkx.algorithms.community import greedy_modularity_communities, girvan_newman
import community
from datetime import datetime
sys.path.append('/code/dyn/object_map/third_parites/')
from natsort import natsorted
from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer, util
import distinctipy
from statistics import mode
import glob
from scipy import ndimage

class MaskGraph:
    def __init__(self, geo_matrix, cap_matrix, clip_matrix, color_matrix, geo_matrix_2d, 
                 threshold_geo=0, threshold_cap=0, threshold_clip=0, threshold_color=0, threshold_geo_2d = 0,
                 method="threshold"):
        self.num_nodes = len(geo_matrix)
        self.nodes = [i for i in range(self.num_nodes)]
        self.adjacency_matrix = np.zeros_like(geo_matrix)
        self.weighted_matrix = np.zeros_like(geo_matrix)
        self.graph = nx.Graph()

        # 构建连接边
        if method == "threshold":
            adjacent_matrix = (geo_matrix > threshold_geo) * (cap_matrix > threshold_cap) * (clip_matrix > threshold_clip) * (color_matrix > threshold_color)
            self.adjacency_matrix[adjacent_matrix] = 1
        elif method == "weighted":
            self.weighted_matrix = (geo_matrix * threshold_geo) + (cap_matrix * threshold_cap) + (clip_matrix * threshold_clip) + (color_matrix * threshold_color) + (geo_matrix_2d * threshold_geo_2d)
            index_matrix = (self.weighted_matrix >= 1.0)
            self.adjacency_matrix[index_matrix] = 1

    def get_adjacency_matrix(self):
        # 获取邻接矩阵
        return self.adjacency_matrix
    
    def get_weighted_matrix(self):
        return self.weighted_matrix
    
    def get_edges(self):
        sparse_matrix = coo_matrix(self.adjacency_matrix)
        row_indices = sparse_matrix.row
        col_indices = sparse_matrix.col
        self.edges = set()
        for i in range(len(row_indices)):
            if row_indices[i] < col_indices[i]:
                # if i <= 5:
                #     print("non-weigthed mode")
                # self.edges.add((row_indices[i], col_indices[i]))
                # if i <= 5:
                #     print(f"weighted_matrix values equal {self.weighted_matrix[row_indices[i]][col_indices[i]]}.")
                if i <= 5:
                    print("weighted mode")
                self.edges.add((row_indices[i], col_indices[i], self.weighted_matrix[row_indices[i]][col_indices[i]]))
        # for i in range(self.adjacency_matrix.shape[0]):
        #     for j in range(i+1,self.adjacency_matrix.shape[0]):
        #         # if i <= 5:
        #         #     print("non-weigthed mode")
        #         # self.edges.add((row_indices[i], col_indices[i]))
        #         # if i <= 5:
        #         #     print(f"weighted_matrix values equal {self.weighted_matrix[row_indices[i]][col_indices[i]]}.")
        #         self.edges.add((i,j, self.weighted_matrix[i][j]))
        self.edges = list(self.edges)
    
    def gen_graph(self, vis=True):
        self.graph.add_nodes_from(self.nodes)
        # self.graph.add_edges_from(self.edges)
        self.graph.add_weighted_edges_from(self.edges)
        if vis:
            nx.draw(self.graph, with_labels=True, node_color='lightblue', node_size=800, font_size=10, font_weight='bold')
            plt.title('Intial Mask Cluster Graph')
            plt.show()
    
    def mask_cluster(self, vis=True, method="Louvain"):
        if method == "Louvain":
            partition = community.best_partition(self.graph, weight='weight')
            node_colors = [partition[node] for node in self.graph.nodes()]
            if vis:
                nx.draw(self.graph, with_labels=True, node_color=node_colors, cmap=plt.cm.rainbow)
                plt.title('Louvain Mask Cluster Graph')
                plt.show()
        if method == "Greedy":
            communities = greedy_modularity_communities(self.graph, weight='weight')
            color_map = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    color_map[node] = i
            node_colors = [color_map[node] for node in self.graph.nodes()]
            if vis:
                nx.draw(self.graph, with_labels=True, node_color=node_colors, cmap=plt.cm.rainbow)
                plt.title('Greedy Mask Cluster Graph')
                plt.show()
        elif method == "GNewman":
            communities = girvan_newman(self.graph)
            node_colors = {}
            for i, comm in enumerate(next(communities)):
                for node in comm:
                    node_colors[node] = i
            node_colors = [node_colors[node] for node in self.graph.nodes()]
            print(len(node_colors))
            if vis:
                nx.draw(self.graph, with_labels=True, node_color=node_colors, cmap=plt.cm.rainbow)
                plt.title('Graph with Girvan-Newman Clustering')
                plt.show()
        return node_colors



def stack_mask(mask_list):
    stack_list, stack_array = None, None
    if isinstance(mask_list[0], list):
        for mask in mask_list:       
            if stack_list is not None:
                stack_list += mask
            else:
                stack_list = mask
        return stack_list
    if isinstance(mask_list[0], np.ndarray):
        stack_array = np.concatenate(mask_list, axis=0)
        return stack_array


def filter_id(mask_ids, thre):
    # 过滤观测次数太少的那些，当做999
    count_dict = {x: mask_ids.count(x) for x in set(mask_ids)}
    inst_mask = [count_dict[x] <= thre for x in mask_ids]
    indexes = [i for i, m in enumerate(inst_mask) if m]
    for i in indexes:
        mask_ids[i] = 999
    return  mask_ids

def index_2_rgb(mask_ids, entity_colors, frame_count):
    mask_rgb = []
    usd_id = []
    # 实例着色时用000黑色mask掉那些观测比较少的物体小组件
    count_dict = {x: mask_ids.count(x) for x in set(mask_ids)}
    thre = frame_count/50
    inst_mask = [count_dict[x] <= thre for x in mask_ids]
    indexes = [i for i, m in enumerate(inst_mask) if m]
    for i in indexes:
        mask_ids[i] = 999
    for c in mask_ids:
        if c == 999:
            mask_rgb += [[0,0,0]]
        else:
            mask_rgb += [entity_colors[c]]
    return mask_rgb,  mask_colors


def cmap_2_rgb(mask_colors):
    cmap = cm.get_cmap('viridis')
    # 计算数组中每个元素对应的颜色
    mask_rgb = [list(cmap(x / max(mask_colors))[:3]) for x in mask_colors]
    # cmap = plt.cm.rainbow
    # mask_rgb = [list(cmap(color)[:3]) for color in mask_colors]
    for i in mask_rgb:
        i[0] = int(i[0]*255)
        i[1] = int(i[1]*255)
        i[2] = int(i[2]*255)
    return mask_rgb


def make_colors():
    # entityv2 所使用的entity分割颜色来自于COCO数据集
    from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
    colors = []
    colors.append([255,0,0])
    colors.append([0,255,0])
    colors.append([0,0,255])
    for cate in COCO_CATEGORIES:
        colors.append(cate["color"])
    return colors

def make_random_colors(num=200):
    # random_colors = distinctipy.get_colors(num, pastel_factor=1)  
    np.random.seed(42)
    random_colors = np.random.randint(0, 256, size=(10000, 3))
    random_colors[1] = [255,0,0]
    random_colors[2] = [0,255,0]
    random_colors[3] = [0,0,255]
    return random_colors

def get_parser():
    parser = argparse.ArgumentParser(description="mask_graph demo for mask_graph generation.")
    parser.add_argument(
        "--config_file",
        default="/code/dyn/object_map/maskcluster/configs/replica.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input_mask",
        nargs="+",
        help="the direction of input_mask from entity, TAP and CLIP "
    )
    parser.add_argument(
        "--input_depth",
        nargs="+",
        help="A list of space separated input depth images; "
    )
    parser.add_argument(
        "--input_rgb",
        nargs="+",
        help="A list of space separated input rgb images; "
    )
    parser.add_argument(
        "--input_pose",
        nargs="+",
        help="the direction of RGB-D pose from capture "
    )
    parser.add_argument(
        "--output_graph",
        help="A file or directory to save output mask_graph. "
    )
    parser.add_argument(
        "--output_dir",
        help="A file or directory to save output img. "
    )
    parser.add_argument(
        "--input_semantic",
        nargs="+",
        help="A file or directory of gt semantic. "
    )
    return parser


def pcd_denoise_dbscan(pcd: o3d.geometry.PointCloud, eps=0.02, min_points=10) -> o3d.geometry.PointCloud:
    ### Remove noise via clustering
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
        # if len(largest_cluster_points) < 5:
        #     mask = np.ones(obj_points.shape[0], dtype=bool)
        #     return pcd, mask
        largest_cluster_pcd = o3d.geometry.PointCloud()
        largest_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)
        # Create mask for the largest cluster
        mask = np.zeros(obj_points.shape[0], dtype=bool)
        mask[largest_mask] = True
        pcd = largest_cluster_pcd
        return pcd, mask
    # 对于小物体的噪声降低
    else:
        pcd_clusters = pcd.cluster_dbscan(
            eps=eps,
            min_points=int(min_points/5),
        )
        obj_points = np.asarray(pcd.points)
        pcd_clusters = np.array(pcd_clusters)
        counter = Counter(pcd_clusters)
        if counter and (-1 in counter):
            del counter[-1]
        if counter:
            # 只保留最大团
            most_common_label, _ = counter.most_common(1)[0]
            largest_mask = pcd_clusters == most_common_label
            largest_cluster_points = obj_points[largest_mask]
            largest_cluster_pcd = o3d.geometry.PointCloud()
            largest_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)
            # Create mask for the largest cluster
            mask = np.zeros(obj_points.shape[0], dtype=bool)
            mask[largest_mask] = True
            pcd = largest_cluster_pcd
            return pcd, mask
            # # 保留所有非噪声
            # largest_mask = pcd_clusters == most_common_label
        else:
            pcd_clusters = pcd.cluster_dbscan(
                eps=eps,
                min_points=int(min_points/10),
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
                largest_cluster_pcd = o3d.geometry.PointCloud()
                largest_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)
                # Create mask for the largest cluster
                mask = np.zeros(obj_points.shape[0], dtype=bool)
                mask[largest_mask] = True
                pcd = largest_cluster_pcd
                return pcd, mask
    mask = np.ones(obj_points.shape[0], dtype=bool)
    return pcd, mask

    
def extract_largest_connected_components(mask):
    '''
    对于多个联通区域的mask，得到所有mask区域
    '''
    # labeled_mask, num_labels = ndimage.label(mask)
    # largest_components = []
    # for label in range(1, num_labels + 1):
    #     component_mask = labeled_mask == label
    #     component_mask = torch.tensor(component_mask, device=torch.device('cuda'))
    #     largest_components.append(component_mask)
    num_labels, labeled_mask = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    components = []
    for label in range(1, num_labels):
        component_mask = labeled_mask == label
        component_mask = torch.tensor(component_mask, device=torch.device('cuda'))
        components.append(component_mask)
    return components

def project_mask_pc(mask_input, depth_input, rgb_input, pose_input, depth_scale, K_input, min_depth=0.07, max_depth=10, filter=True):
    mask_pc = []
    mask_bbox = []
    mask_color = []
    depth_array = cv2.imread(depth_input, -1)
    depth_array = torch.tensor(depth_array / depth_scale, dtype=torch.float32, device=torch.device('cuda'))
    H, W = depth_array.shape
    rgb_array = cv2.imread(rgb_input)
    if rgb_array.shape[0] != W:
        rgb_array = cv2.resize(rgb_array, (W, H), interpolation=cv2.INTER_NEAREST)
    if min_depth > 0:
        depth_array[depth_array < min_depth] = 0
    if max_depth > 0:
        depth_array[depth_array > max_depth] = 0
    fx, fy, cx, cy = K_input[0,0],K_input[1,1],K_input[0,2],K_input[1,2]
    # Convert to 3D coordinates
    mask_output = []
    not_non_mask = depth_array > 0
    # 用于判断mask是否可用
    mask_ok = torch.ones(len(mask_input), dtype=torch.bool, device=torch.device('cuda'))
    for i, mask in enumerate(mask_input):
        # 如果mask尺寸不对，resize
        if mask.shape[0] != W:
            mask_array_int = mask.astype(np.uint8) * 255
            mask = cv2.resize(mask_array_int, (W, H), interpolation=cv2.INTER_NEAREST)
            mask = (mask != 0)
        mask_raw = mask.copy()
        mask = torch.tensor(mask).to('cuda')
        # 排除掉无效点
        mask = mask & not_non_mask
        # 判断mask是否可用
        v, u = torch.nonzero(mask, as_tuple=True)
        # 如果当前mask内没有可用的，去掉这个mask
        if v.numel() == 0:
            mask_ok[i]=False
            continue
        # 原始的pcd
        depth_mask = depth_array[mask]
        x = (u - cx) * depth_mask / fx
        y = (v - cy) * depth_mask / fy
        z = depth_mask
        points = torch.stack((x, y, z), dim=-1)
        points = points.view(-1, 3)
        pcd_raw = o3d.geometry.PointCloud()
        pcd_raw.points = o3d.utility.Vector3dVector(points.cpu().numpy())
        pcd_raw = pcd_raw.transform(pose_input)
        # o3d.visualization.draw_geometries([pcd_raw])
        # 进行连通分量标记
        largest_components = extract_largest_connected_components(mask_raw)
        # print(len(largest_components))
        mask_new = mask.clone()
        pcd = o3d.geometry.PointCloud()
        if filter:
            mask_new_clone = mask_new.clone()
        for component_mask in largest_components:
            component_mask = component_mask & not_non_mask
            if component_mask[component_mask].shape[0] < 100:
                if filter:
                    mask_new_clone[component_mask] = False
                continue
                
            v, u = torch.nonzero(component_mask, as_tuple=True)
            depth_mask = depth_array[component_mask]
            x = (u - cx) * depth_mask / fx
            y = (v - cy) * depth_mask / fy
            z = depth_mask
            points = torch.stack((x, y, z), dim=-1)
            points = points.view(-1, 3)
            pcd_n = o3d.geometry.PointCloud()
            pcd_n.points = o3d.utility.Vector3dVector(points.cpu().numpy())
            pcd_n = pcd_n.transform(pose_input)
            # dbscan_remove_noise
            if not filter:
                pcd_n = pcd_n.voxel_down_sample(voxel_size=0.025)
                pcd_n, mask_filter_n = pcd_denoise_dbscan(
                    pcd_n, 
                    eps=0.05, 
                    min_points=10
                )
            else:
                # 如果不降采样的话，需要稍微调整一下参数
                pcd_n, mask_filter_n = pcd_denoise_dbscan(
                    pcd_n, 
                    eps=0.05, 
                    min_points=100
                )
            pcd += pcd_n
            if filter:
                mask_new_clone[component_mask] = torch.tensor(mask_filter_n.copy()).to("cuda")
        # o3d.visualization.draw_geometries([pcd])
        pc = np.asarray(pcd.points) 
        if pc.shape[0] < 10:
            mask_ok[i]=False
            continue
        if filter:
            mask_new = mask_new_clone
        mask_new = mask_new.cpu().numpy()
        mask_output.append(mask_new)
        # 将掩码图像转换为RGB格式
        # mask_vis = cv2.cvtColor(mask.astype(np.uint8)*100+mask_new.astype(np.uint8)*155, cv2.COLOR_GRAY2RGB)
        # # 创建一个窗口以显示图像
        # cv2.imshow('Mask Before Projection', mask_vis)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        bbox = np.array([pc[:, 0].min(), pc[:, 1].min(), pc[:, 2].min(),
                             pc[:, 0].max(), pc[:, 1].max(), pc[:, 2].max()])
        mask_pc += [pc]
        mask_bbox += [bbox]
        
        # 得到颜色直方图, 转换颜色空间
        # 设置每个通道的 bin 数量
        hist_channels = []
        for i in range(3):
            # 还用mask，保证参数稳定
            hist_channel = cv2.calcHist([rgb_array], [i], mask.cpu().numpy().astype(np.uint8), [32], [0, 256])
            # hist_channel = cv2.calcHist([rgb_array], [i], mask_new.astype(np.uint8), [32], [0, 256])
            hist_channels.append(hist_channel.flatten())
            # # 可视化直方图（可选）
            # plt.plot(hist_channels[0])
            # plt.title('Histogram')
            # plt.xlabel('Bins')
            # plt.ylabel('# of Pixels')
            # plt.show()
        hist = np.hstack(hist_channels)
        mask_color += [hist]
    return mask_pc, mask_bbox, mask_color, mask_output, mask_ok

def compute_iou(box1, box2):
    # 计算两个 3D bounding box 的交集部分  
    # TODO 3D bounding box 如果物体自己本身比较小的话 和大物体计算IOU的时候吃亏呢
    min_corner = np.maximum(box1[:3], box2[:3])
    max_corner = np.minimum(box1[3:], box2[3:])
    intersection = np.maximum(0, max_corner - min_corner).prod()
    
    # 计算两个 3D bounding box 的并集部分
    vol1 = (box1[3] - box1[0]) * (box1[4] - box1[1]) * (box1[5] - box1[2])
    vol2 = (box2[3] - box2[0]) * (box2[4] - box2[1]) * (box2[5] - box2[2])
    union = vol1 + vol2 - intersection
    
    # 计算 IOU
    iou = intersection / union if union > 0 else 0.0
    return iou

def adjacent_matrix_geo(mask_pc_input):
    num_masks = len(mask_pc_input)    
    adjmat_geo = np.eye(num_masks)
    start_time = time.time()
    for i in range(num_masks):
        for j in range(i + 1, num_masks):
            # 计算 bounding box
            mask1 = mask_pc_input[i]
            mask2 = mask_pc_input[j]
            box1 = np.array([mask1[:, 0].min(), mask1[:, 1].min(), mask1[:, 2].min(),
                             mask1[:, 0].max(), mask1[:, 1].max(), mask1[:, 2].max()])
            box2 = np.array([mask2[:, 0].min(), mask2[:, 1].min(), mask2[:, 2].min(),
                             mask2[:, 0].max(), mask2[:, 1].max(), mask2[:, 2].max()])            
            # 计算 IOU
            iou = compute_iou(box1, box2)
            adjmat_geo[i, j] = iou
            adjmat_geo[j, i] = iou  # 对称性
    end_time = time.time()
    # print(f"Mask Graph geometric adjacent matrixes computing for {end_time-start_time}s.")
    return adjmat_geo

def compute_3d_iou_matrix(bounding_boxes):
    '''
    计算3d bbox的IOU
    '''
    # 提取边界框坐标的各个维度
    x1 = bounding_boxes[:, 0]
    y1 = bounding_boxes[:, 1]
    z1 = bounding_boxes[:, 2]
    x2 = bounding_boxes[:, 3]
    y2 = bounding_boxes[:, 4]
    z2 = bounding_boxes[:, 5]
    # 计算边界框的体积
    volume = (x2 - x1) * (y2 - y1) * (z2 - z1)
    # 计算交集部分的体积
    min_x = np.maximum.outer(x1, x1)
    max_x = np.minimum.outer(x2, x2)
    min_y = np.maximum.outer(y1, y1)
    max_y = np.minimum.outer(y2, y2)
    min_z = np.maximum.outer(z1, z1)
    max_z = np.minimum.outer(z2, z2)
    intersection_volume = np.maximum(0, max_x - min_x) * np.maximum(0, max_y - min_y) * np.maximum(0, max_z - min_z)
    # 计算并集部分的体积
    union_volume = np.expand_dims(volume, axis=1) + np.expand_dims(volume, axis=0) - intersection_volume
    # 计算较小区域的体积
    smaller_volume = np.minimum.outer(volume, volume)
    iou_matrix = intersection_volume / smaller_volume
    # # 计算 IoU
    # iou_matrix = intersection_volume / union_volume
    iou_matrix[np.isnan(iou_matrix)] = 0  # 处理 NaN 值（当两个边界框相交面积为 0 时）
    return iou_matrix


def compute_iou_2d(bounding_boxes):
    """
    计算两个二维边界框之间的 IoU 矩阵
    :param bounding_boxes: 形状为 (N, 4) 的边界框张量，其中 N 表示边界框的数量，
                           每个边界框由 [x_min, y_min, x_max, y_max] 表示左上角和右下角坐标
    :return: 形状为 (N, N) 的 IoU 矩阵，其中 IoU[i, j] 表示第 i 个边界框与第 j 个边界框之间的 IoU
    """
    # 提取边界框坐标的各个维度
    x1 = bounding_boxes[:, 0]
    y1 = bounding_boxes[:, 1]
    x2 = bounding_boxes[:, 2]
    y2 = bounding_boxes[:, 3]
    # 计算边界框的面积
    area = (x2 - x1) * (y2 - y1)
    # 计算交集部分的面积
    min_x = torch.max(x1.unsqueeze(1), x1.unsqueeze(0))
    max_x = torch.min(x2.unsqueeze(1), x2.unsqueeze(0))
    min_y = torch.max(y1.unsqueeze(1), y1.unsqueeze(0))
    max_y = torch.min(y2.unsqueeze(1), y2.unsqueeze(0))
    intersection_area = torch.clamp(max_x - min_x, min=0) * torch.clamp(max_y - min_y, min=0)
    # 计算并集部分的面积
    union_area = area.unsqueeze(1) + area.unsqueeze(0) - intersection_area
    # 计算 IoU
    iou_matrix = intersection_area / union_area
    iou_matrix[iou_matrix.isnan()] = 0  # 处理 NaN 值（当两个边界框相交面积为 0 时）
    return iou_matrix


# 用于计算2D投影IoU矩阵的系列函数
def get_rays(w=None, h=None, K=None):
    ix, iy = torch.meshgrid(
        torch.arange(0,w,10), torch.arange(0,h,10), indexing='xy')
    # ix, iy = torch.meshgrid(
    #     torch.arange(0,w), torch.arange(0,h), indexing='xy')
    rays_d = torch.stack(
                [(ix-K[0, 2]) / K[0,0],
                (iy-K[1,2]) / K[1,1],
                torch.ones_like(ix)], -1).float()
    return rays_d

def adjacent_matrix_feat(mask_capft):    
    mask_capft_tensor = torch.tensor(mask_capft, dtype=torch.float32)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        mask_capft_tensor = mask_capft_tensor.to(device)
    start_time = time.time()
    adjmat_cap = torch.mm(mask_capft_tensor, mask_capft_tensor.t()) / (
    torch.norm(mask_capft_tensor, dim=1)[:, None] * torch.norm(mask_capft_tensor, dim=1)[:, None].t())
    end_time = time.time()
    # print(f"Mask Graph captionfeats adjacent matrixes computing for {end_time-start_time}s.")
    adjmat_cap = adjmat_cap.cpu().numpy()
    return adjmat_cap

def cross_entropy_loss(p, q):
    """
    计算两个概率分布之间的交叉熵损失
    """
    return -np.sum(p * np.log(q), axis=1)

def compute_color_matrix(matrix):
    '''
    计算color相似性
    '''
    matrix_normalized = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    # 计算特征矩阵的转置
    features_transpose = matrix_normalized.T
    # 计算内积
    dot_product = np.dot(matrix_normalized, features_transpose)
    return dot_product


from sklearn.cluster import DBSCAN
def get_majority_cluster_mean(vectors, eps = 0.2, min_samples = 2):
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


# def ray_box_intersection(origins, directions, bounds_min, bounds_max):
#     tmin = (bounds_min - origins) / directions
#     tmax = (bounds_max - origins) / directions
#     t1 = torch.min(tmin, tmax)
#     t2 = torch.max(tmin, tmax)
#     near = torch.amax(t1, dim=1)
#     far = torch.amin(t2, dim=1)
#     hit = near <= far
#     front = far > 0
#     hit = hit & front
#     return near, far, hit, hit.nonzero().squeeze(1)

def ray_box_intersection(origins, directions, bounds_min, bounds_max):
    tmin = (bounds_min - origins.unsqueeze(1)) / directions.unsqueeze(1)
    tmax = (bounds_max - origins.unsqueeze(1)) / directions.unsqueeze(1)
    t1 = torch.min(tmin, tmax)
    t2 = torch.max(tmin, tmax)
    near = torch.amax(t1, dim=2)
    far = torch.amin(t2, dim=2)
    hit = (near <= far) & (far > 0)
    hit = hit.T
    return near, far, hit

# def min_rect_bbox(mask):
#     '''
#     寻找非零像素点的坐标，得到最小bbox
#     '''
#     # 获取非零像素点的坐标
#     nonzero_indices = torch.nonzero(mask)
#     if len(nonzero_indices) == 0:
#         # 如果 mask 中没有非零元素，返回空的矩形
#         return torch.zeros((4, 2), dtype=torch.int32)
#     # 计算轴对齐矩形
#     x, y, w, h = cv2.boundingRect(torch.stack([nonzero_indices[:, 1], nonzero_indices[:, 0]], dim=1).cpu().numpy())
#     # 根据左上角点和宽高计算矩形的四个顶点
#     rect = torch.tensor([[y, x], [y, x + w], [y + h, x + w], [y + h, x]], dtype=torch.int32)
#     return rect

def min_rect_bbox(masks):
    '''
    寻找非零像素点的坐标，得到最小bbox
    '''
    bboxes = []
    for mask in masks:
        nonzero_indices = torch.nonzero(mask)
        if len(nonzero_indices) == 0:
            # 如果 mask 中没有非零元素，返回空的矩形
            # bboxes.append(torch.zeros((4, 2), dtype=torch.int32))
            bboxes.append(torch.zeros((4,), dtype=torch.int32))
        else:
            # 计算轴对齐矩形
            x, y, w, h = cv2.boundingRect(torch.stack([nonzero_indices[:, 1], nonzero_indices[:, 0]], dim=1).cpu().numpy())
            # 根据左上角点和宽高计算矩形的四个顶点
            rect = torch.tensor([[y, x], [y, x + w], [y + h, x + w], [y + h, x]], dtype=torch.int32)
            rect = torch.tensor([rect[0,0],rect[0,1],rect[2,0],rect[2,1]])
            bboxes.append(rect)
    # 将列表中的张量堆叠起来
    stacked_bboxes = torch.stack(bboxes)
    return stacked_bboxes


def compute_2d_iou_matrix(cfg, frame_count, input_depth, Twc, K_input, mask_bbox_stack):
    # 计算2D投影IoU矩阵的代码段
    print(f"mask_bbox_stack.shape for 2D projection IoU: {mask_bbox_stack.shape}.")
    mask_bbox_tensor = torch.tensor(mask_bbox_stack)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        mask_bbox_tensor = mask_bbox_tensor.to(device)
    Twc = torch.tensor(Twc)
    image_W = cfg["image_W"]
    image_H = cfg["image_H"]
    skip = cfg["skip"]
    rays_d = get_rays(w=image_W, h=image_H, K=K_input)
    print(f"check mask length before 2D iou computing...{frame_count}")
    iou_2d_matrix = torch.zeros((mask_bbox_tensor.shape[0], mask_bbox_tensor.shape[0])).to("cuda")
    for i in trange(frame_count):
        depth_i = cv2.imread(input_depth[i], -1) / 1000.0
        rays_dw = rays_d * depth_i[::10, ::10, None]
        rays_dw = rays_dw.reshape(-1,3)
        rays_dw = rays_dw @ Twc[i][:3,:3].transpose(-1,-2)
        rays_ow = Twc[i][:3, 3].unsqueeze(0).expand_as(rays_dw)
        if torch.cuda.is_available():
            device = torch.device("cuda")
            rays_ow = rays_ow.to(device)
            rays_dw = rays_dw.to(device)
        
        near, far, hit = ray_box_intersection(rays_ow, rays_dw, mask_bbox_tensor[:, :3], mask_bbox_tensor[:, 3:])
        hit_mask = hit.view(-1, int(depth_i.shape[0]/10), int(depth_i.shape[1]/10))
        this_2d_bbox = min_rect_bbox(hit_mask).to("cuda")
        iou_2d_i = compute_iou_2d(this_2d_bbox)
        iou_2d_matrix = (iou_2d_matrix * i + iou_2d_i) / (i + 1)
        
        # start = time.time()
        # # hit_mask_i = []
        # hit_bbox_i = []
        # print("0")
        # for j in range(mask_bbox_tensor.shape[0]):
        #     _, _, hit, _ = ray_box_intersection(rays_ow, rays_dw, mask_bbox_tensor[j,:3], mask_bbox_tensor[j,3:])
        #     hit_mask = hit.view(rays_d.shape[0], rays_d.shape[1])
        #     # if hit.sum().item() <= 0:
        #     #     continue
        #     # hit_mask = hit_mask.cpu().numpy()
        #     this_2d_bbox = min_rect_bbox(hit_mask)
        #     # hit_bbox_i += [[this_2d_bbox[0,0],this_2d_bbox[0,1],this_2d_bbox[2,0],this_2d_bbox[2,1]]]
        #     hit_bbox_i += [torch.tensor([this_2d_bbox[0,0],this_2d_bbox[0,1],this_2d_bbox[2,0],this_2d_bbox[2,1]])]
        #     # hit_mask_i += [hit_mask]
        # print("1")
        # all_bbox = torch.stack(hit_bbox_i).to("cuda")
        # print(all_bbox.shape)
        # iou_2d_i = compute_iou_2d(all_bbox)
        # iou_2d_matrix = (iou_2d_matrix*i+iou_2d_i)/(i+1)
        # print("2")
        
    return iou_2d_matrix.cpu().numpy()

def check_similarity(arr_n384, arr_384, threshold=0.8):
    '''
    检查相似性是否超过阈值
    '''
    for vec_n384 in arr_n384:
        similarity = np.dot(vec_n384, arr_384)
        if similarity > threshold:
            return True
    return False


def compute_similarity_matrix(global_pc, global_pc_capft, wall_fts, floor_fts, ceiling_fts, cap_thre=0.8, weight_pc=0.8, weight_caption=0.2, dis_thre=0.02, ratio_thre=0.5):
    '''
    后处理，得到相似性矩阵
    按照overlap和caption的加权相似性得到
    '''
    num_point_clouds = len(global_pc)
    similarity_pc = np.zeros((num_point_clouds, num_point_clouds))
    similarity_caption = np.zeros((num_point_clouds, num_point_clouds))
    # 提取字典中的点云对象
    keys = list(global_pc.keys())
    point_clouds = list(global_pc.values())
    point_caption = list(global_pc_capft.values())
    for i in range(num_point_clouds):
        for j in range(i + 1, num_point_clouds):
            pc1 = point_clouds[i]
            pc2 = point_clouds[j]
            # 计算点云之间的距离
            distances_1_to_2 = np.asarray(pc1.compute_point_cloud_distance(pc2))
            distances_2_to_1 = np.asarray(pc2.compute_point_cloud_distance(pc1))
            # 计算距离小于阈值的点所占的比例
            ratio_1_to_2 = np.mean(distances_1_to_2 < dis_thre)
            ratio_2_to_1 = np.mean(distances_2_to_1 < dis_thre)
            similarity_pc[i, j] = max(ratio_1_to_2, ratio_2_to_1)
            similarity_pc[j, i] = similarity_pc[i, j]
            similarity_caption[i,j] = np.dot(point_caption[i],point_caption[j])
            similarity_caption[j, i] = similarity_caption[i, j]
    similarity_matrix = similarity_pc*weight_pc + similarity_caption*weight_caption
    # 根据相似性，得到映射 
    mapping = {}
    mapping_counter = 4 # 从1开始
    for i in range(num_point_clouds):
        for j in range(i + 1, num_point_clouds):
            # 对于是三种背景的东西，不可让其他的东西融在一起，对于i或j是背景，不做判断，直接映射
            # 判断当前是否超过阈值，超过阈值直接写即可
            if check_similarity(wall_fts, point_caption[i], threshold=cap_thre):
                mapping[keys[i]] = 1
                continue
            elif check_similarity(floor_fts, point_caption[i], threshold=cap_thre):
                mapping[keys[i]] = 2
                continue
            elif check_similarity(ceiling_fts, point_caption[i], threshold=cap_thre):
                mapping[keys[i]] = 3
                continue
            if check_similarity(wall_fts, point_caption[j], threshold=cap_thre):
                mapping[keys[j]] = 1
                continue
            elif check_similarity(floor_fts, point_caption[j], threshold=cap_thre):
                mapping[keys[j]] = 2
                continue
            elif check_similarity(ceiling_fts, point_caption[j], threshold=cap_thre):
                mapping[keys[j]] = 3
                continue
            # 如果i和j都不是背景，再去判断
            if similarity_matrix[i, j] > ratio_thre:
                # 如果相似性大于阈值，则将 i 和 j 映射到同一个键
                if keys[i] not in mapping:
                    mapping[keys[i]] = mapping_counter
                    mapping_counter += 1
                # 现在只知道i不是背景，如果j是北京的话，也不要加进来，等到了它再说
                if keys[j] not in mapping :
                    mapping[keys[j]] = mapping[keys[i]]
    # 找到所有未被映射的点云索引
    unmapped = [i for i in range(num_point_clouds) if keys[i] not in mapping]
    # 将未被映射的点云添加到映射字典中
    for idx in unmapped:
        mapping[keys[idx]] = mapping_counter
        mapping_counter += 1
    # 把999映射到0
    mapping[999] = 0
    return similarity_matrix, mapping, mapping_counter


def compute_similarity_matrix_thre(global_pc, global_pc_capft, global_pc_color, wall_fts, floor_fts, ceiling_fts, 
                                   cap_thre=0.8, dis_thre=0.02,  weight_pc=0.7, weightcaption=0.7, weightcolor=0.7):
    '''
    后处理，得到相似性矩阵，这个是按照阈值方式，防止过融合
    
    '''
    num_point_clouds = len(global_pc)
    similarity_pc = np.zeros((num_point_clouds, num_point_clouds))
    similarity_caption = np.zeros((num_point_clouds, num_point_clouds))
    similarity_color = np.zeros((num_point_clouds, num_point_clouds))
    # 提取字典中的点云对象
    keys = list(global_pc.keys())
    point_clouds = list(global_pc.values())
    point_caption = list(global_pc_capft.values())
    point_color = list(global_pc_color.values())
    for i in range(num_point_clouds):
        for j in range(i + 1, num_point_clouds):
            pc1 = point_clouds[i]
            pc2 = point_clouds[j]
            # 计算点云之间的距离
            distances_1_to_2 = np.asarray(pc1.compute_point_cloud_distance(pc2))
            distances_2_to_1 = np.asarray(pc2.compute_point_cloud_distance(pc1))
            # 计算距离小于阈值的点所占的比例
            ratio_1_to_2 = np.mean(distances_1_to_2 < dis_thre)
            ratio_2_to_1 = np.mean(distances_2_to_1 < dis_thre)
            similarity_pc[i, j] = max(ratio_1_to_2, ratio_2_to_1)
            similarity_pc[j, i] = similarity_pc[i, j]
            similarity_caption[i,j] = np.dot(point_caption[i],point_caption[j])
            similarity_caption[j, i] = similarity_caption[i, j]
            similarity_color[i,j] = np.dot(point_color[i],point_color[j])
            similarity_color[j, i] = similarity_color[i, j]
    # 满足三阈值，或者点云完全重叠
    similarity_matrix = (similarity_pc>weight_pc) & (similarity_caption>weightcaption) & (similarity_color>weightcolor) | (similarity_pc>0.9) 
    # 根据相似性，得到映射 
    mapping = {}
    mapping_counter = 4 # 从1开始
    for i in range(num_point_clouds):
        for j in range(i + 1, num_point_clouds):
            # 对于是三种背景的东西，不可让其他的东西融在一起，对于i或j是背景，不做判断，直接映射
            # 判断当前是否超过阈值，超过阈值直接写即可
            if check_similarity(wall_fts, point_caption[i], threshold=cap_thre):
                mapping[keys[i]] = 1
                continue
            elif check_similarity(floor_fts, point_caption[i], threshold=cap_thre):
                mapping[keys[i]] = 2
                continue
            elif check_similarity(ceiling_fts, point_caption[i], threshold=cap_thre):
                mapping[keys[i]] = 3
                continue
            if check_similarity(wall_fts, point_caption[j], threshold=cap_thre):
                mapping[keys[j]] = 1
                continue
            elif check_similarity(floor_fts, point_caption[j], threshold=cap_thre):
                mapping[keys[j]] = 2
                continue
            elif check_similarity(ceiling_fts, point_caption[j], threshold=cap_thre):
                mapping[keys[j]] = 3
                continue
            # 如果i和j都不是背景，再去判断
            if similarity_matrix[i, j] == True:
                # 如果相似性大于阈值，则将 i 和 j 映射到同一个键
                if keys[i] not in mapping:
                    mapping[keys[i]] = mapping_counter
                    mapping_counter += 1
                # 现在只知道i不是背景，如果j是北京的话，也不要加进来，等到了它再说
                if keys[j] not in mapping :
                    mapping[keys[j]] = mapping[keys[i]]
    # 找到所有未被映射的点云索引
    unmapped = [i for i in range(num_point_clouds) if keys[i] not in mapping]
    # 将未被映射的点云添加到映射字典中
    for idx in unmapped:
        mapping[keys[idx]] = mapping_counter
        mapping_counter += 1
    # 把999映射到0
    mapping[999] = 0
    return similarity_matrix, mapping, mapping_counter

if __name__=="__main__":
    args = get_parser().parse_args()
    # 从yaml文件中读取config配置文件 为多层字典结构
    with open(args.config_file, 'r') as f:
        cfg = yaml.safe_load(f)
        
    # 一些预定义的caption，用来判断是不是墙面等
    sbert_model = SentenceTransformer('/home/dyn/multimodal/SBERT/pretrained/model/all-MiniLM-L6-v2')
    # room1
    captions_wall = cfg["captions_wall"]
    captions_floor = cfg["captions_floor"]
    captions_ceiling = cfg["captions_ceiling"]
    
    wall_fts = sbert_model.encode(captions_wall, convert_to_tensor=True, device="cuda")
    wall_fts = wall_fts / wall_fts.norm(dim=-1, keepdim=True)
    wall_fts = wall_fts.detach().cpu().numpy()
    floor_fts = sbert_model.encode(captions_floor, convert_to_tensor=True, device="cuda")
    floor_fts = floor_fts / floor_fts.norm(dim=-1, keepdim=True)
    floor_fts = floor_fts.detach().cpu().numpy()
    ceiling_fts = sbert_model.encode(captions_ceiling, convert_to_tensor=True, device="cuda")
    ceiling_fts = ceiling_fts / ceiling_fts.norm(dim=-1, keepdim=True)
    ceiling_fts = ceiling_fts.detach().cpu().numpy()
    
    skip = cfg["skip"]
    # debug用
    use_num = cfg["use_num"]
    start = cfg["start"]
    # 从mask.pkl中读取各个mask的图像、caption、captionft、clipft
    print("read from", args.input_mask[0])
    mask_info = pickle.load(open(args.input_mask[0], 'rb'))
    if use_num != -1:
        mask = mask_info["mask"][start:start+use_num]
        mask_cap = mask_info["caption"][start:start+use_num]
        # print(mask_cap)
        mask_capft = mask_info["capfeat"][start:start+use_num]
        mask_clift = mask_info["clipfeat"][start:start+use_num]
    else:
        mask = mask_info["mask"]
        mask_cap = mask_info["caption"]
        # print(mask_cap)
        mask_capft = mask_info["capfeat"]
        mask_clift = mask_info["clipfeat"]
        
    # # 遍历每个文本列表，将其保存为txt文件
    # for i, cap_list in enumerate(mask_cap):
    #     txt_file_path = os.path.join("/data/dyn/results/object/results/vMAP/614room3/caption", f"caption_{i}.txt")
    #     with open(txt_file_path, "w", encoding="utf-8") as txt_file:
    #         for cap in cap_list:
    #             txt_file.write(cap + "\n")
    # print("ok")
                
    # 将mask stack成一个连续的mask 但是这样的话每个mask和帧id的对应关系应该也得存储一下方便使用
    
    # 从目标路径读取各帧所对应的depth图像以及对应的pose以供mask投影到3D space
    args.input_depth = natsorted(args.input_depth)
    args.input_depth = args.input_depth[0:-1:skip]
    args.input_rgb = natsorted(args.input_rgb)
    args.input_rgb = args.input_rgb[0:-1:skip]
    Twc = np.loadtxt(args.input_pose[0], delimiter=" ").reshape([-1, 4, 4])
    # 是否区分前景背景，还是直接使用真值
    if_bg = cfg["if_bg"]
    if if_bg:
        # args.input_semantic = glob.glob(args.input_semantic)
        args.input_semantic = natsorted(args.input_semantic)
        args.input_semantic = args.input_semantic[0:-1:skip]
    # 计算旋转矩阵（绕 X、Y、Z 轴旋转指定角度）
    theta_x = np.radians(cfg["x_the"])  # 绕 X 轴旋转角度（弧度）
    theta_y = np.radians(cfg["y_the"])  # 绕 Y 轴旋转角度（弧度）
    theta_z = np.radians(cfg["z_the"])  # 绕 Z 轴旋转角度（弧度）
    # 计算绕 X 轴的旋转矩阵
    R_x = np.array([
        [1, 0, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x), 0],
        [0, np.sin(theta_x), np.cos(theta_x), 0],
        [0, 0, 0, 1]
    ])
    # 计算绕 Y 轴的旋转矩阵
    R_y = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y), 0],
        [0, 1, 0, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y), 0],
        [0, 0, 0, 1]
    ])
    # 计算绕 Z 轴的旋转矩阵
    R_z = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0, 0],
        [np.sin(theta_z), np.cos(theta_z), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    # 构造总的旋转矩阵
    R_total = np.dot(np.dot(R_x, R_y), R_z)
    # 对每个位姿矩阵进行旋转
    for i in range(len(Twc)):
        # Twc[i] = np.dot(Twc[i], R_total.T)
        Twc[i] = np.dot(R_total.T, Twc[i])
        
    if use_num != -1:
        Twc = Twc[0:-1:skip,:,:][start:start+use_num,:,:]
    else:
        Twc = Twc[0:-1:skip,:,:]
    K_input = np.eye(3)
    K_input[0, 0] = cfg["fx"]
    K_input[1, 1] = cfg["fy"]
    K_input[0, 2] = cfg["cx"]
    K_input[1, 2] = cfg["cy"]
    depth_scale = cfg["depth_scale"]
    # 为每个mask投影得到对应的空间点云  因为是replica  所以每一个位置都有有效深度  所以只要有mask应该就会有3D点云
    print("Projecting mask onto pointcloud...")
    mask_all_pc = []
    mask_all_bbox = []
    mask_all_color = []
    mask_filter = [] 
    for i in trange(len(mask)):
        # mask也按照投影后的过滤一下得到新的mask
        filter = bool(cfg["if_filter"])
        mask_pc, mask_bbox, mask_color, mask_output, mask_ok = project_mask_pc(mask[i], args.input_depth[i], args.input_rgb[i], Twc[i], depth_scale, K_input, filter = filter)
        # 去掉那些无效的mask
        mask_cap[i] = [value for value, keep in zip(mask_cap[i], mask_ok) if keep]
        mask_capft[i] = [value for value, keep in zip(mask_capft[i], mask_ok) if keep]
        mask_clift[i] = [value for value, keep in zip(mask_clift[i], mask_ok) if keep]
        
        mask_all_pc += [mask_pc]
        mask_all_bbox += [mask_bbox]
        mask_all_color += [mask_color]
        mask_filter += [mask_output]
    
    mask_cap_stack = stack_mask(copy.deepcopy(mask_cap))
    mask_capft_stack = stack_mask(copy.deepcopy(mask_capft))
    mask_clift_stack = np.concatenate(stack_mask(copy.deepcopy(mask_clift)), axis=0)
    print(f"Total number of Mask Nodes is {len(mask_cap_stack)}.")
        
    # # 可视化投影点云判断2D-to-3D投影变换是否自然
    # for i in trange(len(mask_all_pc)):
    #     mask_all_pcd = []
    #     pcd_i=o3d.geometry.PointCloud()
    #     points_i = np.concatenate(mask_all_pc[i], axis=0)
    #     points_i=o3d.utility.Vector3dVector(points_i)
    #     pcd_i.points = points_i
    #     mask_all_pcd += [pcd_i]
    #     for bbox in mask_all_bbox[i]:
    #         bbox_pc = o3d.geometry.AxisAlignedBoundingBox(min_bound=bbox[:3], max_bound=bbox[3:])
    #         mask_all_pcd += [bbox_pc]
    #     o3d.visualization.draw_geometries(mask_all_pcd)
        
    # mask_pc_stack = stack_mask(copy.deepcopy(mask_all_pc))
    mask_bbox_stack = np.stack(stack_mask(copy.deepcopy(mask_all_bbox))) # (mask_num,6)
    mask_color_stack = np.stack(stack_mask(copy.deepcopy(mask_all_color))) # (mask_num,3x32)
    print("Projected pointcloud OK!")
    print("Mask Graph adjacent matrixes computing ....")
    geo_matrix = compute_3d_iou_matrix(copy.deepcopy(mask_bbox_stack))
    cap_matrix = adjacent_matrix_feat(copy.deepcopy(mask_capft_stack))
    clip_matrix = adjacent_matrix_feat(copy.deepcopy(mask_clift_stack))
    frame_count = len(mask)
    geo_2_matrix = np.zeros((mask_bbox_stack.shape[0], mask_bbox_stack.shape[0]))
    if cfg["weight_geo_2d"] > 0:
        geo_2_matrix = compute_2d_iou_matrix(cfg, frame_count, args.input_depth, copy.deepcopy(Twc), K_input, mask_bbox_stack=copy.deepcopy(mask_bbox_stack))
    color_matrix = compute_color_matrix(copy.deepcopy(mask_color_stack))
    
    print("Adjacent matrix of geo/cap/clip generated OK!")
    # for debug 获得三个mask融合的经验阈值
    if cfg["graph_method"] == "threshold":
        thres_geo = cfg["threshold_geo"]
        thres_cap = cfg["threshold_cap"]
        thres_clip = cfg["threshold_clip"]
        thres_color = cfg["threshold_color"]
        print(f"geo_matrix: {(np.sum(geo_matrix>thres_geo)-geo_matrix.shape[0])/2}")    
        print(f"cap_matrix: {(np.sum(cap_matrix>thres_cap)-cap_matrix.shape[0])/2}")
        print(f"clip_matrix: {(np.sum(clip_matrix>thres_clip)-clip_matrix.shape[0])/2}")
        print(f"color_matrix: {(np.sum(color_matrix>thres_color)-color_matrix.shape[0])/2}")
        print(f"all_matrix: {(np.sum((geo_matrix>thres_geo)*(cap_matrix>thres_cap)*(clip_matrix>thres_clip))-clip_matrix.shape[0])/2}")
    # 采用weighted加权融合阈值构建连接边的方式
    elif cfg["graph_method"] == "weighted":
        thres_geo = cfg["weight_geo"]
        thres_cap = cfg["weight_cap"]
        thres_clip = cfg["weight_clip"]
        thres_color = cfg["weight_color"]
        thres_geo_2d = cfg["weight_geo_2d"]
    else:
        print("unexpected graph generation method!")
    # print(geo_matrix.shape)
    # print(cap_matrix.shape)
    # print(clip_matrix.shape)
    # print(color_matrix.shape)
    # print(geo_2_matrix.shape)
    # 根据三重相似性矩阵和mask生成方法得到mask_graph
    maskgraph = MaskGraph(geo_matrix, cap_matrix, clip_matrix, color_matrix, geo_2_matrix, thres_geo, thres_cap, thres_clip, thres_color, thres_geo_2d, method=cfg["graph_method"])
    maskgraph.get_edges()
    maskgraph.gen_graph(vis=False)
    # cluster_colors = maskgraph.mask_cluster(method="Greedy",vis=False)
    cluster_ids = maskgraph.mask_cluster(method="Louvain",vis=False)
    
    # random_colors = make_colors()
    random_colors = make_random_colors(num=400)

    # 过滤一些观测次数太少的，太少的为999
    cluster_ids = filter_id(cluster_ids, int(frame_count/50))
    
    # 得到嵌套列表的id
    cluster_mask_id = []
    curr_idx = 0
    for m in mask_filter:
        cluster_mask_id += [cluster_ids[curr_idx: curr_idx+len(m)]]
        curr_idx += len(m)
    
    print("The object num before is:", len(set(cluster_ids)))
    
    
    # 后处理，聚合一些比较近的物体
    global_pc = {}
    global_pc_color = {}
    global_pc_capft = {}
    global_pc_color = {}
    threshold = cfg["cap_thre"]
    for i in range(len(mask_filter)):
        if if_bg:
            gt_img = cv2.imread(args.input_semantic[i], cv2.IMREAD_UNCHANGED).astype(np.int32)
        for j in range(len(mask_filter[i])):
            # 看一些地板有哪些caption
            # if check_similarity(floor_fts, mask_capft[i][j], threshold=0.5):
            #     print(mask_cap[i][j])
            # 太小的也不要了
            # if np.count_nonzero(mask[i][j]) > 50 and cluster_mask_id[i][j]!=999:
            if  cluster_mask_id[i][j]!=999:
                this_id = cluster_mask_id[i][j]
                # 加入现有全局点云
                if this_id in global_pc.keys():
                    global_pc[this_id] = np.vstack((global_pc[this_id],mask_all_pc[i][j]))
                    if if_bg:
                        # 判断一些这个mask里面的真值id是否是背景语义
                        gt_id = mode(gt_img[mask[i][j]])
                        if gt_id == cfg["gt_wall_id"]:
                            global_pc_capft[this_id] = np.vstack((global_pc_capft[this_id],wall_fts[0]))
                        elif gt_id == cfg["gt_floor_id"]:
                            global_pc_capft[this_id] = np.vstack((global_pc_capft[this_id],floor_fts[0]))
                        elif gt_id == cfg["gt_ceiling_id"]:
                            global_pc_capft[this_id] = np.vstack((global_pc_capft[this_id],ceiling_fts[0]))
                        else:
                            global_pc_capft[this_id] = np.vstack((global_pc_capft[this_id],mask_capft[i][j]))
                    else:
                        # global_pc_capft[this_id] = np.vstack((global_pc_capft[this_id],mask_capft[i][j]))
                        # 对于墙面，caption要调整一下，因为wall很难被识别，如果来回融合就乱了
                        if check_similarity(wall_fts, mask_capft[i][j], threshold=threshold):
                            global_pc_capft[this_id] = np.vstack((global_pc_capft[this_id],wall_fts[0]))
                        elif check_similarity(floor_fts, mask_capft[i][j], threshold=threshold):
                            global_pc_capft[this_id] = np.vstack((global_pc_capft[this_id],floor_fts[0]))
                        elif check_similarity(ceiling_fts, mask_capft[i][j], threshold=threshold):
                            global_pc_capft[this_id] = np.vstack((global_pc_capft[this_id],ceiling_fts[0]))
                        else:
                            global_pc_capft[this_id] = np.vstack((global_pc_capft[this_id],mask_capft[i][j]))
                    global_pc_color[this_id] = np.vstack((global_pc_color[this_id],mask_all_color[i][j]))
                else:
                    global_pc[this_id] = mask_all_pc[i][j]
                    if if_bg:
                        # 判断一些这个mask里面的真值id是否是背景语义
                        gt_id = mode(gt_img[mask[i][j]])
                        if gt_id == cfg["gt_wall_id"]:
                            global_pc_capft[this_id] = wall_fts[0]
                        elif gt_id == cfg["gt_floor_id"]:
                            global_pc_capft[this_id] = floor_fts[0]
                        elif gt_id == cfg["gt_ceiling_id"]:
                            global_pc_capft[this_id] = ceiling_fts[0]
                        else:
                            global_pc_capft[this_id] = mask_capft[i][j]
                    else:
                        # 对于墙面，caption要调整一下
                        if check_similarity(wall_fts, mask_capft[i][j], threshold=threshold):
                            global_pc_capft[this_id] = wall_fts[0]
                        elif check_similarity(floor_fts, mask_capft[i][j], threshold=threshold):
                            global_pc_capft[this_id] = floor_fts[0]
                        elif check_similarity(ceiling_fts, mask_capft[i][j], threshold=threshold):
                            global_pc_capft[this_id] = ceiling_fts[0]
                        else:
                            global_pc_capft[this_id] = mask_capft[i][j]
                    global_pc_color[this_id] = mask_all_color[i][j]
            else:
                cluster_mask_id[i][j]=999
                
    for global_id, pc in global_pc.items():
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=0.01)
        global_pc[global_id] = downsampled_pcd
        # 不是去均值，而是看最多
        if np.ndim(global_pc_capft[global_id])==2:
            global_pc_capft[global_id] = get_majority_cluster_mean(global_pc_capft[global_id])
            # global_pc_capft[global_id] = np.mean(global_pc_capft[global_id], axis=0)
            global_pc_capft[global_id] = global_pc_capft[global_id] /  np.linalg.norm(global_pc_capft[global_id])
        if np.ndim(global_pc_color[global_id])==2:
            global_pc_color[global_id] = np.mean(global_pc_color[global_id], axis=0)
            global_pc_color[global_id] = global_pc_color[global_id] /  np.linalg.norm(global_pc_color[global_id])
            
    # # 可视化一些初始分类，不能有背景囊括其他的情况出现
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # # 遍历每个global_id的点云并可视化
    # for global_id, pc in global_pc.items():
    #     # 设置颜色
    #     pc.paint_uniform_color(np.array(random_colors[global_id])/255.0)
    #     # 添加到可视化窗口
    #     vis.add_geometry(pc)
    # vis.run()
    # vis.destroy_window()
    
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_graph = os.path.join(args.output_graph,  current_time, "before")
    os.makedirs(output_graph, exist_ok=True)
    # 将聚类之后的mask_color绘制回到原先的图像上判断跨视图之间的mask是否一致    
    for i in range(len(mask_filter)):
        inst_image = np.zeros((mask_filter[0][0].shape[0], mask_filter[0][0].shape[1], 3), dtype=np.uint8)
        for j in range(len(mask_filter[i])):
            if cluster_mask_id[i][j] != 999:
                inst_image[mask_filter[i][j]] = random_colors[cluster_mask_id[i][j]]
        out_test_path = output_graph+"/inst_"+str(i)+".png"
        cv2.imwrite(out_test_path, inst_image)
    print("Mask graph generated successfully!", output_graph)
    
    # 计算overlap
    # similarity_matrix, mapping, mapping_counter = compute_similarity_matrix(global_pc, global_pc_capft, wall_fts, floor_fts, ceiling_fts, cap_thre=0.85,
    #                                                                         weight_pc=0.5, weight_caption=0.5, dis_thre=0.02, ratio_thre=0.5)
    
    similarity_matrix, mapping, mapping_counter = compute_similarity_matrix_thre(global_pc, global_pc_capft, global_pc_color, wall_fts, floor_fts, ceiling_fts, 
                                                                                 cap_thre=cfg["cap_thre"], weight_pc=cfg["weight_pc"], dis_thre=cfg["dis_thre"],
                                                                                 weightcaption=cfg["weightcaption"], weightcolor=cfg["weightcolor"])
    
    print(mapping)
    print("The object num after is:", mapping_counter)
    for i in range(len(mask_filter)):
        for j in range(len(mask_filter[i])):
            cluster_mask_id[i][j] = mapping[cluster_mask_id[i][j]]
                
    output_graph = os.path.join(args.output_graph,  current_time, "after")
    os.makedirs(output_graph, exist_ok=True)
    # 将聚类之后的mask_color绘制回到原先的图像上判断跨视图之间的mask是否一致    
    for i in range(len(mask_filter)):
        inst_image = np.zeros((mask_filter[0][0].shape[0], mask_filter[0][0].shape[1], 3), dtype=np.uint8)
        for j in range(len(mask_filter[i])):
            if cluster_mask_id[i][j] != 0:
                inst_image[mask_filter[i][j]] = random_colors[cluster_mask_id[i][j]]
        out_test_path = output_graph+"/inst_"+str(i)+".png"
        cv2.imwrite(out_test_path, inst_image)
    print("Mask graph generated successfully!", output_graph)
    
    # 按照vmap所需要的格式写
    dir_class = os.path.join(args.output_dir,  "class_our")
    os.makedirs(dir_class, exist_ok=True)
    dir_instance = os.path.join(args.output_dir,  "instance_our")
    os.makedirs(dir_instance, exist_ok=True)
    all_clip_feat = []
    all_cap_feat = []
    all_cap = []
    for i in range(len(mask_filter)):
        # 每个图像的特征是一个大字典
        clip_feat = {}
        cap_feat = {}
        caption = {}
        inst_image = np.zeros((mask_filter[0][0].shape[0], mask_filter[0][0].shape[1]), dtype=np.int32) 
        for j in range(len(mask_filter[i])):
            if cluster_mask_id[i][j] != 0:
                inst_image[mask_filter[i][j]] = cluster_mask_id[i][j]
                # 把特征放在大字典里面
                clip_feat.update({cluster_mask_id[i][j]: mask_clift[i][j]})
                cap_feat.update({cluster_mask_id[i][j]: mask_capft[i][j]})
                caption.update({cluster_mask_id[i][j]: mask_cap[i][j]})
        dir_class_path = dir_class+"/semantic_class_"+str(i)+".png"
        cv2.imwrite(dir_class_path, inst_image)
        dir_instance_path = dir_instance+"/semantic_instance_"+str(i)+".png"
        cv2.imwrite(dir_instance_path, inst_image)
        all_clip_feat.append(clip_feat)
        all_cap_feat.append(cap_feat)
        all_cap.append(caption)
    # 写clip和caption特征到文件
    object_clipfeat = args.output_dir+"object_clipfeat.pkl"
    object_capfeat = args.output_dir+"object_capfeat.pkl"
    object_caption = args.output_dir+"object_caption.pkl"
    with open(object_clipfeat, 'wb') as f:
        print("write to",object_clipfeat, len(all_clip_feat))
        pickle.dump(all_clip_feat, f)
    with open(object_capfeat, 'wb') as f:
        print("write to",object_capfeat, len(all_cap_feat))
        pickle.dump(all_cap_feat, f)
    with open(object_caption, 'wb') as f:
        print("write to",object_caption, len(all_cap))
        pickle.dump(all_cap, f)
    
    
    # # 可视化最后的结果
    # global_pc_new = {}
    # global_pc_capft_new = {}
    # for global_id, pc in global_pc.items():
    #     global_id_new = mapping[global_id]
    #     if global_id_new in global_pc_new.keys():
    #         global_pc_new[global_id_new] = np.vstack((global_pc_new[global_id_new], np.array(global_pc[global_id].points)))
    #         global_pc_capft_new[global_id_new] = np.vstack((global_pc_capft_new[global_id_new], global_pc_capft[global_id]))
    #     else:
    #         global_pc_new[global_id_new] = np.array(global_pc[global_id].points)
    #         global_pc_capft_new[global_id_new] = global_pc_capft[global_id]
            
    # for global_id_new, pc_new in global_pc_new.items():
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(pc_new)
    #     downsampled_pcd = pcd.voxel_down_sample(voxel_size=0.01)
    #     global_pc_new[global_id_new] = downsampled_pcd
    #     if np.ndim(global_pc_capft_new[global_id_new])==2:
    #         global_pc_capft_new[global_id_new] = np.mean(global_pc_capft_new[global_id_new], axis=0)
    #         global_pc_capft_new[global_id_new] = global_pc_capft_new[global_id_new] /  np.linalg.norm(global_pc_capft_new[global_id_new])
            
            
    # # 可视化一些初始分类，不能有背景囊括其他的情况出现
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # # 遍历每个global_id的点云并可视化
    # for global_id_new, pc_new in global_pc_new.items():
    #     # 设置颜色
    #     pc_new.paint_uniform_color(np.array(random_colors[global_id_new])/255.0)
    #     # 添加到可视化窗口
    #     vis.add_geometry(pc_new)
    # vis.run()
    # vis.destroy_window()