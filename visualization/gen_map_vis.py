'''
这个用来生成用于可视化的地图
'''
import sys
sys.path.append("/code1/dyn/github_repos/OpenObj/objnerf")
import time
import loss
from vmap import *
import utils
import open3d
import dataset
import vis
from functorch import vmap
import argparse
from cfg import Config
import re
import shutil
import open3d as o3d
import cv2
import render_rays
import matplotlib.pyplot as plt
from datetime import datetime
from utils import performance_measure, bbox_bbox2open3d
from plyfile import *
import numpy as np
import os
import yaml
from sklearn.decomposition import PCA



            
if __name__ == "__main__":
    use_for_man_map = True
    
    
    #############################################
    # init config
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


       # setting params
    parser = argparse.ArgumentParser(description='Model training for single GPU')
    parser.add_argument('--dataset_name', default='Replica',type=str)
    parser.add_argument('--scene_name', default='room_0',type=str)
    args = parser.parse_args()
    
    

    scene_name = args.scene_name
    dataset_name = args.dataset_name
    log_dir = '/data/dyn/results/object/results/vMAP/'+scene_name
    config_file = '/code/dyn/object_map/vMAP/configs/'+dataset_name+'/'+scene_name+'.json'
    os.makedirs(log_dir, exist_ok=True)  # saving logs
    shutil.copy(config_file, log_dir)
    cfg = Config(config_file)       # config params


    # 初始化物体的列表
    obj_dict = {}   # 只包含物体
    vis_dict = {}   # 还包含背景
    
    # 打开ckpt的文件夹
    ckpt_dir = os.path.join(log_dir, "ckpt")
    obj_id_list = [int(subdir) for subdir in os.listdir(ckpt_dir) if os.path.isdir(os.path.join(ckpt_dir, subdir))]

    dataloader = dataset.init_loader(cfg)
    dataloader_iterator = iter(dataloader)
    dataset_len = len(dataloader)
    # 随便取出一个用于初始化
    sample = next(dataloader_iterator)
    rgb = sample["image"].to(cfg.data_device)
    depth = sample["depth"].to(cfg.data_device)
    twc = sample["T"].to(cfg.data_device)
    bbox_dict = sample["bbox_dict"]
    bbox2d = bbox_dict[0]
    live_frame_id = sample["frame_id"]
    inst = sample["obj"].to(cfg.data_device)
    state = torch.zeros_like(inst, dtype=torch.uint8, device=cfg.data_device)
    
    for obj_id in obj_id_list:
        print("loaded obj:", obj_id)
        if obj_id == 0:
            scene_bg = sceneObject(cfg, obj_id, rgb, depth, state, bbox2d, twc, live_frame_id, clip_feat = None, caption_feat = None)
            ok = scene_bg.load_checkpoints(os.path.join(ckpt_dir, str(obj_id))+"/obj_"+str(obj_id)+".pth")
            vis_dict.update({obj_id: scene_bg})
        else:
            scene = sceneObject(cfg, obj_id, rgb, depth, state, bbox2d, twc, live_frame_id, clip_feat = None, caption_feat = None)
            ok = scene.load_checkpoints(os.path.join(ckpt_dir, str(obj_id))+"/obj_"+str(obj_id)+".pth")
            if ok:
                obj_dict.update({obj_id: scene})
                vis_dict.update({obj_id: scene})
            
    # 读取 YAML 文件，得到mapping的semantic
    if dataset_name == "Replica":
        with open('./replica_color.yaml', 'r') as file:
            data = yaml.safe_load(file)
        # 提取第一个字典
        mapping_dict = data["mapping"]
        mapped_color = data["mapped_colors"]
    # 读取 YAML 文件，得到mapping的semantic
    elif dataset_name == "Scannet":
        with open('./scannet_color.yaml', 'r') as file:
            data = yaml.safe_load(file)
        # 提取第一个字典
        mapping_dict = data["order"]
        mapped_color = data["mapped_colors"]
            
    # 一个大字典，记录所有物体的mesh和原始颜色和特征
    all_obj = {}
    # 记录每个物体的mesh和颜色和特征
    for obj_id, obj_k in vis_dict.items():
        # if obj_id != 6:
        #     continue
        print("obj:", obj_id)
        bound = obj_k.bbox3dour
        # bound.extent = bound.extent*1.5 # 有时候观测不全，后面是空洞
        adaptive_grid_dim = int(np.minimum(np.max(bound.extent)//cfg.live_voxel_size+1, cfg.grid_dim))
        pcd, mesh, part_feat = obj_k.trainer.meshing(bound, obj_k.obj_center, grid_dim=128, save_pcd=False, save_mesh=True, if_color=True, if_part=True)
        raw_colors = mesh.visual.vertex_colors
        print(raw_colors.shape)
        part_feat = part_feat.cpu().numpy()
        part_feat = part_feat/np.linalg.norm(part_feat, axis=-1, keepdims=True)
        print(part_feat.shape)
        
        # open3d_mesh = vis.trimesh_to_open3d(mesh)
        this_obj = {
            "clip_feat": obj_k.clip_feat,
            "caption_feat": obj_k.caption_feat,
            "class_id": obj_k.semantic_id,
            "mesh": mesh,
            "color": raw_colors,
            "part_feat": part_feat,
        }
        
        all_obj.update({obj_id: this_obj})
    
    import gzip
    import pickle
    save_path = log_dir+'/map_vis.pkl.gz'
    print(os.path.dirname(save_path))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(save_path)
    with gzip.open(save_path, "wb") as f:
        pickle.dump(all_obj, f)