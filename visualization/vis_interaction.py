"""
将得到的地图可视化并实现查询等
"""
import sys
sys.path.append("/code1/dyn/github_repos/OpenObj/nerftrain")
import vis
from cfg import Config
import copy
import json
import os
import pickle
import gzip
import argparse
import random
import matplotlib
import numpy as np
import pandas as pd
import open3d as o3d
import torch.nn.functional as F
import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer
from utils import get_majority_cluster_mean
import distinctipy
import json
from PIL import Image
from pathlib import Path
from tqdm import trange
import re
import yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import clip
# import openai

saved_viewpoint = None

def main():
    # init config
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # setting params
    parser = argparse.ArgumentParser(description='Model training for single GPU')
    parser.add_argument('--dataset_name', default='Replica',type=str)
    parser.add_argument('--scene_name', default='room_0',type=str)
    parser.add_argument('--is_partcolor', action='store_true', help='Whether the scene is special or not')

    args = parser.parse_args()
    scene_name = args.scene_name
    dataset_name = args.dataset_name
    is_partcolor = args.is_partcolor
    log_dir = '/data/dyn/results/object/results/vMAP/'+scene_name
    config_file = '/code/dyn/object_map/vMAP/configs/'+dataset_name+'/'+scene_name+'.json'
    cfg = Config(config_file)       # config params
    result_path = log_dir+'/map_vis.pkl.gz'
    
    
    # 读取 YAML 文件，得到mapping的semantic
    if dataset_name == "Replica":
        with open('/data/dyn/object/vmap/semantic_mapping.yaml', 'r') as file:
            data = yaml.safe_load(file)
        # 提取第一个字典
        mapping_dict = data["mapping"]
        mapped_color = data["mapped_colors"]
        
    # 加载所有物体需要用到的结果
    with gzip.open(result_path, "rb") as f:
        all_obj = pickle.load(f)
    # 必须要有sbert模型哦
    print("Initializing SBERT model...")
    sbert_model = SentenceTransformer("/home/dyn/multimodal/SBERT/pretrained/model/all-MiniLM-L6-v2")
    sbert_model = sbert_model.to("cuda")
    print("Done initializing SBERT model.")
    # 还有clip模型
    clip_model, preprocess = clip.load("ViT-B/32", device="cuda")
    # 为了生成颜色
    cmap = matplotlib.colormaps.get_cmap("rainbow")
    # # gpt加载
    # openai.api_key = cfg.openai_key
    # openai.api_base = cfg.api_base
    # TIMEOUT = 25  # timeout in seconds
    # DEFAULT_PROMPT = """
    # You are an object picker that picks the three objects from a sequence of objects 
    # that are most relevant to the query statement, the first being the most relevant. 
    # The input format is: object index value and corresponding description, and query statement. 
    # The output format is: The three most relevant objects are [Index0,Index1,Index2].
    # Here's some example for you. 
    # Input: 
    # 'The sequence of objects: [0]A silver car.  [1]A green tree.  [2]A paved city road.  [3]A red car. 
    # Query statement: driving tools
    # '. 
    # You should output like this: 
    # '[0,3,2]
    # '
    # Input: 
    # 'The sequence of objects: [0]A green tree.  [1]A white building.  [2]A paved city road.  [3]A red car. 
    # Query statement: a comfortable bed
    # '. 
    # You should output like this: 
    # '[1,3,2]
    # '
    # Please note! Be sure to give three index values! No more and no less!
    # Please note! Be sure to give three index values! No more and no less!
    # Please note! Be sure to give three index values! No more and no less!
    
    
    # Note that the following are real inputs:
    # """
    # formatted_sentences = []
    # for i in range(len(objects)):
    #     formatted_sentences.append(f"[{i}]{objects[i]['caption']}.  ")
    # caption_all = " ".join(formatted_sentences)
    # caption_all = "The sequence of objects: "+caption_all+"\n"
    # # print(caption_all)
    
    # 创建可视化对象 设置窗口名
    vis_window = o3d.visualization.VisualizerWithKeyCallback()
    
    if result_path is not None:
        vis_window.create_window(window_name=f'Open3D - {os.path.basename(result_path)}', width=1680, height=945)
    else:
        vis_window.create_window(window_name=f'Open3D', width=1280, height=720)
    view_control = vis_window.get_view_control()
    
    all_mesh = []
    list_of_keys = list(all_obj.keys())
    # 向场景中添加几何图形并创建相应的着色器
    for obj in all_obj.values():
        all_mesh.append(vis.trimesh_to_open3d(obj["mesh"]))
    # 添加mesh进去
    for mesh in all_mesh:
        vis_window.add_geometry(mesh)  
    print("vis over")
    

    # 加载视角参数
    if os.path.isfile("vis_params/"+scene_name+"_vis_params.json"):
        global saved_viewpoint
        loaded_view_params = o3d.io.read_pinhole_camera_parameters("vis_params/"+scene_name+"_vis_params.json")
        saved_viewpoint = loaded_view_params
        view_control.convert_from_pinhole_camera_parameters(saved_viewpoint)
        
    # 判断哪些mesh所有顶点高于mesh，删除掉这些，可视化好看，同时计算得到最小bbox
    ceiling_idx=[]
    most_idx=[]
    all_bbox=[]
    for idx, mesh in enumerate(all_mesh):
        vertices = np.asarray(mesh.vertices)
        min_height = np.min(vertices[:, 2])
        # print(min_height)
        # 如果最大高度大于2米，则移除该网格对象
        height = 1
        if scene_name == "room_2":
            height = -0.5
        if scene_name == "office_0":
            height = 0.5
        if min_height > height:
            ceiling_idx.append(idx)
        # # 去掉room_0的地毯以可视化好看
        # if scene_name == "room_0" and all_obj[list_of_keys[idx]]["class_id"]+1 in [50,98]:
        #     ceiling_idx.append(idx)
        wall_id = 93
        if dataset_name == "Scannet":
            wall_id = 1
        # 611的墙面分的不是很好哦
        if scene_name == "611":
            if list_of_keys[idx] != 46:
                most_idx.append(idx)
        elif all_obj[list_of_keys[idx]]["class_id"]+1 != wall_id:
            # print(all_obj[list_of_keys[idx]]["class_id"])
            most_idx.append(idx)
        # 最小bbox
        bbox = mesh.get_axis_aligned_bounding_box()
        all_bbox.append(bbox)
    print("These objects are too high:", ceiling_idx)
    print("These are most objects:", most_idx)
    # 默认移除这些太高的
    if dataset_name == "Replica":
        for idx in ceiling_idx:
            vis_window.remove_geometry(all_mesh[idx], reset_bounding_box=False)
            vis_window.remove_geometry(all_bbox[idx], reset_bounding_box=False)
    # 设置实例颜色
    instance_colors = distinctipy.get_colors(len(all_mesh), pastel_factor=0.5)  
    
    all_partcolor = []
    all_clipfeat = []
    all_sbertfeat = []
    for idx, mesh in enumerate(all_mesh):
        part_feat = all_obj[list_of_keys[idx]]["part_feat"]
        clip_feat = all_obj[list_of_keys[idx]]["clip_feat"]
        eps = 0.2
        min_samples = 2
        if np.ndim(clip_feat) == 2:
            clip_feat = get_majority_cluster_mean(clip_feat, eps, min_samples)
        clip_feat=torch.tensor(clip_feat, device="cuda").clone().detach()
        all_clipfeat.append(clip_feat)
        caption_feat = all_obj[list_of_keys[idx]]["caption_feat"]
        if np.ndim(caption_feat) == 2:
            caption_feat = get_majority_cluster_mean(caption_feat, eps, min_samples)
        caption_feat=torch.tensor(caption_feat, device="cuda").clone().detach()
        all_sbertfeat.append(caption_feat)
        if is_partcolor:
            # 计算PCA
            pca = PCA(n_components=3)
            scaler = StandardScaler()
            standardized_features = scaler.fit_transform(part_feat)
            scaled_colors = pca.fit_transform(standardized_features)  # part_feat 是形状为 (N, 512) 的特征张量
            min_value = scaled_colors.min()
            max_value = scaled_colors.max()
            scaled_colors = (scaled_colors - min_value) / (max_value - min_value)  # 将特征值缩放到 [0, 1] 区间
            scaled_colors = scaled_colors.clip(0, 1)  # 将特征值限制在 [0, 1] 区间
            all_partcolor.append(scaled_colors)
    all_clipfeat = torch.cat([feat.unsqueeze(0) for feat in all_clipfeat], dim=0)
    all_sbertfeat = torch.cat([feat.unsqueeze(0) for feat in all_sbertfeat], dim=0)
    print(all_clipfeat.shape, all_sbertfeat.shape)
    print("over paerfeat color")
    
        


    main.show_ceiling = False
    def toggle_ceiling(vis_window):
        '''
        隐藏or显示背景物体
        '''
        if ceiling_idx is None:
            print("No ceiling objects found.")
            return
        for idx in ceiling_idx:
            if main.show_ceiling:
                vis_window.add_geometry(all_mesh[idx], reset_bounding_box=False)
                # vis_window.add_geometry(all_bbox[idx], reset_bounding_box=False)
            else:
                vis_window.remove_geometry(all_mesh[idx], reset_bounding_box=False)
                # vis_window.remove_geometry(all_bbox[idx], reset_bounding_box=False)
        print("remove these high objects", main.show_ceiling)
        main.show_ceiling = not main.show_ceiling
        
        

    main.hide_most = True
    def hide_most(vis_window):
        '''
        隐藏or显示大部分物体以方便拖动
        '''
        for idx in most_idx:
            if main.hide_most:
                vis_window.remove_geometry(all_mesh[idx], reset_bounding_box=False)
                # vis_window.remove_geometry(all_bbox[idx], reset_bounding_box=False)
            else:
                if (not main.show_ceiling) and (idx in ceiling_idx):
                    continue
                vis_window.add_geometry(all_mesh[idx], reset_bounding_box=False)
                # vis_window.add_geometry(all_bbox[idx], reset_bounding_box=False)
        print("remove most objects", main.hide_most)
        main.hide_most = not main.hide_most
        
        

    def color_by_instance(vis_window):
        '''
        根据实例随机着色
        '''
        # 设置每个对象的点云颜色
        for i in range(len(all_mesh)):
            color = instance_colors[i]
            mesh = all_mesh[i]
            mesh.vertex_colors = o3d.utility.Vector3dVector(np.tile(color, (len(mesh.vertices), 1)))
        for mesh in all_mesh:
            vis_window.update_geometry(mesh)
        print("seted instance color")
        
            
    def color_by_class(vis_window):
        '''
        根据语义类别，按照类别着色
        '''
        # 设置每个对象的点云颜色
        
        for i in range(len(all_mesh)):
            semantic_id = mapping_dict[all_obj[list_of_keys[i]]["class_id"]+1]
            color = mapped_color[semantic_id]
            mesh = all_mesh[i]
            colors = np.tile(color, (len(mesh.vertices), 1))
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        for mesh in all_mesh:
            vis_window.update_geometry(mesh)
        print("seted class color")
            
    def color_by_rgb(vis_window):
        '''
        RGB着色
        '''
        # 设置每个对象的点云颜色
        for i in range(len(all_obj)):
            colors = np.asarray(all_obj[list_of_keys[i]]["color"])[...,:-1]/255
            brightness_factor = 0.8  # 调整亮度的因子，值越小越暗
            darkened_colors = colors * brightness_factor
            mesh = all_mesh[i]
            mesh.vertex_colors = o3d.utility.Vector3dVector(darkened_colors)
        for mesh in all_mesh:
            vis_window.update_geometry(mesh)
        print("seted rgb color")
        
    def color_by_partfeat(vis_window):
        '''
        安装partfeat着色
        '''
        # 设置每个对象的点云颜色
        for i in range(len(all_obj)): # 将特征值限制在 [0, 1] 区间
            # print(scaled_colors.shape)
            mesh = all_mesh[i]
            mesh.vertex_colors = o3d.utility.Vector3dVector(all_partcolor[i])
        for mesh in all_mesh:
            vis_window.update_geometry(mesh)
        print("seted partfeat color")
        

    def sim_and_update(similarities, vis_window, top_num = 0):
        '''
        文本、图像查询的相似性计算及点云更新
        '''
        top_indices = None
        if top_num != 0:
            top_values, top_indices = similarities.topk(top_num)
        max_value = similarities.max()
        min_value = similarities.min()
        normalized_similarities = (similarities - min_value) / (max_value - min_value)
        similarity_colors = cmap(normalized_similarities.detach().cpu().numpy())[..., :3]        
        # 更新点云对象的颜色属性，以反映每个对象的相似性
        for i in range(len(all_mesh)):
            mesh = all_mesh[i]
            if top_indices is None:
                mesh.vertex_colors = o3d.utility.Vector3dVector(np.tile(
                        [similarity_colors[i, 0].item(), similarity_colors[i, 1].item(), similarity_colors[i, 2].item()], 
                        (len(mesh.vertices), 1)    
                    ))
            elif top_indices is not None and i in top_indices:
                mesh.vertex_colors = o3d.utility.Vector3dVector(np.tile(
                        [1, 0, 0], 
                        (len(mesh.vertices), 1)    
                    ))
            else:
                colors = np.asarray(all_obj[list_of_keys[i]]["color"])[...,:-1]/255
                brightness_factor = 0.5  # 调整亮度的因子，值越小越暗
                darkened_colors = colors * brightness_factor
                mesh = all_mesh[i]
                mesh.vertex_colors = o3d.utility.Vector3dVector(darkened_colors)
        for mesh in all_mesh:
            vis_window.update_geometry(mesh)  
        print("seted object color")      

    def color_by_sbert_sim(vis_window):
        '''
        文本查询，得到相似性
        '''
        text_query = input("Enter your query: ")
        text_queries = [text_query]
        top_num = int(input("Enter your top num (such as 0(all) or 1 or 2): "))
        # 对输入的文本进行sbert编码
        with torch.no_grad():
            object_sbert_ft = sbert_model.encode(text_queries, convert_to_tensor=True)
            object_sbert_ft /= object_sbert_ft.norm(dim=-1, keepdim=True)
            object_sbert_ft = object_sbert_ft.squeeze()
        with torch.no_grad():
            object_clip_ft = clip_model.encode_text(clip.tokenize(text_queries).to("cuda"))
            object_clip_ft /= object_clip_ft.norm(dim=-1, keepdim=True)
            object_clip_ft = object_clip_ft.squeeze()
        similarities_sbert = F.cosine_similarity(object_sbert_ft.unsqueeze(0), all_sbertfeat, dim=-1)
        similarities_clip = F.cosine_similarity(object_clip_ft.unsqueeze(0), all_clipfeat, dim=-1)
        # 两个求取加权和
        similarities = similarities_sbert * 0.2 + similarities_clip * 0.8
        sim_and_update(similarities, vis_window, top_num = top_num)
        
    def sim_and_update_part(similarities, vis_window, part_clip_ft, top_num = 1):
        '''
        部件级可视化更新
        '''
        # 先找到在哪些物体上做特征可视化
        top_values, top_indices = similarities.topk(top_num)
        for idx in top_indices:
            this_obj_partfeat = all_obj[list_of_keys[idx]]["part_feat"]
            this_obj_partfeat = all_obj[list_of_keys[idx]]["part_feat"]
            this_obj_partfeat = torch.tensor(this_obj_partfeat, device="cuda")
            similarities= F.cosine_similarity(part_clip_ft.unsqueeze(0), this_obj_partfeat, dim=-1)
            max_value = similarities.max()
            min_value = similarities.min()
            normalized_similarities = (similarities - min_value) / (max_value - min_value)
            similarity_colors = cmap(normalized_similarities.detach().cpu().numpy())[..., :3]  
            print(similarity_colors.shape) 
            # 改变这些mesh的颜色
            mesh = all_mesh[idx]
            mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(similarity_colors)) 
        # 更新点云对象的颜色属性，以反映每个对象的相似性
        for i in range(len(all_mesh)):
            mesh = all_mesh[i]
            if i not in top_indices:
                colors = np.asarray(all_obj[list_of_keys[i]]["color"])[...,:-1]/255
                brightness_factor = 0.5  # 调整亮度的因子，值越小越暗
                darkened_colors = colors * brightness_factor
                mesh = all_mesh[i]
                mesh.vertex_colors = o3d.utility.Vector3dVector(darkened_colors)
        for mesh in all_mesh:
            vis_window.update_geometry(mesh)       
            
            
    def color_by_part_sim(vis_window):
        '''
        文本查询，检索部件级特征
        '''
        object_query = input("Enter your query for object: ")
        object_query = [object_query]
        top_num = int(input("Enter your top num (such as  1 or 2): "))
        part_query = input("Enter your query for part: ")
        part_query = [part_query]
        # 对输入的文本进行sbert编码
        with torch.no_grad():
            object_sbert_ft = sbert_model.encode(object_query, convert_to_tensor=True)
            object_sbert_ft /= object_sbert_ft.norm(dim=-1, keepdim=True)
            object_sbert_ft = object_sbert_ft.squeeze()
        with torch.no_grad():
            object_clip_ft = clip_model.encode_text(clip.tokenize(object_query).to("cuda"))
            object_clip_ft /= object_clip_ft.norm(dim=-1, keepdim=True)
            object_clip_ft = object_clip_ft.squeeze()
            # 部件级只需要clip特征
            part_clip_ft = clip_model.encode_text(clip.tokenize(part_query).to("cuda"))
            part_clip_ft /= part_clip_ft.norm(dim=-1, keepdim=True)
            part_clip_ft = part_clip_ft.squeeze()
        similarities_sbert = F.cosine_similarity(object_sbert_ft.unsqueeze(0), all_sbertfeat, dim=-1)
        similarities_clip = F.cosine_similarity(object_clip_ft.unsqueeze(0), all_clipfeat, dim=-1)
        # 两个求取加权和
        similarities = similarities_sbert * 0.2 + similarities_clip * 0.8
        # 直接得到前多少个物体，默认1
        sim_and_update_part(similarities, vis_window, part_clip_ft, top_num = top_num)
        print("seted part color")


    # def color_by_llm(vis_window):
    #     '''
    #     文本查询
    #     '''
    #     text_query = input("Enter your query for LLM: ")
    #     text_queries = "Query statement: "+text_query
    #     chat_completion = openai.ChatCompletion.create(
    #         model="gpt-4",
    #         messages=[{"role": "user", "content": DEFAULT_PROMPT + "\n\n" + caption_all+text_queries}],
    #         timeout=TIMEOUT,  # Timeout in seconds
    #     )
    #     input_text = chat_completion["choices"][0]["message"]["content"]
    #     print(input_text)
    #     pattern = r'\[([^]]+)\]'  # 匹配方括号中的内容
    #     match = re.search(pattern, input_text)
    #     extracted_content = []
    #     if match:
    #         extracted_content = match.group(1)
    #     index = [int(x) for x in extracted_content.split(',')]
    #     print(index)
    #     # 更新点云对象的颜色属性，以反映每个对象的相似性
    #     for i in range(len(objects)):
    #         if i == index[0]:
    #             pcd = pcds[i]
    #             print(objects[i]["caption"])
    #             pcd.colors = o3d.utility.Vector3dVector(np.tile([1,0,0], (len(pcd.points), 1)))
    #         elif i == index[1]:
    #             pcd = pcds[i]
    #             print(objects[i]["caption"])
    #             pcd.colors = o3d.utility.Vector3dVector(np.tile([0,1,0], (len(pcd.points), 1)))
    #         elif i == index[2]:
    #             pcd = pcds[i]
    #             print(objects[i]["caption"])
    #             pcd.colors = o3d.utility.Vector3dVector(np.tile([0,0,1], (len(pcd.points), 1)))
    #         else:
    #             pcd = pcds[i]
    #             pcd.colors = o3d.utility.Vector3dVector(np.tile([0.7,0.7,0.7], (len(pcd.points), 1)))
    #     for pcd in pcds:
    #         vis_window.update_geometry(pcd)        

    
        
    main.show_bbox = True
    def show_bbox(vis_window):
        line_color=(0.357, 0.608, 0.835)
        if main.show_bbox:
            for i, geometry in enumerate(all_bbox):
                geometry.color=line_color
                vis_window.add_geometry(geometry, reset_bounding_box=False)
        else:
            for geometry in all_bbox:
                vis_window.remove_geometry(geometry, reset_bounding_box=False)
        main.show_bbox = not main.show_bbox
    
    def save_viewpoint(vis_window):
        global saved_viewpoint
        saved_viewpoint = view_control.convert_to_pinhole_camera_parameters()
        # 保存视角参数
        view_params = view_control.convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters("vis_params/"+scene_name+"_vis_params.json", view_params)
        
    def save_viewpoint_this(vis_window):
        global saved_viewpoint
        saved_viewpoint = view_control.convert_to_pinhole_camera_parameters()
        
    def restore_viewpoint(vis_window):
        global saved_viewpoint
        if saved_viewpoint is not None:
            view_control.convert_from_pinhole_camera_parameters(saved_viewpoint)
            
            
    vis_window.register_key_callback(ord("C"), toggle_ceiling)
    vis_window.register_key_callback(ord("A"), show_bbox)
    vis_window.register_key_callback(ord("S"), color_by_class)
    vis_window.register_key_callback(ord("I"), color_by_instance)
    vis_window.register_key_callback(ord("R"), color_by_rgb)
    if is_partcolor:
        vis_window.register_key_callback(ord("O"), color_by_partfeat)  # 注释因为太耗时了，一般也用不到
    vis_window.register_key_callback(ord("F"), color_by_sbert_sim)
    vis_window.register_key_callback(ord("P"), color_by_part_sim)
    vis_window.register_key_callback(ord("H"), hide_most)  
    # vis_window.register_key_callback(ord("M"), color_by_llm)   
    # vis_window.register_key_callback(ord("V"), save_viewpoint)  # 注释以防止瞎按
    vis_window.register_key_callback(ord("T"), save_viewpoint_this)  # 保存视角到全局变量但不覆盖本地文件，仅本次使用
    vis_window.register_key_callback(ord("X"), restore_viewpoint)   
        
    vis_window.run()   

if __name__ == "__main__":
    main()