# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os
from natsort import natsorted

# fmt: off
import sys
# sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.append('/code/dyn/object_map/third_parites/')
sys.path.append('/code/dyn/object_map/third_parites/detectron2/')
sys.path.append('/code/dyn/object_map/third_parites/detectron2/projects/CropFormer/')
# sys.path.append('/code/dyn/object_map/third_parites/detectron2/projects/CropFormer/')

# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
from tqdm import tqdm, trange
import torch
from PIL import Image

import pickle

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from demo_cropformer.predictor import VisualizationDemo

# from inst_class import InstFrame
import spacy

# tap模型
from tokenize_anything import model_registry
from tokenize_anything.utils.image import im_rescale
from tokenize_anything.utils.image import im_vstack
# sbert模型
from sentence_transformers import SentenceTransformer, util
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
# clip模型
# import open_clip
import clip

def make_colors():
    # entityv2 所使用的entity分割颜色来自于COCO数据集
    from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
    colors = []
    for cate in COCO_CATEGORIES:
        colors.append(cate["color"])
    return colors


# constants
WINDOW_NAME = "cropformer demo"

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="/code/dyn/object_map/third_parites/detectron2/projects/CropFormer/configs/entityv2/entity_segmentation/cropformer_hornet_3x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )

    parser.add_argument(
        "--input_depth",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )

    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--out-type",
        type=int,
        default=0,
        help="0: only mask, 1: img+mask",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

def min_rect_bbox(mask):
    '''
    寻找非零像素点的坐标，得到最小bbox
    '''
    nonzero_indices = np.nonzero(mask)
    if len(nonzero_indices[0]) == 0:
        # 如果 mask 中没有非零元素，返回空的矩形
        return np.zeros((4, 2), dtype=np.intp)
    # 计算轴对齐矩形
    x, y, w, h = cv2.boundingRect(np.column_stack(nonzero_indices))
    # 根据左上角点和宽高计算矩形的四个顶点
    rect = np.array([[y, x], [y, x + w], [y + h, x + w], [y + h, x]], dtype=np.intp)
    return rect
    
def closest_distance(mask1, mask2):
    '''
    计算mask的最大距离
    '''
    # 找到mask1和mask2的边界点
    points_mask1 = np.transpose(np.nonzero(mask1))
    points_mask2 = np.transpose(np.nonzero(mask2))
    # 取十分之一的点
    sample_size1 = max(int(points_mask1.shape[0]/10),5)
    sample_size2 = max(int(points_mask2.shape[0]/10),5)
    # 随机选择一小部分点进行计算
    sample_indices_mask1 = np.random.choice(points_mask1.shape[0], size=sample_size1, replace=False)
    sample_indices_mask2 = np.random.choice(points_mask2.shape[0], size=sample_size2, replace=False)
    sample_points_mask1 = points_mask1[sample_indices_mask1]
    sample_points_mask2 = points_mask2[sample_indices_mask2]
    # 构建KD树
    tree_mask2 = cKDTree(sample_points_mask2)
    # 查询最近点对
    distances, _ = tree_mask2.query(sample_points_mask1)
    # 找到最小距离
    min_distance = np.min(distances)
    return min_distance


def split_mask(mask_id, mask, min_area=100, dis_eps=100, if_vis=False):
    '''
    判断mask有多少联通区域，面积大，距离远，则分开，默认最多两个物体被分为一个了
    '''
    # 连通性分析，得到连通区域数量
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8),connectivity=8)
    if num_labels > 2:
        # 移除小于min_area个像素的子mask
        filtered_masks = []
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                filtered_masks.append(labels == i)
            # 小于阈值就直接mask_id为0吧
            else:
                mask_id[labels == i]=0
        if len(filtered_masks) > 0: 
            # 使用DBSCAN对剩余的子mask进行聚类
            distances = np.zeros((len(filtered_masks), len(filtered_masks)))
            # 计算两两mask之间的最短距离
            for i in range(len(filtered_masks)):
                for j in range(i + 1, len(filtered_masks)):
                    distances[i, j] = closest_distance(filtered_masks[i], filtered_masks[j])
                    distances[j, i] = distances[i, j]
                    
            clustering = DBSCAN(eps=dis_eps, min_samples=1, metric='precomputed').fit(distances)
            clu_labels = clustering.labels_
            
            # 构建聚类后的mask列表
            clustered_masks = {}
            for i, label in enumerate(clu_labels):
                if label != -1:  # -1代表噪声，不属于任何聚类
                    if label not in clustered_masks:
                        clustered_masks[label] = filtered_masks[i]
                    else:
                        clustered_masks[label] = np.logical_or(clustered_masks[label], filtered_masks[i])
        else:
            clustered_masks = {}
    else:
        clustered_masks = {0:mask}
    # # 显示分割结果
    if if_vis and len(clustered_masks)>0:
        for label, clustered_mask in clustered_masks.items():
            cv2.imshow(f'Segmented Mask {label}', clustered_mask.astype(np.uint8) * 255)
            cv2.moveWindow(f'Segmented Mask {label}', 200 , 200)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return mask_id, clustered_masks

def draw_text_on_image(image, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.8, color=(255, 255, 255), thickness=2, bg_color=(255, 255, 255), bg_alpha=0.7):
    """
    在图像上绘制文本
    """
    # 转换成 OpenCV 中的图像格式
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # 获取文本大小
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    # 计算背景框的位置
    bg_x, bg_y = position
    bg_w = text_width + 2
    bg_h = text_height + 3
    # 绘制带有透明背景的矩形
    cv2.rectangle(image, (bg_x, bg_y - bg_h), (bg_x + bg_w, bg_y), bg_color, -1)
    # 绘制文本
    cv2.putText(image, text, position, font, font_scale, (color[2], color[1], color[0]), thickness)
    # 转换回 RGB 格式
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

if __name__ == "__main__":
    # 一些初始化
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    cfg = setup_cfg(args)

    colors = make_colors()
    demo = VisualizationDemo(cfg)
    
    if len(args.input) == 1:
        print("len(args.input) == 1")
        args.input = glob.glob(os.path.expanduser(args.input[0]))
        assert args.input, "The input path(s) was not found"
    # 读取目录下对齐的RGB-D图像序列
    args.input = natsorted(args.input) # 将读取到的image按照序列先后进行由小到大排序
    # args.input_depth = natsorted(args.input_depth)

    # maskcluster的运行图像间隔skip
    # skip = 3000
    skip = 10
    args.input = args.input[0:-1:skip]
    print(args.input[0:-1:skip])
    
    # 所有图像mask_id矩阵的列表
    mat_all_image = []
    # 所有图像的所有mask列表的列表
    mask_all_image = []
    # 所有图像的bbox列表的列表
    bbox_all_image = []
    # 所有图像
    color_mask_all = []
    # 所有图像的颜色
    colors_all_image = []
    
    seg_count_max = 999 #10 100
    seg_count = 0
    for path in tqdm(args.input, disable=not args.output):
        seg_count += 1
        if seg_count > seg_count_max:
            break
        # 使用的PLT的图像格式
        print(f"Processing frame id = {path}")
        img = read_image(path, format="BGR")
        
        #####################################################
        ########## 一、使用cropformer得到分割mask_id矩阵 ######
        #####################################################
        predictions = demo.run_on_image(img) 
        pred_masks = predictions["instances"].pred_masks
        pred_scores = predictions["instances"].scores
        # 按照得分筛选
        selected_indexes = (pred_scores >= args.confidence_threshold)
        selected_scores = pred_scores[selected_indexes]
        selected_masks  = pred_masks[selected_indexes]
        print(f"Detected {selected_masks.shape[0]} instances after threshold.")
        _, m_H, m_W = selected_masks.shape
        mask_id = np.zeros((m_H, m_W), dtype=np.uint8)
        # rank 经过修改之后 mask id刚好是按照scores的大小 越靠前的id 分割score值越大
        selected_scores, ranks = torch.sort(selected_scores, descending = True)
        ranks = ranks + 1
        for i in range(len(ranks)):
            # 0就是没有超过mask阈值的部分
            mask_id[(selected_masks[ranks[i]-1]==1).cpu().numpy()] = int(i+1)
            
        # 把矩阵保存
        mat_all_image.append(mask_id)
        unique_mask_id = np.unique(mask_id)
        # print(unique_mask_id)
        color_mask = np.zeros(img.shape, dtype=np.uint8)
        # 所有的mask矩阵（不包含不确定，即0的哦）、和对应的矩形框
        masks = []
        min_rects = []
        # if seg_count == 1:
        #     i = 0
        # 现在的mask个数
        now_mask_num = unique_mask_id.shape[0]
        # 可能会增加的mask id号是
        add_num = []
        # 第一遍流程，拆分
        for count in unique_mask_id:
            # print("count:",count)
            # 未知的像素不用计算mask
            if count == 0:
                continue
            mask_this = mask_id == count
            # 其实mask_id里对应的1,2,3,...是按照置信度分数高低来排序的
            # 寻找非零像素点的坐标，少于N个就不要了，设置为未知把
            nonzero_indices = np.nonzero(mask_this)
            if len(nonzero_indices[0]) < 100:
                mask_id[mask_this] = 0
                continue
            # 判断mask有几个联通区域，是否需要拆分
            dis_eps = (mask_this.shape[0]+mask_this.shape[1])/10
            # print("dis_eps",dis_eps)
            mask_id, clustered_masks = split_mask(mask_id, mask_this, min_area=100, dis_eps=dis_eps, if_vis=(seg_count==3200))
            # 需要拆分
            # 拆完之后都很小，不用处理了
            if len(clustered_masks) == 0:
                continue
            # 拆完之后有多个
            if len(clustered_masks)>1:
                for index, (label, clustered_mask) in enumerate(clustered_masks.items()):
                    # 处了第一个修改，全部修改mask_id内容
                    if index > 0:
                        mask_id[clustered_mask] = now_mask_num
                        now_mask_num += 1
        # 重新开始
        colors_this = []
        unique_mask_id = np.unique(mask_id)
        for count in unique_mask_id:
            if count == 0:
                continue
            mask_this = mask_id == count
            nonzero_indices = np.nonzero(mask_this)
            if len(nonzero_indices[0]) < 100:
                mask_id[mask_this] = 0
                continue
            # 计算最小bbox
            min_rect = min_rect_bbox(mask_this)
            # 合格的物体则往上加
            masks.append(mask_this)
            min_rects.append(min_rect)
            color_mask[mask_this] = colors[count]
            colors_this.append(colors[count])
            
        # 把mask和bbox列表保存
        mask_all_image.append(masks)
        bbox_all_image.append(min_rects)
        colors_all_image.append(colors_this)
        for min_rect in min_rects:
            # 绘制矩形
            cv2.drawContours(color_mask, [min_rect], 0, (255, 255, 255), thickness=3)
        
        # 可视化保存
        color_mask_all.append(color_mask)
    
    
    # 清空显存 通过对tensor张量的完全delete来释放占用的显存？
    del predictions,pred_masks,pred_scores,selected_masks,mask_id,color_mask
    torch.cuda.empty_cache()
    ######################################################
    ########## 二、使用tap为边界框生成caption并编码 ########
    ######################################################
    # 使用tap模型和图像创建分割器
    model_type = "tap_vit_l"
    checkpoint = "/home/dyn/outdoor/tokenize-anything/weights/tap_vit_l_03f8ec.pkl"
    tap_model = model_registry[model_type](checkpoint=checkpoint)
    concept_weights = "/home/dyn/outdoor/tokenize-anything/weights/merged_2560.pkl"
    tap_model.concept_projector.reset_weights(concept_weights)
    tap_model.text_decoder.reset_cache(max_batch_size=1000)
    # SBERT文本编码器
    sbert_model = SentenceTransformer('/home/dyn/multimodal/SBERT/pretrained/model/all-MiniLM-L6-v2')
    # NLP用于提取主干
    nlp = spacy.load("en_core_web_sm")
    
    # 所有图像的caption列表保存下来
    caption_all_image = []
    # 所有图像的caption特征列表保存下来
    capfeat_all_image = []
    seg_count = 0
    for path in tqdm(args.input, disable=not args.output):
        seg_count += 1
        if seg_count > seg_count_max:
            break
        print(f"Processing frame id = {path}")
        img = read_image(path, format="BGR")
        min_rects = bbox_all_image[seg_count-1]
        # 2.1 图像预处理
        img_list, img_scales = im_rescale(img, scales=[1024], max_size=1024)
        input_size, original_size = img_list[0].shape, img.shape[:2]
        img_batch = im_vstack(img_list, fill_value=tap_model.pixel_mean_value, size=(1024, 1024))
        inputs = tap_model.get_inputs({"img": img_batch})
        inputs.update(tap_model.get_features(inputs))
        # 2.2 根据上面的mask转化需要的格式
        batch_points = np.zeros((len(min_rects), 2, 3), dtype=np.float32)
        for j in range(len(min_rects)):
            # 最小的点
            batch_points[j, 0, 0] = min_rects[j][0, 0]
            batch_points[j, 0, 1] = min_rects[j][0, 1]
            batch_points[j, 0, 2] = 2
            # 最大的点
            batch_points[j, 1, 0] = min_rects[j][2, 0]
            batch_points[j, 1, 1] = min_rects[j][2, 1]
            batch_points[j, 1, 2] = 3
        # print(batch_points)
        inputs["points"] = batch_points
        inputs["points"][:, :, :2] *= np.array(img_scales, dtype="float32")
        # 2.3 模型开始推理，得到captions
        outputs = tap_model.get_outputs(inputs)
        iou_pred = outputs["iou_pred"].detach().cpu().numpy()
        point_score = batch_points[:, 0, 2].__eq__(2).__sub__(0.5)[:, None]
        rank_scores = iou_pred + point_score * ([1000] + [0] * (iou_pred.shape[1] - 1))
        mask_index = np.arange(rank_scores.shape[0]), rank_scores.argmax(1)
        sem_tokens = outputs["sem_tokens"][mask_index].unsqueeze_(1)
        captions = tap_model.generate_text(sem_tokens)
        # 提取主干
        new_captions = []
        # 提取句子的主语
        for sentence in captions:
            doc = nlp(str(sentence))
            subject = ""
            for npp in doc.noun_chunks:
                if sentence.startswith(str(npp)):
                    subject = str(npp)
                    break
            if not subject:
                subject = sentence
            new_captions.append(subject)
        
        # 绘制并保存
        color_mask = color_mask_all[seg_count-1]
        colors_this = colors_all_image[seg_count-1]
        for j, (caption, min_rect, color_this) in enumerate(zip(new_captions, min_rects, colors_this)):
            # 计算边界框中心坐标
            bbox_center = ((min_rect[0, 0] + min_rect[2, 0]) // 2, (min_rect[0, 1] + min_rect[2, 1]) // 2)
            # 在图像上绘制文本
            color_mask = draw_text_on_image(color_mask, caption, bbox_center, color=color_this)
        
        vis_mask = np.concatenate((img, color_mask), axis=0)
        if args.output:
            os.makedirs(args.output, exist_ok=True)
            if os.path.isdir(args.output):
                assert os.path.isdir(args.output), args.output
                out_filename = os.path.join(args.output+"vis", os.path.basename(path))
            else:
                assert len(args.input) == 1, "Please specify a directory with args.output"
                out_filename = args.output
            os.makedirs(args.output+"vis", exist_ok=True)
            cv2.imwrite(out_filename, vis_mask)
            
        caption_all_image.append(new_captions)
        # 2.4 把caption特征保存下来
        caption_fts = sbert_model.encode(new_captions, convert_to_tensor=True, device="cuda")
        caption_fts = caption_fts / caption_fts.norm(dim=-1, keepdim=True)
        caption_fts = caption_fts.detach().cpu().numpy()
        capfeat_all_image.append(caption_fts)
        
    # # 清空显存
    # del captions,caption_fts,tap_model,sbert_model,capfeat_all_image
    torch.cuda.empty_cache()
    ######################################################
    ############ 三、使用clip为边界框生成clip特征 ##########
    ######################################################
    # CLIP编码器
    clip_model, preprocess = clip.load("ViT-B/32", device="cuda")
    # 弃用，不知道为啥OOM
    # clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    #     "ViT-H-14", "laion2b_s32b_b79k"
    # )
    # clip_model = clip_model.to("cuda")
    # clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
    # 所有图像的CLIP特征列表保存下来
    clipfeat_all_image = []
    seg_count = 0
    for path in tqdm(args.input, disable=not args.output):
        seg_count += 1
        if seg_count > seg_count_max:
            break
        print(f"Processing frame id = {path}")
        img = read_image(path, format="BGR")
        min_rects = bbox_all_image[seg_count-1]
        image = Image.fromarray(img)
        padding = 20  # CLIP可以物体往外扩N个像素
        image_feats = []
        print(len(min_rects))
        for j in range(len(min_rects)):
            x_min, y_min, x_max, y_max = min_rects[j][0, 0], min_rects[j][0, 1], min_rects[j][2, 0], min_rects[j][2, 1]
            # 检查并调整填充以避免超出图像边界
            image_width, image_height = image.size
            left_padding = min(padding, x_min)
            top_padding = min(padding, y_min)
            right_padding = min(padding, image_width - x_max)
            bottom_padding = min(padding, image_height - y_max)
            # 应用调整后的填充
            x_min -= left_padding
            y_min -= top_padding
            x_max += right_padding
            y_max += bottom_padding
            cropped_image = image.crop((x_min, y_min, x_max, y_max))
            # cv2.imshow('Image', np.array(cropped_image))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # 从裁剪中获取剪辑的预处理图像
            preprocessed_image = preprocess(cropped_image).unsqueeze(0).to("cuda")
            with torch.no_grad():
                crop_feat = clip_model.encode_image(preprocessed_image)
            crop_feat /= crop_feat.norm(dim=-1, keepdim=True)
            crop_feat_cpu = crop_feat.detach().cpu().numpy()
            # DEBUG
            # with torch.no_grad():
            #     text_feature = clip_model.encode_text(clip.tokenize("livingroom bedroom").to("cuda"))
            # text_feature /= text_feature.norm(dim=-1, keepdim=True)
            # similarities = torch.cosine_similarity(crop_feat, text_feature, dim=1)
            # print(caption_all_image[seg_count-1][j])
            # print(similarities)
            image_feats.append(crop_feat_cpu)
        clipfeat_all_image.append(image_feats)

    
    ###################################################################
    ####################### 四、初始Mask结果与特征保存 ##################
    ###################################################################
    # 创建一个字典保存所有列表
    # mask_all_image--列表每个元素为图像上所有的mask列表 列表元素为numpy bool型数组
    # bbox_all_image--列表每个元素为图像上所有的bbox列表 列表元素为4*2维的int型数组
    # caption_all_image--列表每个元素为图像上所有的mask对应的caption text列表 列表元素为str型列表
    # capfeat和clipfeat同理 每个元素为图像上所有的mask对应的captionfeat or clipfeat 列表元素为numpy数组
    all_data = {
        "mask": mask_all_image,
        "bbox": bbox_all_image,
        "caption": caption_all_image,
        "capfeat": capfeat_all_image,
        "clipfeat": clipfeat_all_image
    }
    # 保存字典到文件
    out_filename = args.output+"mask_init_all.pkl"
    with open(out_filename, 'wb') as f:
        pickle.dump(all_data, f)