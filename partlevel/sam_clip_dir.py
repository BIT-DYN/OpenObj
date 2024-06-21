import numpy as np
import cv2
import argparse
import torch
import clip
import sys
sys.path.append("/home/dyn/multimodal/Grounded-Segment-Anything/segment_anything")
from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
from segment_anything import sam_model_registry
from PIL import Image
import matplotlib.pyplot as plt
import os
from natsort import natsorted
from tqdm import tqdm


#传参数
def args_getter():
    parser = argparse.ArgumentParser(description='Sam embedded with CLIP')
    parser.add_argument(
        "--input_image",
        nargs="+",
        help="A list of space separated input rgb images; "
    )
    parser.add_argument('--output_dir',type=str,help='the results output directory')
    parser.add_argument('--down_sample',type=int,help='the results output directory')
    args = parser.parse_args()
    return args

#对每个图片进行mask提取
def mask_getter(image:np.ndarray):
    sam_checkpoint = "/data/dyn/weights/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)
    return masks

#得到扩大后的bbox坐标
def bbox_getter(bbox_original:list,height:int,width:int):
    x_min,y_min = bbox_original[0:2]
    width_mask,height_mask = bbox_original[2:]
    x_max = x_min + bbox_original[2]
    y_max = y_min + bbox_original[3]
    width_mask_new = round(bbox_original[2]*1.3)
    height_mask_new = round(bbox_original[3]*1.3)
    width_inc = round((width_mask_new-width_mask)/2)
    height_inc = round((height_mask_new-height_mask)/2)
    left_inc = min(width_inc,x_min)
    top_inc = min(height_inc,y_min)
    right_inc = min(width_inc,width-x_max)
    bottom_inc = min(height_inc,height-y_max)
    x_min -= left_inc
    y_min -= top_inc
    x_max += right_inc
    y_max += bottom_inc
    return [x_min,y_min,x_max,y_max]

def main(args:argparse.Namespace) ->None:
    #加载CLIP模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    # 按照阈值筛除
    Threshold = 0.9
    input_image = args.input_image
    skip = 10
    input_image = natsorted(input_image)
    input_image = input_image[0:-1:skip]
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print("Start!!!")
    for idx, file in tqdm(enumerate(input_image)):
        
        # if idx < 225:
        #     continue
        
        ##开始对输入图片进行分割并提取clip##
        image_original = cv2.imread(file)
        image_original = cv2.cvtColor(image_original,cv2.COLOR_BGR2RGB)
        height,width = image_original.shape[:2]
        masks = mask_getter(image_original)#得到图片的mask信息
        # 尝试使用predicted_iou过滤下没用的物体
        print("Before mask filtering:", len(masks))
        post_masks = []
        for mask in masks:
            if mask['predicted_iou'] > Threshold:
                post_masks.append(mask)
        print("After mask filtering: ", len(post_masks))
        #对图片按bbox进行分割得到每个mask的clip特征
        mask_feature_all=[]
        bbox_all=[]
        for i,masks_data in enumerate(masks):
            print(f'start generate mask{i} features...')
            #把bbox扩大一定的大小
            bbox = bbox_getter(masks_data['bbox'],height,width)
            bbox_all.append(bbox)
            #根据bbox分割图片
            crop_image = image_original[bbox[1]:bbox[3],bbox[0]:bbox[2]]
            # crop_image_1=cv2.cvtColor(crop_image,cv2.COLOR_RGB2BGR)
            # cv2.imwrite(output_dir+'/'+f'{i}.png', crop_image_1)
            #将cv读取的图片转化成PIL，方便clip操作
            pil_crop_image = Image.fromarray(crop_image)
            clip_crop_image = preprocess(pil_crop_image).unsqueeze(0).to(device)
            with torch.no_grad():
                mask_feature = clip_model.encode_image(clip_crop_image)
            mask_feature_all.append(mask_feature[0])
        
        # 生成逐像素CLIP特征图
        down_sample = args.down_sample
        per_pixel_feature=torch.zeros(int(height/down_sample),int(width/down_sample),mask_feature_all[0].shape[0],device=device)
        per_pixel_weight_sum=torch.zeros(int(height/down_sample),int(width/down_sample),device=device)
        
        for i,masks_data in enumerate(masks):
            mask = masks_data['segmentation']
            # 将mask也降维放进去
            mask = mask[::down_sample, ::down_sample]
            mask = torch.tensor(mask).to(device).bool()
            mask_feature_this = mask_feature_all[i]*masks_data['stability_score']
            per_pixel_feature[mask.bool()] = mask_feature_this.float()
            per_pixel_weight_sum[mask] += masks_data['stability_score']
        per_pixel_weight_sum = per_pixel_weight_sum.unsqueeze(-1).expand_as(per_pixel_feature)
        
        # 存一下生成的逐像素点的特征张量
        # # torch保存
        # torch.save(per_pixel_feature, output_dir+'/'+str(idx*skip)+'.pt')
        # print("save successful!", output_dir)
        # np保存
        np.save(output_dir + '/' + str(idx * skip) + '.npy', per_pixel_feature.cpu().numpy())
        print("save successful!", output_dir)
    
        # # debug测试相似度计算采用的
        # text_query = ['cabinet']
        # with torch.no_grad():
        #     text_feature = clip_model.encode_text(clip.tokenize(text_query).to(device))
        # text_feature/=text_feature.norm(p=2,dim=-1,keepdim=True)
        # similarities_matrix = torch.zeros(per_pixel_feature.shape[0], per_pixel_feature.shape[1], device=device)
        # for jj in range(per_pixel_feature.shape[0]):
        #     for ii in range(per_pixel_feature.shape[1]):
        #         # 获取当前像素点的特征向量
        #         pixel_feature = per_pixel_feature[jj, ii]  # 假设特征张量的最后一维是特征向量
        #         # 计算当前像素点特征与文本特征之间的余弦相似度
        #         similarity = torch.cosine_similarity(pixel_feature, text_feature, dim=-1)
        #         # 将相似性存储到相似性矩阵中
        #         similarities_matrix[jj, ii] = similarity
        # similarities_matrix = similarities_matrix.cpu().numpy()
        # # 热力图
        # cmap = plt.cm.jet
        # plt.figure()
        # plt.imshow(similarities_matrix, cmap=cmap, aspect='auto')    
        # plt.colorbar(label='Similarity')
        # plt.show()    

if __name__ == '__main__':
    args = args_getter()
    main(args)



    