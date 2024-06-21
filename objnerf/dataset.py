import imgviz
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import cv2
import os
from utils import enlarge_bbox, get_bbox2d, get_bbox2d_batch, box_filter
import glob
from torchvision import transforms
import image_transforms
import open3d
import time
import pickle
import math
from natsort import natsorted
import torch.nn.functional as F


def init_loader(cfg, multi_worker=True):
    if cfg.dataset_format == "Replica":
        dataset = Replica(cfg)
    elif cfg.dataset_format == "ScanNet":
        dataset = ScanNet(cfg)
    else:
        print("Dataset format {} not found".format(cfg.dataset_format))
        exit(-1)

    # init dataloader
    if multi_worker:
        # multi worker loader，并行加载，效率高吧
        dataloader = DataLoader(dataset, batch_size=None, shuffle=False, sampler=None,
                                batch_sampler=None, num_workers=4, collate_fn=None,
                                pin_memory=True, drop_last=False, timeout=0,
                                worker_init_fn=None, generator=None, prefetch_factor=2,
                                persistent_workers=True)
    else:
        # single worker loader
        dataloader = DataLoader(dataset, batch_size=None, shuffle=False, sampler=None,
                                     batch_sampler=None, num_workers=0)

    return dataloader

class Replica(Dataset):
    def __init__(self, cfg):
        self.imap_mode = cfg.imap_mode
        # 数据采样，不是2000帧全部用
        self.satrt = cfg.start
        self.stride = cfg.stride
        self.root_dir = cfg.dataset_dir
        traj_file = os.path.join(self.root_dir, "traj_w_c.txt")
        self.Twc = np.loadtxt(traj_file, delimiter=" ").reshape([-1, 4, 4])
        obj_clipfeat_file = os.path.join(self.root_dir, "object_clipfeat.pkl")
        self.obj_clipfeat = pickle.load(open(obj_clipfeat_file, 'rb'))
        obj_capfeat_file = os.path.join(self.root_dir, "object_capfeat.pkl")
        self.obj_capfeat = pickle.load(open(obj_capfeat_file, 'rb'))
        self.depth_transform = transforms.Compose(
            [image_transforms.DepthScale(cfg.depth_scale),
             image_transforms.DepthFilter(cfg.max_depth)])
        # 是否执行部件级理解
        self.part_mode = cfg.part_mode
        # background semantic classes: undefined--1, undefined-0 beam-5 blinds-12 curtain-30 ceiling-31 floor-40 pillar-60 vent-92 wall-93 wall-plug-95 window-97 rug-98
        # 使用语义id号来判断是不是属于背景的
        # self.background_cls_list = [5,12,30,31,40,60,92,93,95,97,98,79]
        # 把墙视为背景，不确定的都是属于墙好了
        # self.background_cls_list = [93]
        # 我们自己的数据是1
        self.background_cls_list = [1]
        
        # Not sure: door-37 handrail-43 lamp-47 pipe-62 rack-66 shower-stall-73 stair-77 switch-79 wall-cabinet-94 picture-59
        # 这个scale是相对尺寸，bbox往外两边各扩大1/10
        self.bbox_scale = 0.2  # 1 #1.5 0.9== s=1/9, s=0.2
        

    def __len__(self):
        length = int((len(os.listdir(os.path.join(self.root_dir, "depth")))-self.satrt)/self.stride)
        print("The length of dataset is:", length)
        return length

    def __getitem__(self, idx):
        # 按照尺度，确定该处理哪一帧了
        idx = int(self.satrt+idx*self.stride)
        idx_no = int(idx/10)
        bbox_dict = {}
        obj_clipfeat_dict = {}
        obj_capfeat_dict = {}
        rgb_file = os.path.join(self.root_dir, "rgb", "rgb_" + str(idx) + ".png")
        depth_file = os.path.join(self.root_dir, "depth", "depth_" + str(idx) + ".png")
        inst_file = os.path.join(self.root_dir, "instance_our", "semantic_instance_" + str(idx_no) + ".png")
        obj_file = os.path.join(self.root_dir, "class_our", "semantic_class_" + str(idx_no) + ".png")
        # 部件级特征的文件
        if self.part_mode:
            # part_file = os.path.join(self.root_dir, "partlevel", str(idx) + ".pt")
            # part_feat = torch.load(part_file)
            part_file = os.path.join(self.root_dir, "partlevel", str(idx) + ".npy")
            part_feat = np.load(part_file).transpose((1, 0, 2))
            part_feat = torch.tensor(part_feat)
        
        obj_clipfeat_this = self.obj_clipfeat[idx_no]
        obj_capfeat_this = self.obj_capfeat[idx_no]
        depth = cv2.imread(depth_file, -1).astype(np.float32).transpose(1,0)
        image = cv2.imread(rgb_file).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(1,0,2)
        # 语义类别
        obj = cv2.imread(obj_file, cv2.IMREAD_UNCHANGED).astype(np.int32).transpose(1,0)   # uint16 -> int32
        # 实例分割
        inst = cv2.imread(inst_file, cv2.IMREAD_UNCHANGED).astype(np.int32).transpose(1,0)  # uint16 -> int32
        # 把0改成-1吧，不能当做背景，而是未知
        obj[obj==0]=-1
        inst[inst==0]=-1

        bbox_scale = self.bbox_scale

        if self.imap_mode:
            obj = np.zeros_like(obj)
        else:
            obj_ = np.zeros_like(obj)
            inst_list = []
            batch_masks = []
            # 提取物体的mask们
            for inst_id in np.unique(inst):
                if inst_id == -1:
                    continue
                inst_mask = inst == inst_id
                # if np.sum(inst_mask) <= 2000: # too small    20  400
                #     continue
                # 判断这个mask物体的语义类别
                sem_cls = np.unique(obj[inst_mask])  # sem label, only interested obj
                assert sem_cls.shape[0] != 0
                # 如果是背景，不做处理
                if sem_cls in self.background_cls_list:
                    continue
                # 否则的话，就记录下来这个物体的mask
                obj_mask = inst == inst_id
                # 对于我们的数据集，需要腐蚀一些防止落在外面
                # obj_mask = cv2.erode(obj_mask.astype(np.uint8), np.ones((5, 5)), iterations=1).astype(bool)
                # 物体mask的列表
                batch_masks.append(obj_mask)
                # 把id放进去了，一会下面读取作为实例id
                inst_list.append(inst_id)
            if len(batch_masks) > 0:
                batch_masks = torch.from_numpy(np.stack(batch_masks))
                # 从mask的边界得到bbox
                cmins, cmaxs, rmins, rmaxs = get_bbox2d_batch(batch_masks)
                for i in range(batch_masks.shape[0]):
                    w = rmaxs[i] - rmins[i]
                    h = cmaxs[i] - cmins[i]
                    if w <= 10 or h <= 10:  # too small  太小东西当做未知吧
                        # print(idx, inst_list[i],"too small")
                        continue
                    # 再把bbox向外扩张一点
                    bbox_enlarged = enlarge_bbox([rmins[i], cmins[i], rmaxs[i], cmaxs[i]], scale=bbox_scale,
                                                 w=obj.shape[1], h=obj.shape[0])
                    # inst_list.append(inst_id)
                    inst_id = inst_list[i]
                    obj_[batch_masks[i]] = 1
                    # bbox_dict.update({int(inst_id): torch.from_numpy(np.array(bbox_enlarged).reshape(-1))}) # batch format
                    # 放到字典里面去，当前帧的实例id对应着一个bbox
                    bbox_dict.update({inst_id: torch.from_numpy(np.array([bbox_enlarged[1], bbox_enlarged[3], bbox_enlarged[0], bbox_enlarged[2]]))})  # bbox order
                    # 把特征写入字典
                    obj_clipfeat_dict.update({inst_id: obj_clipfeat_this[inst_id]})
                    obj_capfeat_dict.update({inst_id: obj_capfeat_this[inst_id]})
            # 背景设置为0
            for sem_cls in self.background_cls_list:
                inst[inst == sem_cls] = 0 
            # 没有物体mask的地方都视为啥也不知道
            inst[(obj_==0) & (inst!=0)] = -1
            obj = inst
            # obj_ids = np.unique(obj)
            # print(idx, "obj_ids", obj_ids)
        # 背景的id为0，bbox是整个图片的大小，对我们来说是墙面，1
        
        if 1 in obj_clipfeat_this:
            bbox_dict.update({0: torch.from_numpy(np.array([int(0), int(obj.shape[0]), 0, int(obj.shape[1])]))})  # bbox order
            obj_clipfeat_dict.update({0: obj_clipfeat_this[1]})
            obj_capfeat_dict.update({0: obj_capfeat_this[1]})
        # 可以是真值位姿，也可以是ORB-SLAM等得到的
        T = self.Twc[idx]   # could change to ORB-SLAM pose or else
        # 物体的姿态，就是单位阵，有可能是动态的话，也要确定以防伪影
        T_obj = np.eye(4)   # obj pose, if dynamic
        # 输出原始rgb图像，深度图像，位姿，物体位姿，整个图像上每个物体的mask，每个物体的bbox词典，当前帧id
        sample = {"image": image, "depth": depth, "T": T, "T_obj": T_obj,
                  "obj": obj, "bbox_dict": bbox_dict, "frame_id": idx,
                  "obj_clip":obj_clipfeat_dict, "obj_cap": obj_capfeat_dict}
        # 如果执行部件级理解，加入partfeat
        if self.part_mode:
            sample.update({"part_feat": part_feat})
        # 没有文件，则报错哦
        if image is None or depth is None:
            print(rgb_file)
            print(depth_file)
            raise ValueError
        # 需要把深度图像转换一下，成为m，并设置最大深度值
        if self.depth_transform:
            sample["depth"] = self.depth_transform(sample["depth"])
        return sample

class ScanNet(Dataset):
    def __init__(self, cfg):
        self.imap_mode = cfg.imap_mode
        # 数据采样，不是2000帧全部用
        self.satrt = cfg.start
        self.stride = cfg.stride
        self.root_dir = cfg.dataset_dir
        self.color_paths = sorted(glob.glob(os.path.join(
            self.root_dir, 'color', '*.jpg')), key=lambda x: int(os.path.basename(x)[:-4]))
        self.depth_paths = sorted(glob.glob(os.path.join(
            self.root_dir, 'depth', '*.png')), key=lambda x: int(os.path.basename(x)[:-4]))
        # self.inst_paths = sorted(glob.glob(os.path.join(
        #     self.root_dir, 'instance-filt', '*.png')), key=lambda x: int(os.path.basename(x)[:-4])) # instance-filt
        # self.sem_paths = sorted(glob.glob(os.path.join(
        #     self.root_dir, 'label-filt', '*.png')), key=lambda x: int(os.path.basename(x)[:-4]))  # label-filt
        # self.inst_paths = sorted(glob.glob(os.path.join(
        #     self.root_dir, "instance_our", "*.png")), key=lambda x: int(os.path.basename(x)[:-4])) # instance-filt
        # self.sem_paths = sorted(glob.glob(os.path.join(
        #     self.root_dir, "class_our", "*.png")), key=lambda x: int(os.path.basename(x)[:-4])) # instance-filt
        
        self.inst_paths = natsorted(glob.glob(os.path.join(self.root_dir, "instance_our", "*.png")))
        self.sem_paths = natsorted(glob.glob(os.path.join(self.root_dir, "class_our", "*.png")))
        
        # 参数文件
        obj_clipfeat_file = os.path.join(self.root_dir, "object_clipfeat.pkl")
        self.obj_clipfeat = pickle.load(open(obj_clipfeat_file, 'rb'))
        obj_capfeat_file = os.path.join(self.root_dir, "object_capfeat.pkl")
        self.obj_capfeat = pickle.load(open(obj_capfeat_file, 'rb'))
        # # pose文件需要单独加载
        # self.load_poses(os.path.join(self.root_dir, 'pose'))
        
        traj_file = os.path.join(self.root_dir, "traj_w_c.txt")
        self.Twc = np.loadtxt(traj_file, delimiter=" ").reshape([-1, 4, 4])
        
        self.n_img = len(self.color_paths)
        self.depth_transform = transforms.Compose(
            [image_transforms.DepthScale(cfg.depth_scale),
             image_transforms.DepthFilter(cfg.max_depth)])
        # self.rgb_transform = rgb_transform
        self.W = cfg.W
        self.H = cfg.H
        self.fx = cfg.fx
        self.fy = cfg.fy
        self.cx = cfg.cx
        self.cy = cfg.cy
        self.intrinsic_open3d = open3d.camera.PinholeCameraIntrinsic(
            width=self.W,
            height=self.H,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
        )
        # 设置最小像素数，太小的bbox的物体就不要了
        self.min_pixels = 1500
        # from scannetv2-labels.combined.tsv
        #1-wall, 3-floor, 16-window, 41-ceiling, 232-light switch   0-unknown? 21-pillar 161-doorframe, shower walls-128, curtain-21, windowsill-141
        # scannet数据集中的背景类别id
        # self.background_cls_list = [-1, 0, 1, 3, 16, 41, 232, 21, 161, 128, 21]
        self.background_cls_list = [1]
        self.bbox_scale = 0.2
        self.inst_dict = {}
        # 是否执行部件级理解
        self.part_mode = cfg.part_mode
        self.part_down = cfg.part_down

    def load_poses(self, path):
        self.poses = []
        pose_paths = sorted(glob.glob(os.path.join(path, '*.txt')),
                            key=lambda x: int(os.path.basename(x)[:-4]))
        for pose_path in pose_paths:
            with open(pose_path, "r") as f:
                lines = f.readlines()
            ls = []
            for line in lines:
                l = list(map(float, line.split(' ')))
                ls.append(l)
            c2w = np.array(ls).reshape(4, 4)
            self.poses.append(c2w)

    def __len__(self):
        length = math.ceil((self.n_img-self.satrt)/self.stride)
        print("The length of dataset is:", length)
        return length

    def __getitem__(self, index):
        # 按照尺度，确定该处理哪一帧了
        index = int(self.satrt+index*self.stride)
        index_no = int(index/10)
        bbox_scale = self.bbox_scale
        color_path = self.color_paths[index]
        depth_path = self.depth_paths[index]
        inst_path = self.inst_paths[index_no]
        sem_path = self.sem_paths[index_no]
        obj_clipfeat_this = self.obj_clipfeat[index_no]
        obj_capfeat_this = self.obj_capfeat[index_no]
        color_data = cv2.imread(color_path).astype(np.uint8)
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB).transpose(1,0,2)
        depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32).transpose(1,0)
        depth_data = np.nan_to_num(depth_data, nan=0.)
        T = self.Twc[index]
        # if self.poses is not None:
        #     T = self.poses[index]
        #     if np.any(np.isinf(T)):
        #         if index + 1 == self.__len__():
        #             print("pose inf!")
        #             return None
        #         return self.__getitem__(index + 1)
        
        # 部件级特征的文件
        if self.part_mode:
            # part_file = os.path.join(self.root_dir, "partlevel", str(idx) + ".pt")
            # part_feat = torch.load(part_file)
            part_file = os.path.join(self.root_dir, "partlevel", str(index) + ".npy")
            part_feat = np.load(part_file).transpose((1, 0, 2))
            part_feat = torch.tensor(part_feat)
            
            if self.part_down == 10:
                # 降采样
                part_feat = part_feat.permute(2, 0, 1).unsqueeze(0)
                part_feat = F.interpolate(part_feat, scale_factor=0.5, mode='bilinear', align_corners=False)
                part_feat = part_feat.squeeze(0).permute(1, 2, 0)

        H, W = depth_data.shape
        color_data = cv2.resize(color_data, (W, H), interpolation=cv2.INTER_LINEAR)
        # 深度图像转换一下
        if self.depth_transform:
            depth_data = self.depth_transform(depth_data)
        bbox_dict = {}
        obj_clipfeat_dict = {}
        obj_capfeat_dict = {}
        if self.imap_mode:
            inst_data = np.zeros_like(depth_data).astype(np.int32)
        else:
            # inst_data = cv2.imread(inst_path, cv2.IMREAD_UNCHANGED)
            # inst_data = cv2.resize(inst_data, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int32)
            # sem_data = cv2.imread(sem_path, cv2.IMREAD_UNCHANGED)#.astype(np.int32)
            # sem_data = cv2.resize(sem_data, (W, H), interpolation=cv2.INTER_NEAREST)
            # 语义类别
            sem_data = cv2.imread(sem_path, cv2.IMREAD_UNCHANGED).astype(np.int32).transpose(1,0)   # uint16 -> int32
            # 实例分割
            inst_data = cv2.imread(inst_path, cv2.IMREAD_UNCHANGED).astype(np.int32).transpose(1,0)  # uint16 -> int32

            # # 实例id都+1，把0空出来作为背景
            # inst_data += 1  # shift from 0->1 , 0 is for background
            
            # 把0改成-1吧，不能当做背景，而是未知
            inst_data[inst_data==0]=-1
            sem_data[sem_data==0]=-1

        if self.imap_mode:
            obj = np.zeros_like(inst_data)
        else:
            obj_ = np.zeros_like(inst_data)
            inst_list = []
            batch_masks = []
            # 提取物体的mask们
            for inst_id in np.unique(inst_data):
                if inst_id == -1:
                    continue
                inst_mask = inst_data == inst_id
                # if np.sum(inst_mask) <= 2000: # too small    20  400
                #     continue
                # 判断这个mask物体的语义类别
                sem_cls = np.unique(inst_data[inst_mask])  # sem label, only interested obj
                assert sem_cls.shape[0] != 0
                # 如果是背景，不做处理
                if sem_cls in self.background_cls_list:
                    continue
                # 否则的话，就记录下来这个物体的mask
                obj_mask = inst_data == inst_id
                # 对于我们的数据集，需要腐蚀一些防止落在外面
                # obj_mask = cv2.erode(obj_mask.astype(np.uint8), np.ones((5, 5)), iterations=1).astype(bool)
                # 物体mask的列表
                batch_masks.append(obj_mask)
                # 把id放进去了，一会下面读取作为实例id
                inst_list.append(inst_id)
            if len(batch_masks) > 0:
                batch_masks = torch.from_numpy(np.stack(batch_masks))
                # 从mask的边界得到bbox
                cmins, cmaxs, rmins, rmaxs = get_bbox2d_batch(batch_masks)
                for i in range(batch_masks.shape[0]):
                    w = rmaxs[i] - rmins[i]
                    h = cmaxs[i] - cmins[i]
                    if w <= 10 or h <= 10:  # too small  太小东西当做未知吧
                        # print(idx, inst_list[i],"too small")
                        continue
                    # 再把bbox向外扩张一点
                    bbox_enlarged = enlarge_bbox([rmins[i], cmins[i], rmaxs[i], cmaxs[i]], scale=bbox_scale,
                                                 w=inst_data.shape[1], h=inst_data.shape[0])
                    # inst_list.append(inst_id)
                    inst_id = inst_list[i]
                    obj_[batch_masks[i]] = 1
                    # bbox_dict.update({int(inst_id): torch.from_numpy(np.array(bbox_enlarged).reshape(-1))}) # batch format
                    # 放到字典里面去，当前帧的实例id对应着一个bbox
                    bbox_dict.update({inst_id: torch.from_numpy(np.array([bbox_enlarged[1], bbox_enlarged[3], bbox_enlarged[0], bbox_enlarged[2]]))})  # bbox order
                    # 把特征写入字典
                    obj_clipfeat_dict.update({inst_id: obj_clipfeat_this[inst_id]})
                    obj_capfeat_dict.update({inst_id: obj_capfeat_this[inst_id]})
            # 背景设置为0
            for sem_cls in self.background_cls_list:
                inst_data[inst_data == sem_cls] = 0 
            # 没有物体mask的地方都视为啥也不知道，包括两部分，一个是没有前景，另一个是没有物体
            inst_data[(obj_==0) & (inst_data!=0)] = -1
            obj = inst_data
            # obj_ids = np.unique(obj)
            # print(idx, "obj_ids", obj_ids)
        # 背景的id为0，bbox是整个图片的大小，对我们来说是墙面，1
        # 如果看得到墙面的话，加进去
        if 1 in obj_clipfeat_this:
            bbox_dict.update({0: torch.from_numpy(np.array([int(0), int(obj.shape[0]), 0, int(obj.shape[1])]))})  # bbox order
            obj_clipfeat_dict.update({0: obj_clipfeat_this[1]})
            obj_capfeat_dict.update({0: obj_capfeat_this[1]})
        # 物体的姿态，就是单位阵，有可能是动态的话，也要确定以防伪影
        T_obj = np.eye(4)   # obj pose, if dynamic
        # 输出原始rgb图像，深度图像，位姿，物体位姿，整个图像上每个物体的mask，每个物体的bbox词典，当前帧id
        sample = {"image": color_data, "depth": depth_data, "T": T, "T_obj": T_obj,
                  "obj": obj, "bbox_dict": bbox_dict, "frame_id": index,
                  "obj_clip":obj_clipfeat_dict, "obj_cap": obj_capfeat_dict}
        # 如果执行部件级理解，加入partfeat
        if self.part_mode:
            sample.update({"part_feat": part_feat})
        
        # # 可视化一下
        # import distinctipy
        # import matplotlib.pyplot as plt
        # unique_labels = np.unique(obj)
        # unique_labels = unique_labels[unique_labels != -1]
        # colors = distinctipy.get_colors(len(unique_labels), pastel_factor=1)
        # color_map = {label: color for label, color in zip(unique_labels, colors)}
        # obj_rgb = np.zeros((obj.shape[0], obj.shape[1], 3), dtype=np.uint8)
        # for label, color in color_map.items():
        #     obj_rgb[obj == label] = np.array(color)*255
        # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        # axes[0].imshow(color_data)
        # axes[0].set_title('Original Image')
        # axes[0].axis('off')
        # axes[1].imshow(obj_rgb)
        # axes[1].set_title('Instance Visualization')
        # axes[1].axis('off')
        # plt.show()
        if color_data is None or depth_data is None:
            print(color_path)
            print(depth_path)
            raise ValueError
        return sample
    

