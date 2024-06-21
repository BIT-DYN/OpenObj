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
import shutil
import open3d as o3d
import cv2
import render_rays
import matplotlib.pyplot as plt
from datetime import datetime
from utils import performance_measure
import yaml
import clip
import distinctipy
# sbert模型
from sentence_transformers import SentenceTransformer, util
from utils import get_majority_cluster_mean
import csv

if __name__ == "__main__":
    #############################################
    # init config
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # setting params
    parser = argparse.ArgumentParser(description='Model training for single GPU')
    parser.add_argument('--logdir', default='./logs/debug',
                        type=str)
    parser.add_argument('--config',
                        default='./configs/Replica/config_replica_room0_vMAP.json',
                        type=str)
    args = parser.parse_args()

    log_dir = args.logdir
    config_file = args.config
    os.makedirs(log_dir, exist_ok=True)  # saving logs
    shutil.copy(config_file, log_dir)
    cfg = Config(config_file)       # config params
    n_sample_per_step = cfg.n_per_optim
    n_sample_per_step_bg = cfg.n_per_optim_bg

    # 可视化的一些参数，固定了参数视角为z轴10米
    if cfg.if_vis:
        vis3d = open3d.visualization.Visualizer()
        vis3d.create_window(window_name="3D mesh vis",
                            width=cfg.W,
                            height=cfg.H,
                            left=600, top=50)
        view_ctl = vis3d.get_view_control()
        view_ctl.set_constant_z_far(10.)

    # 数据集中相机的内参
    cam_info = cameraInfo(cfg)
    intrinsic_open3d = open3d.camera.PinholeCameraIntrinsic(
        width=cfg.W,
        height=cfg.H,
        fx=cfg.fx,
        fy=cfg.fy,
        cx=cfg.cx,
        cy=cfg.cy)
    
    # 初始化物体的列表
    obj_dict = {}   # 只包含物体
    vis_dict = {}   # 还包含背景

    # init for training
    AMP = False
    if AMP:
        scaler = torch.cuda.amp.GradScaler()  # amp https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/
    # 设置好优化器的参数
    optimiser = torch.optim.AdamW([torch.autograd.Variable(torch.tensor(0))], lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    #############################################
    # init data stream
    if not cfg.live_mode:
        # 加载数据集
        # load dataset
        dataloader = dataset.init_loader(cfg)
        dataloader_iterator = iter(dataloader)
        dataset_len = len(dataloader)
    else:
        dataset_len = 1000000
        # # init ros node
        # torch.multiprocessing.set_start_method('spawn')  # spawn
        # import ros_nodes
        # track_to_map_Buffer = torch.multiprocessing.Queue(maxsize=5)
        # # track_to_vis_T_WC = torch.multiprocessing.Queue(maxsize=1)
        # kfs_que = torch.multiprocessing.Queue(maxsize=5)  # to store one more buffer
        # track_p = torch.multiprocessing.Process(target=ros_nodes.Tracking,
        #                                              args=(
        #                                              (cfg), (track_to_map_Buffer), (None),
        #                                              (kfs_que), (True),))
        # track_p.start()


    # 初始化vmap，这里用来存网络参数的，没有背景物体的诶
    fc_models, pe_models = [], []
    scene_bg = None
    
    # 加载语义参数配置文件
    class_names = []
    if cfg.dataset_format == "Replica":
        parter_dir = os.path.dirname(os.path.dirname(os.path.dirname(cfg.dataset_dir)))
        with open(parter_dir+'/render_config.yaml', 'r') as file:
            data = yaml.safe_load(file)
            # 提取classes中所有children的name
        class_names = [item['name']  for item in data['classes']]
    elif cfg.dataset_format == "ScanNet":
        parter_dir = os.path.dirname(cfg.dataset_dir)
        file_path = "/data/dyn/ScanNet/scans/scannetv2-labels.combined.tsv"
        # 读取文件并提取信息
        with open(file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile, delimiter='\t')
            for row in reader:
                nyu40class = row['nyu40class']
                if nyu40class not in class_names:
                    class_names.append(nyu40class)
            # # 按照nyu40id的顺序重新排列class_names
            # class_names = [None] * (len(class_names))
            # for row in reader:
            #     nyu40class = row['nyu40class']
            #     nyu40id = int(row['nyu40id'])
            #     class_names[nyu40id-1] = nyu40class
        
    print("class_names: ")
    print(class_names)
    
    clip_model, preprocess = clip.load("ViT-B/32", device="cuda")
    with torch.no_grad():
        class_clipfeat = clip_model.encode_text(clip.tokenize(class_names).to("cuda"))
    class_clipfeat /= class_clipfeat.norm(dim=-1, keepdim=True)
    class_clipfeat = class_clipfeat.cpu().numpy()
    print("class_clipfeat:", class_clipfeat.shape)
    
    # SBERT文本编码器
    sbert_model = SentenceTransformer('/home/dyn/multimodal/SBERT/pretrained/model/all-MiniLM-L6-v2')
    class_capfeat = sbert_model.encode(class_names, convert_to_tensor=True, device="cuda")
    class_capfeat /= class_capfeat.norm(dim=-1, keepdim=True)
    class_capfeat = class_capfeat.cpu().numpy()
    print("class_capfeat:", class_capfeat.shape)
    
    # 堆叠为总特征
    class_allfeat = np.hstack((class_clipfeat, class_capfeat))
    print("class_allfeat:", class_allfeat.shape)
    
    # 对于导入的每帧特征，构建一个字典保存在全局，任何物体可以调用
    global_partfeat = None
    
    
    # 正式开始增量式建图
    for frame_id in tqdm(range(dataset_len)):
        print("*********************************************")
        # get new frame data
        with performance_measure(f"getting next data"):
            if not cfg.live_mode:
                # 得到下一帧数据，包括全局的实例id了
                sample = next(dataloader_iterator)
            else:
                pass
        # 这个sample里面的id已经是全局的id了
        if sample is not None:  # new frame
            last_frame_time = time.time()
            # 把当前帧的所有信息加载出来
            with performance_measure(f"Appending data"):
                rgb = sample["image"].to(cfg.data_device)
                depth = sample["depth"].to(cfg.data_device)
                twc = sample["T"].to(cfg.data_device)
                bbox_dict = sample["bbox_dict"]
                obj_clip = sample["obj_clip"]
                obj_cap = sample["obj_cap"]
                part_feat = None
                if "frame_id" in sample.keys():
                    live_frame_id = sample["frame_id"]
                else:
                    live_frame_id = frame_id
                if cfg.part_mode:
                    part_feat = sample["part_feat"].to(cfg.data_device)
                    if global_partfeat is None:
                        global_partfeat = part_feat.unsqueeze(0)
                    else:
                        global_partfeat = torch.cat((global_partfeat, part_feat.unsqueeze(0)), dim=0)
                if not cfg.live_mode:
                    inst = sample["obj"].to(cfg.data_device)
                    obj_ids = torch.unique(inst)
                else:
                    inst_data_dict = sample["obj"]
                    obj_ids = inst_data_dict.keys()
                # append new frame info to objs in current view
                # 对于当前帧的每一个物体，把他加到全局地图
                for obj_id in obj_ids:
                    if obj_id == -1:    # unsured area
                        continue
                    obj_id = int(obj_id)
                    # 把当前物体实例的mask转为state，0是不属于，1就是属于该实例，2则是不确定
                    if not cfg.live_mode:
                        state = torch.zeros_like(inst, dtype=torch.uint8, device=cfg.data_device)
                        state[inst == obj_id] = 1
                        state[inst == -1] = 2
                        # # 可视化一下， 定义颜色映射：黑色、绿色、红色
                        # colors = [(0, 0, 0), (0, 255, 0), (0, 0, 255)]
                        # state_rgb = np.zeros((state.shape[0], state.shape[1], 3), dtype=np.uint8)
                        # for state_val, color in zip([0, 1, 2], colors):
                        #     state_rgb[state.cpu().numpy() == state_val] = color
                        # cv2.imshow('State Visualization', state_rgb)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                    else:
                        inst_mask = inst_data_dict[obj_id].permute(1,0)
                        label_list = torch.unique(inst_mask).tolist()
                        state = torch.zeros_like(inst_mask, dtype=torch.uint8, device=cfg.data_device)
                        state[inst_mask == obj_id] = 1
                        state[inst_mask == -1] = 2
                        
                    # 得到当前帧当前物体的bbox
                    bbox = bbox_dict[obj_id]
                    if obj_id in vis_dict.keys():
                        # 如果这个物体是已经存在的了，就是相当于增加了一个新的帧
                        scene_obj = vis_dict[obj_id]
                        scene_obj.append_keyframe(rgb, depth, state, bbox, twc, live_frame_id, clip_feat = obj_clip[obj_id][0], caption_feat = obj_cap[obj_id])
                    else: 
                        # 如果这个物体是新的物体，则初始化一个新的
                        # 不能超过100个物体哦
                        print("current num is",len(obj_dict.keys()))
                        if len(obj_dict.keys()) >= cfg.max_n_models:
                            print("models full!!!! current num ", len(obj_dict.keys()))
                            continue
                        print("init new obj ", obj_id)
                        # 注意背景的初始化参数是不同的哦
                        if cfg.do_bg and obj_id == 0:  
                            # 如果是背景物体
                            scene_bg = sceneObject(cfg, obj_id, rgb, depth, state, bbox, twc, live_frame_id, clip_feat = obj_clip[obj_id][0], caption_feat = obj_cap[obj_id])
                            # scene_bg.init_obj_center(intrinsic_open3d, depth, state, twc)
                            optimiser.add_param_group({"params": scene_bg.trainer.fc_occ_map.parameters(), "lr": cfg.learning_rate, "weight_decay": cfg.weight_decay})
                            optimiser.add_param_group({"params": scene_bg.trainer.pe.parameters(), "lr": cfg.learning_rate, "weight_decay": cfg.weight_decay})
                            vis_dict.update({obj_id: scene_bg})
                        else:
                            scene_obj = sceneObject(cfg, obj_id, rgb, depth, state, bbox, twc, live_frame_id, clip_feat = obj_clip[obj_id][0], caption_feat = obj_cap[obj_id])
                            # scene_obj.init_obj_center(intrinsic_open3d, depth, state, twc)
                            # obj_dict里面不包含背景
                            obj_dict.update({obj_id: scene_obj})
                            vis_dict.update({obj_id: scene_obj})
                            # params = [scene_obj.trainer.fc_occ_map.parameters(), scene_obj.trainer.pe.parameters()]
                            optimiser.add_param_group({"params": scene_obj.trainer.fc_occ_map.parameters(), "lr": cfg.learning_rate, "weight_decay": cfg.weight_decay})
                            optimiser.add_param_group({"params": scene_obj.trainer.pe.parameters(), "lr": cfg.learning_rate, "weight_decay": cfg.weight_decay})
                            if cfg.training_strategy == "vmap":
                                update_vmap_model = True
                                # 对于vmap的更新是物体级别的，把模型构建好，加入进来，不包含背景的诶
                                fc_models.append(obj_dict[obj_id].trainer.fc_occ_map)
                                pe_models.append(obj_dict[obj_id].trainer.pe)
                        # 可用来计算参数权重的大小
                        # ###################################
                        # # measure trainable params in total
                        # total_params = 0
                        # obj_k = obj_dict[obj_id]
                        # for p in obj_k.trainer.fc_occ_map.parameters():
                        #     if p.requires_grad:
                        #         total_params += p.numel()
                        # for p in obj_k.trainer.pe.parameters():
                        #     if p.requires_grad:
                        #         total_params += p.numel()
                        # print("total param ", total_params)

        # dynamically add vmap
        # 动态地把vmap要更新的参数添加到优化器中，等待优化
        with performance_measure(f"add vmap"):
            if cfg.training_strategy == "vmap" and update_vmap_model == True:
                fc_model, fc_param, fc_buffer = utils.update_vmap(fc_models, optimiser)
                pe_model, pe_param, pe_buffer = utils.update_vmap(pe_models, optimiser)
                update_vmap_model = False


        ##################################################################
        # training data preperation, get training data for all objs
        # 准备训练数据们
        # 深度值
        Batch_N_gt_depth = []
        # RGB值
        Batch_N_gt_rgb = []
        # 有效深度值的mask
        Batch_N_depth_mask = []
        # 属于该物体的mask
        Batch_N_obj_mask = []
        # 输入的点云
        Batch_N_input_pcs = []
        # 采样的光线长度
        Batch_N_sampled_z = []
        if cfg.part_mode:
            # 2D部件级特征
            Batch_N_gt_partfeat = []

        # 正式开始进行采样操作
        with performance_measure(f"Sampling over {len(obj_dict.keys())} objects,"):
            if cfg.do_bg and scene_bg is not None:
                # 背景的采样结果，每次都要对背景进行处理
                gt_rgb, gt_depth, valid_depth_mask, obj_mask, input_pcs, sampled_z, gt_partfeat \
                    = scene_bg.get_training_samples(cfg.n_iter_per_frame * cfg.win_size_bg, cfg.n_samples_per_frame_bg,
                                                    cam_info.rays_dir_cache, global_partfeat)
                bg_gt_depth = gt_depth.reshape([gt_depth.shape[0] * gt_depth.shape[1]])
                bg_gt_rgb = gt_rgb.reshape([gt_rgb.shape[0] * gt_rgb.shape[1], gt_rgb.shape[2]])
                if cfg.part_mode:
                    bg_gt_partfeat = gt_partfeat.reshape([gt_partfeat.shape[0] * gt_partfeat.shape[1], gt_partfeat.shape[2]])
                bg_valid_depth_mask = valid_depth_mask
                bg_obj_mask = obj_mask
                # 得到有关背景物体的点云
                bg_input_pcs = input_pcs.reshape(
                    [input_pcs.shape[0] * input_pcs.shape[1], input_pcs.shape[2], input_pcs.shape[3]])
                # 采样的点云的深度值
                bg_sampled_z = sampled_z.reshape([sampled_z.shape[0] * sampled_z.shape[1], sampled_z.shape[2]])
            # 再对所有的前景物体都进行处理
            for obj_id, obj_k in obj_dict.items():
                # 得到每个物体的采样结果
                gt_rgb, gt_depth, valid_depth_mask, obj_mask, input_pcs, sampled_z, gt_partfeat \
                    = obj_k.get_training_samples(cfg.n_iter_per_frame * cfg.win_size, cfg.n_samples_per_frame,
                                                 cam_info.rays_dir_cache, global_partfeat)
                # 100*24*9
                # print(input_pcs.shape)
                # 融合前两维度，也就是不管像素是在哪一帧采样的，得到的数量应该是sample_per_frame*num_per_frame
                Batch_N_gt_depth.append(gt_depth.reshape([gt_depth.shape[0] * gt_depth.shape[1]]))
                Batch_N_gt_rgb.append(gt_rgb.reshape([gt_rgb.shape[0] * gt_rgb.shape[1], gt_rgb.shape[2]]))
                Batch_N_depth_mask.append(valid_depth_mask)
                Batch_N_obj_mask.append(obj_mask)
                Batch_N_input_pcs.append(input_pcs.reshape([input_pcs.shape[0] * input_pcs.shape[1], input_pcs.shape[2], input_pcs.shape[3]]))
                Batch_N_sampled_z.append(sampled_z.reshape([sampled_z.shape[0] * sampled_z.shape[1], sampled_z.shape[2]]))
                if cfg.part_mode:
                    Batch_N_gt_partfeat.append(gt_partfeat.reshape([gt_partfeat.shape[0]*gt_partfeat.shape[1],gt_partfeat.shape[2]]))

                # # 可以在open3D中可视化采样点
                # # 这些是对于该物体采样的点
                # pc = open3d.geometry.PointCloud()
                # pc.points = open3d.utility.Vector3dVector(input_pcs.cpu().numpy().reshape(-1,3))
                # # open3d.visualization.draw_geometries([pc])
                # rgb_np = rgb.cpu().numpy().astype(np.uint8).transpose(1,0,2)
                # # print("rgb ", rgb_np.shape)
                # # print(rgb_np)
                # # cv2.imshow("rgb", rgb_np)
                # # cv2.waitKey(1)
                # depth_np = depth.cpu().numpy().astype(np.float32).transpose(1,0)
                # twc_np = twc.cpu().numpy()
                # rgbd = open3d.geometry.RGBDImage.create_from_color_and_depth(
                #     open3d.geometry.Image(rgb_np),
                #     open3d.geometry.Image(depth_np),
                #     depth_trunc=cfg.max_depth,
                #     depth_scale=1,
                #     convert_rgb_to_intensity=False,
                # )
                # T_CW = np.linalg.inv(twc_np)
                # # input image pc
                # input_pc = open3d.geometry.PointCloud.create_from_rgbd_image(
                #     image=rgbd,
                #     intrinsic=intrinsic_open3d,
                #     extrinsic=T_CW)
                # input_pc.points = open3d.utility.Vector3dVector(np.array(input_pc.points) - obj_k.obj_center.cpu().numpy())
                # open3d.visualization.draw_geometries([pc, input_pc])


        ####################################################
        # training
        assert len(Batch_N_input_pcs) > 0
        # move data to GPU  (n_obj, n_iter_per_frame, win_size*num_per_frame, 3)
        # 把数据移动到gpu上面，
        with performance_measure(f"stacking and moving to gpu: "):
            # n,2400,10,3 n是物体数量，对应着刚才设置好的优化器吗，2400*10是针对这个物体的采样点数
            # 2400是因为每次迭代采样120个点，然后20次迭代，10是9个表面采样+1个均匀采样
            Batch_N_input_pcs = torch.stack(Batch_N_input_pcs).to(cfg.training_device)
            Batch_N_gt_depth = torch.stack(Batch_N_gt_depth).to(cfg.training_device)
            Batch_N_gt_rgb = torch.stack(Batch_N_gt_rgb).to(cfg.training_device) / 255. 
            Batch_N_depth_mask = torch.stack(Batch_N_depth_mask).to(cfg.training_device)
            Batch_N_obj_mask = torch.stack(Batch_N_obj_mask).to(cfg.training_device)
            Batch_N_sampled_z = torch.stack(Batch_N_sampled_z).to(cfg.training_device)
            if cfg.part_mode:
                Batch_N_gt_partfeat = torch.stack(Batch_N_gt_partfeat).to(cfg.training_device)
            if cfg.do_bg and scene_bg is not None:
                # 如果要处理背景物体，也把训练数据推到gpu
                bg_input_pcs = bg_input_pcs.to(cfg.training_device)
                bg_gt_depth = bg_gt_depth.to(cfg.training_device)
                bg_gt_rgb = bg_gt_rgb.to(cfg.training_device) / 255.
                bg_valid_depth_mask = bg_valid_depth_mask.to(cfg.training_device)
                bg_obj_mask = bg_obj_mask.to(cfg.training_device)
                bg_sampled_z = bg_sampled_z.to(cfg.training_device)
                if cfg.part_mode:
                    bg_gt_partfeat = bg_gt_partfeat.to(cfg.training_device)
                

        # 开始进行训练
        with performance_measure(f"Training over {len(obj_dict.keys())} objects,"):
            # 对于每一帧训练n_iter_per_frame次，也就是每次还是训练n_sample_per_step个，怕数据太多？
            for iter_step in range(cfg.n_iter_per_frame):
                # 把数据切片，每步训练n_sample_per_step个数据，所一才能
                data_idx = slice(iter_step*n_sample_per_step, (iter_step+1)*n_sample_per_step)
                batch_input_pcs = Batch_N_input_pcs[:, data_idx, ...]
                batch_gt_depth = Batch_N_gt_depth[:, data_idx, ...]
                batch_gt_rgb = Batch_N_gt_rgb[:, data_idx, ...]
                batch_depth_mask = Batch_N_depth_mask[:, data_idx, ...]
                batch_obj_mask = Batch_N_obj_mask[:, data_idx, ...]
                batch_sampled_z = Batch_N_sampled_z[:, data_idx, ...]
                if cfg.part_mode:
                    batch_gt_partfeat = Batch_N_gt_partfeat[:, data_idx, ...]
                if cfg.training_strategy == "forloop":
                    # for loop training
                    batch_alpha = []
                    batch_color = []
                    batch_clip = []
                    for k, obj_id in enumerate(obj_dict.keys()):
                        obj_k = obj_dict[obj_id]
                        embedding_k = obj_k.trainer.pe(batch_input_pcs[k])
                        alpha_k, color_k, clip_k = obj_k.trainer.fc_occ_map(embedding_k)
                        batch_alpha.append(alpha_k)
                        batch_color.append(color_k)
                        batch_clip.append(clip_k)

                    batch_alpha = torch.stack(batch_alpha)
                    batch_color = torch.stack(batch_color)
                    batch_clip = torch.stack(batch_clip)
                # 肯定是vmap训练方式，对于所有物体批次训练，前向传播，得到batch_alpha，batch_color
                elif cfg.training_strategy == "vmap":
                    # batched training
                    batch_embedding = vmap(pe_model)(pe_param, pe_buffer, batch_input_pcs)
                    batch_alpha, batch_color, batch_clip = vmap(fc_model)(fc_param, fc_buffer, batch_embedding)
                    # print("batch alpha ", batch_alpha.shape)
                else:
                    print("training strategy {} is not implemented ".format(cfg.training_strategy))
                    exit(-1)


            # step loss
            # with performance_measure(f"Batch LOSS"):
                # 计算当批次的损失，这个是所有物体的综合损失
                if not cfg.part_mode:
                    batch_loss, _ = loss.step_batch_loss(batch_alpha, batch_color,
                                        batch_gt_depth.detach(), batch_gt_rgb.detach(),
                                        batch_obj_mask.detach(), batch_depth_mask.detach(),
                                        batch_sampled_z.detach())
                else:
                    batch_loss, _ = loss.step_batch_loss(batch_alpha, batch_color,
                                     batch_gt_depth.detach(), batch_gt_rgb.detach(),
                                     batch_obj_mask.detach(), batch_depth_mask.detach(),
                                     batch_sampled_z.detach(), gt_partfeat = batch_gt_partfeat.detach(), pred_partfeat = batch_clip)
                    

                if cfg.do_bg and scene_bg is not None:
                    # 如果要处理背景，也把损失加进来
                    bg_data_idx = slice(iter_step * n_sample_per_step_bg, (iter_step + 1) * n_sample_per_step_bg)
                    bg_embedding = scene_bg.trainer.pe(bg_input_pcs[bg_data_idx, ...])
                    bg_alpha, bg_color, bg_clip = scene_bg.trainer.fc_occ_map(bg_embedding)
                    if not cfg.part_mode:
                        bg_loss, _ = loss.step_batch_loss(bg_alpha[None, ...], bg_color[None, ...],
                                                        bg_gt_depth[None, bg_data_idx, ...].detach(), bg_gt_rgb[None, bg_data_idx].detach(),
                                                        bg_obj_mask[None, bg_data_idx, ...].detach(), bg_valid_depth_mask[None, bg_data_idx, ...].detach(),
                                                        bg_sampled_z[None, bg_data_idx, ...].detach())
                    else:
                        bg_loss, _ = loss.step_batch_loss(bg_alpha[None, ...], bg_color[None, ...],
                                                        bg_gt_depth[None, bg_data_idx, ...].detach(), bg_gt_rgb[None, bg_data_idx].detach(),
                                                        bg_obj_mask[None, bg_data_idx, ...].detach(), bg_valid_depth_mask[None, bg_data_idx, ...].detach(),
                                                        bg_sampled_z[None, bg_data_idx, ...].detach(), gt_partfeat = bg_gt_partfeat[None, bg_data_idx, ...].detach(),
                                                        pred_partfeat = bg_clip[None, ...])
                    batch_loss += bg_loss

            # with performance_measure(f"Backward"):
                if AMP:
                    scaler.scale(batch_loss).backward()
                    scaler.step(optimiser)
                    scaler.update()
                else:
                    # 开始优化所有物体的网络权重
                    batch_loss.backward()
                    optimiser.step()
                optimiser.zero_grad(set_to_none=True)
                # print("loss ", batch_loss.item())

        # 根据优化器得到的参数，更新原始的物体的参数
        with performance_measure(f"updating vmap param"):
            if cfg.training_strategy == "vmap":
                with torch.no_grad():
                    for model_id, (obj_id, obj_k) in enumerate(obj_dict.items()):
                        for i, param in enumerate(obj_k.trainer.fc_occ_map.parameters()):
                            param.copy_(fc_param[i][model_id])
                        for i, param in enumerate(obj_k.trainer.pe.parameters()):
                            param.copy_(pe_param[i][model_id])


        ####################################################################
        torch.cuda.empty_cache()
        # 可视化，在满足条件式全部可视化出来
        if ((frame_id % cfg.n_vis_iter) == 0 or frame_id == dataset_len-1) and frame_id>0:
            print("*"*50)
            print("Ok for save, this frame is: ", sample["frame_id"])
            print("*"*50)

            
            # 给每个物体id分配语义，得到一个字典，把每个id映射到一个语义
            mapping_class = {}
            # 墙、地板、天花板先默认
            mapping_class.update({0:class_names.index("wall")})
            mapping_class.update({2:class_names.index("floor")})
            mapping_class.update({3:class_names.index("ceiling")})
            for obj_id, obj_k in vis_dict.items():
                if obj_id in {0,2,3}:
                    obj_k.set_semantic(mapping_class[obj_id])
                    continue
                this_clip_feat = obj_k.clip_feat
                this_caption_feat = obj_k.caption_feat
                eps = 0.2
                min_samples = 2
                if np.ndim(this_clip_feat) == 2:
                    this_clip_feat = get_majority_cluster_mean(this_clip_feat, eps, min_samples)
                    this_caption_feat = get_majority_cluster_mean(this_caption_feat, eps, min_samples)
                # 加权决定
                # weight_clip = 0.9
                cos_similarities_clip = np.dot(class_clipfeat, this_clip_feat)
                cos_similarities_caption = np.dot(class_capfeat, this_caption_feat)
                # cos_similarities_weight = cos_similarities_clip*weight_clip+cos_similarities_caption*(1-weight_clip)
                most_similar_index_clip = np.argmax(cos_similarities_clip)
                most_similar_index_cap = np.argmax(cos_similarities_caption)
                # caption足够确定就使用它自己即可
                if cos_similarities_caption[most_similar_index_cap] > 0.5:
                    mapping_class.update({obj_id:most_similar_index_cap})
                else:
                    mapping_class.update({obj_id:most_similar_index_clip})
                obj_k.set_semantic(mapping_class[obj_id])


            if cfg.if_ckpt:
                for obj_id, obj_k in vis_dict.items():
                    ckpt_dir = os.path.join(log_dir, "ckpt", str(obj_id))
                    os.makedirs(ckpt_dir, exist_ok=True)
                    bound,_ = obj_k.get_bound(intrinsic_open3d)   # update bound
                    obj_k.save_checkpoints(ckpt_dir, sample["frame_id"])
                    print("Save cpkt to:", ckpt_dir)
                # save current cam pose
                cam_dir = os.path.join(log_dir, "cam_pose")
                os.makedirs(cam_dir, exist_ok=True)
                # torch.save({"twc": twc,}, os.path.join(cam_dir, "twc_frame_{}".format(frame_id) + ".pth"))
                torch.save({"twc": twc,}, os.path.join(cam_dir, "twc_frame.pth"))
                print("Save twc to:", cam_dir)
            
            
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            np.random.seed(42)
            # random_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(class_clipfeat.shape[0])]
            random_colors = distinctipy.get_colors(class_clipfeat.shape[0], pastel_factor=1)
            random_colors = np.array(random_colors) * 255

            if cfg.if_render:
                # 渲染每张图像
                data_iter = iter(dataloader)
                for i in tqdm(range(dataset_len)):
                    sample = next(data_iter)
                    print("Rendering id:",sample["frame_id"])
                    twc = sample["T"]
                    # 渲染的rgb图像和深度图像
                    rendered_rgb_image = np.zeros((cam_info.width, cam_info.height, 3), dtype=np.uint8)
                    # 查看每个物体的mask，语义
                    rendered_mask_image = np.zeros((cam_info.width, cam_info.height, 3), dtype=np.uint8)
                    rendered_maskid_image = np.zeros((cam_info.width, cam_info.height), dtype=np.uint8)
                    # 深度图像先设置大一点
                    rendered_depth_image = np.ones((cam_info.width, cam_info.height), dtype=np.float32)*100
                    rays_dir = cam_info.rays_dir_cache
                    # 从而把采样像素的方向转到全局坐标系
                    for obj_id, obj_k in vis_dict.items():
                        # obj_mask = inst==obj_id
                        # obj_mask = obj_mask.cpu().numpy()
                        # num = np.count_nonzero(obj_mask)
                        # print("obj", obj_id, "have pixel num: ",num)
                        # print("obj", obj_id)
                        # if obj_id == 0:
                        #     continue
                        # if num > 0:
                        # obj_mask, render_depth, render_color = obj_k.render_2D_syn(twc.cpu().numpy(), intrinsic_open3d, rays_dir, chunk_size=1000, do_fine=False, obj_mask=obj_mask)
                        obj_mask, render_depth, render_color,_ = obj_k.render_2D_syn(twc.cpu().numpy(), intrinsic_open3d, rays_dir, chunk_size=3000, do_fine=False)
                        if render_depth is not None:
                            # 判断这个深度值有哪些是比较小的
                            this_depth = np.ones((cam_info.width, cam_info.height), dtype=np.float32)*100
                            this_rgb = np.zeros((cam_info.width, cam_info.height, 3), dtype=np.uint8)
                            this_depth[obj_mask] = render_depth
                            this_rgb[obj_mask] = render_color
                            ok_for_mask = rendered_depth_image > this_depth
                            rendered_rgb_image[ok_for_mask] = this_rgb[ok_for_mask]
                            rendered_rgb_image_vis = np.transpose(rendered_rgb_image, (1, 0, 2))  # 转置xy坐标
                            rendered_rgb_image_vis = cv2.cvtColor(rendered_rgb_image_vis, cv2.COLOR_BGR2RGB)  # 重新排列颜色通道
                            # debug，查看每个物体的mask又没有覆盖
                            rendered_mask_image[ok_for_mask] = random_colors[mapping_class[obj_id]]
                            contours, _ = cv2.findContours(ok_for_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(rendered_mask_image, contours, -1, (0, 0, 0), thickness=1)
                            rendered_maskid_image[ok_for_mask] = mapping_class[obj_id]
                            # 对于背景先认为深度不要，否则其他物体会被掩盖，记得修改
                            if obj_id not in cfg.bg_id:
                                rendered_depth_image[ok_for_mask] = this_depth[ok_for_mask]
                            # print(this_rgb[ok_for_mask])
                            # print(rendered_rgb_image[ok_for_mask] )
                    # 显示图像
                    # 转置图像并交换颜色通道顺序
                    rendered_rgb_image_corrected = np.transpose(rendered_rgb_image, (1, 0, 2))  # 转置xy坐标
                    rendered_depth_image_corrected = np.transpose(rendered_depth_image, (1, 0))  # 转置xy坐标
                    rendered_maskid_image_corrected = np.transpose(rendered_maskid_image, (1, 0))  # 转置xy坐标
                    raw_rgb = sample["image"].numpy()
                    rendered_mask_image = cv2.addWeighted(raw_rgb, 0.2, rendered_mask_image, 1, 0)
                    rendered_mask_image_corrected = np.transpose(rendered_mask_image, (1, 0, 2))  # 转置xy坐标
                    rendered_rgb_image_corrected = cv2.cvtColor(rendered_rgb_image_corrected, cv2.COLOR_BGR2RGB)  # 重新排列颜色通道
                    image_path = os.path.join(log_dir, "render", current_time)
                    print("save to :", image_path)
                    os.makedirs(image_path, exist_ok=True)
                    cv2.imwrite(image_path+"/rgb_"+str(sample["frame_id"])+'.png', rendered_rgb_image_corrected)
                    cv2.imwrite(image_path+"/depth_"+str(sample["frame_id"])+'.png', rendered_depth_image_corrected)
                    cv2.imwrite(image_path+"/maskid_"+str(sample["frame_id"])+'.png', rendered_maskid_image_corrected)
                    cv2.imwrite(image_path+"/mask_"+str(sample["frame_id"])+'.png', rendered_mask_image_corrected)

            if  cfg.if_vis:
                vis3d.clear_geometries()
                
            if cfg.if_obj:
                # 下面是保存mesh文件用的
                for obj_id, obj_k in vis_dict.items():
                    print("obj", obj_id)
                    # 在物体bound内进行展示
                    bound,_ = obj_k.get_bound(intrinsic_open3d)
                    if bound is None:
                        print("get bound failed obj ", obj_id)
                        continue
                    adaptive_grid_dim = int(np.minimum(np.max(bound.extent)//cfg.live_voxel_size+1, cfg.grid_dim))
                    # 得到物体的点云或者mesh，而且还放在了物体的对应中心位置
                    pcd, mesh, _ = obj_k.trainer.meshing(bound, obj_k.obj_center, grid_dim=adaptive_grid_dim, save_pcd=cfg.save_pcd, save_mesh=cfg.save_mesh)
                    if pcd is None and mesh is None:
                        print("meshing failed obj ", obj_id)
                        continue
                    if cfg.save_pcd and pcd is not None:
                        obj_pcd_output = os.path.join(log_dir, "scene_ply")
                        os.makedirs(obj_pcd_output, exist_ok=True)
                        filename = os.path.join(obj_pcd_output, "frame_{}_obj{}.ply".format(int(frame_id*cfg.stride), str(obj_id)))
                        o3d.io.write_point_cloud(filename, pcd)
                    if cfg.save_mesh and mesh is not None:
                        obj_mesh_output = os.path.join(log_dir, "scene_mesh")
                        os.makedirs(obj_mesh_output, exist_ok=True)
                        mesh.export(os.path.join(obj_mesh_output, "frame_{}_obj{}.obj".format(int(frame_id*cfg.stride), str(obj_id))))
                    if  cfg.if_vis:
                        # live vis
                        open3d_mesh = vis.trimesh_to_open3d(mesh)
                        vis3d.add_geometry(open3d_mesh)
                        vis3d.add_geometry(bound)
                        # update vis3d
                        vis3d.poll_events()
                        vis3d.update_renderer()
                        # 删除已经处理完的对象，并手动释放显存
                        del pcd, mesh
                        torch.cuda.empty_cache()
                        vis_dict.pop(obj_id)


