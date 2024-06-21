import torch
import render_rays
import torch.nn.functional as F

def step_batch_loss(alpha, color, gt_depth, gt_color, sem_labels, mask_depth, z_vals,
                    color_scaling=5.0, opacity_scaling=10.0, gt_partfeat=None, pred_partfeat=None, partfeat_scaling=5.0):
    """
    apply depth where depth are valid                                       -> mask_depth
    apply depth, color loss on this_obj & unkown_obj == (~other_obj)        -> mask_obj
    apply occupancy/opacity loss on this_obj & other_obj == (~unknown_obj)  -> mask_sem

    output:
    loss for training
    loss_all for per sample, could be used for active sampling, replay buffer
    """
    mask_obj = sem_labels != 0
    # 一定属于该物体
    mask_obj = mask_obj.detach()
    # 排除不确定的部分
    mask_sem = sem_labels != 2
    mask_sem = mask_sem.detach()

    alpha = alpha.squeeze(dim=-1)
    color = color.squeeze(dim=-1)

    # 根据光线渲染得到占用概率
    occupancy = render_rays.occupancy_activation(alpha)
    # 根据占用概率得到在当前点停止的概率
    termination = render_rays.occupancy_to_termination(occupancy, is_batch=True)  # shape [num_batch, num_ray, points_per_ray]
    # 渲染深度
    render_depth = render_rays.render(termination, z_vals)
    diff_sq = (z_vals - render_depth[..., None]) ** 2
    var = render_rays.render(termination, diff_sq).detach()  # must detach here!
    render_color = render_rays.render(termination[..., None], color, dim=-2)
    render_opacity = torch.sum(termination, dim=-1)     # similar to obj-nerf opacity loss

    # 2D depth loss: only on valid depth & mask
    # [mask_depth & mask_obj]
    # loss_all = torch.zeros_like(render_depth)
    # 深度损失
    loss_depth_raw = render_rays.render_loss(render_depth, gt_depth, loss="L1", normalise=False)
    # 深度损失，只在有效深度和有效物体上使用
    # loss_depth = torch.mul(loss_depth_raw, mask_depth & mask_obj)   # keep dim but set invalid element be zero
    # # 使用每个像素点上深度值的方差作为权重，计算总深度损失
    # loss_depth = render_rays.reduce_batch_loss(loss_depth, var=var, avg=True, mask=mask_depth & mask_obj)   # apply var as imap
    # 修改一下，都在有效值内进行监督，因为不确定的太多了
    loss_depth = torch.mul(loss_depth_raw, mask_sem & mask_obj)   # keep dim but set invalid element be zero
    # 使用每个像素点上深度值的方差作为权重，计算总深度损失
    loss_depth = render_rays.reduce_batch_loss(loss_depth, var=var, avg=True, mask=mask_sem & mask_obj)   # apply var as imap

    # 2D color loss: only on obj mask
    # [mask_obj]
    # 颜色损失
    loss_col_raw = render_rays.render_loss(render_color, gt_color, loss="L1", normalise=False)
    # loss_col = torch.mul(loss_col_raw.sum(-1), mask_obj)
    # # loss_all += loss_col / 3. * color_scaling
    # loss_col = render_rays.reduce_batch_loss(loss_col, var=None, avg=True, mask=mask_obj)
    # print(loss_col_raw.shape)
    # print(loss_col_raw.sum(-1).shape)
    # 颜色也是，修改一下
    loss_col = torch.mul(loss_col_raw.sum(-1), mask_sem & mask_obj)
    # loss_all += loss_col / 3. * color_scaling
    loss_col = render_rays.reduce_batch_loss(loss_col, var=None, avg=True, mask=mask_sem & mask_obj)

    # 2D occupancy/opacity loss: apply except unknown area
    # [mask_sem]
    # loss_opacity_raw = F.mse_loss(torch.clamp(render_opacity, 0, 1), mask_obj.float().detach()) # encourage other_obj to be empty, while this_obj to be solid
    # print("opacity max ", torch.max(render_opacity.max()))
    # print("opacity min ", torch.max(render_opacity.min()))
    # 透明度损失，没有物体的地方是透明的，其他是1
    loss_opacity_raw = render_rays.render_loss(render_opacity, mask_obj.float(), loss="L1", normalise=False)
    # 对于透明度，忽略未知区域，比如mask的周边
    loss_opacity = torch.mul(loss_opacity_raw, mask_sem)  # but ignore -1 unkown area e.g., mask edges
    # loss_all += loss_opacity * opacity_scaling
    loss_opacity = render_rays.reduce_batch_loss(loss_opacity, var=None, avg=True, mask=mask_sem) 
    
    # loss for bp
    # 综合总损失
    l_batch = loss_depth + loss_col * color_scaling + loss_opacity * opacity_scaling
    
    if gt_partfeat is not None:
        render_partfeat = render_rays.render(termination[..., None], pred_partfeat, dim=-2)
        # 颜色损失
        # print(render_partfeat.shape)
        # print(gt_partfeat.shape)
        # print(render_color.shape)
        # print(gt_color.shape)
        loss_partfeat_raw = render_rays.render_loss(render_partfeat, gt_partfeat, loss="cos", normalise=False)
        # print(loss_partfeat_raw.shape)
        # print((mask_sem & mask_obj).shape)
        loss_partfeat = torch.mul(loss_partfeat_raw, mask_sem & mask_obj)
        loss_partfeat = render_rays.reduce_batch_loss(loss_partfeat, var=None, avg=True, mask=mask_sem & mask_obj)
        # print(loss_partfeat)
        # print(loss_depth)
        # print(loss_col)
        # print(color_scaling)
        # print(loss_opacity)
        # print(opacity_scaling)
        l_batch = l_batch + loss_partfeat * partfeat_scaling
        
    loss = l_batch.sum()

    return loss, None       # return loss, loss_all.detach()
