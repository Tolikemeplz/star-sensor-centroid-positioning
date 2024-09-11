import math
from functools import partial

import torch
import torch.nn.functional as F

def focal_loss(pred, target):
    pred = pred.permute(0, 2, 3, 1)

    # 通过比较 target 和1，找出所有正样本的位置，生成一个布尔掩码 pos_inds，然后将布尔值转换为浮点数（True转换为1.0，False转换为0.0）。
    pos_inds = target.eq(1).float()
    # 生成一个布尔掩码 neg_inds，标记所有负样本的位置，负样本的定义是 target 小于1。
    neg_inds = target.lt(1).float()
    # 对于负样本，根据它们与正样本的接近程度赋予不同的权重。接近正样本（即 target 接近1）的负样本会有更小的权重.
    neg_weights = torch.pow(1 - target, 4)

    # 为了防止计算 log 时出现数值不稳定性（比如对0取对数），使用 clamp 函数将 pred 的值限制在 1e-6 到 1 - 1e-6 之间。
    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)

    # 计算正样本的损失。对于每个正样本，使用对数函数和平方函数来增加分类难度大的样本的权重。
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    # 计算负样本的损失。对于每个负样本，使用对数函数和平方函数来增加分类难度大的样本的权重，并且乘以之前计算的负样本权重。
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    # 计算正样本的总数
    num_pos = pos_inds.float().sum()
    # 计算所有正样本的损失的累加和
    pos_loss = pos_loss.sum()
    # 计算所有负样本的损失的累加和
    neg_loss = neg_loss.sum()

    # 如果不存在正样本，那么损失只由负样本损失贡献。如果存在正样本，则总损失是正样本损失和负样本损失的加权和，权重由正样本的总数决定。
    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss

def reg_l1_loss(pred, target, mask, weight):
    pred = pred.permute(0, 2, 3, 1)
    expand_mask = torch.unsqueeze(mask, -1).repeat(1, 1, 1, 2)
    expand_weight = torch.unsqueeze(weight, -1).repeat(1, 1, 1, 2)

    loss = F.l1_loss(pred * expand_mask, target * expand_mask, reduction='none')

    loss = torch.sum(loss * expand_weight)
    loss = loss / (mask.sum() + 1e-4)
    return loss







def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
