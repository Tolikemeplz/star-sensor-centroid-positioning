import datetime
import os
import pandas as pd
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader
from functools import partial
from model.hybrid_centernet import Convnext_Centernet
from model.hybrid_centernet_training import get_lr_scheduler, set_optimizer_lr
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import HybridCenternetDataset, centernet_dataset_collate
from utils.utils import (seed_everything,
                         show_config, worker_init_fn)
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":
    #   Cuda    是否使用Cuda
    Cuda = True
    # 或者选3407,或者11，或者43
    seed = 43
    distributed = False
    #   sync_bn     是否使用sync_bn，DDP模式多卡可用
    sync_bn = False
    #   fp16        是否使用混合精度训练
    fp16 = False
    model_path = ''
    #   input_shape     输入的shape大小，32的倍数
    input_shape = [128, 128]
    #   backbone        主干特征提取网络的选择
    backbone = "convnext"
    #   pretrained      是否使用主干网络的预训练权重，此处使用的是主干的权重，因此是在模型构建的时候进行加载的。
    #                   如果设置了model_path，则主干的权值无需加载，pretrained的值无意义。
    #                   如果不设置model_path，pretrained = True，此时仅加载主干开始训练。
    #                   如果不设置model_path，pretrained = False，Freeze_Train = Fasle，此时从0开始训练，且没有冻结主干的过程。

    # phi用来设置注意力模块
    phi= 8
    # hm_radius怎么设置还要好好计较一下，在dataloader那里
    hm_radius=1

    pretrained = True
    # 模型输出形状为128*2**gt_enlarge
    gt_enlarge = 0
    # Init_Epoch:断点续练时使用
    Init_Epoch = 0
    Freeze_Epoch = 0
    Freeze_batch_size = 64
    UnFreeze_Epoch = 80
    Unfreeze_batch_size = 64
    Freeze_Train = False

    Init_lr = 5e-4
    Min_lr = Init_lr * 0.002
    # 定义lr是否与batchsize有关
    lr_batch_related=True
    optimizer_type = "adamw"
    # 在Adam和AdamW优化器中，eps 被用在计算动量的平方根（即方差估计）时，以确保即使在动量非常小的情况下，方差估计也不会变成零。
    opt_eps = 1e-8
    momentum = 0.9
    weight_decay = 0
    lr_decay_type       = 'cos'

    #   save_period     多少个epoch保存一次权值
    save_period = 5
    #   save_dir        权值与日志文件保存的文件夹
    save_dir            = 'logs'
    #   eval_flag       是否在训练时进行评估，评估对象为验证集
    eval_flag           = True
    #   eval_period     代表多少个epoch评估一次，不建议频繁的评估
    eval_period = 10
    #   num_workers     用于设置是否使用多线程读取数据，1代表关闭多线程
    num_workers = 4
    train_annotation_path = 'dataset_200_train.txt'
    val_annotation_path = 'dataset_200_val.txt'

    seed_everything(seed)

    #   设置用到的显卡
    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0
        rank            = 0


    if backbone == "convnext":
        # asff输出的size是64,所以gt_enlarge=1时,是让sizex2放大2**1倍,即输出size为128
        model = Convnext_Centernet(bifpn_repeat=1,pretrained=pretrained,gt_enlarge=gt_enlarge,phi=phi)
    else:
        model = Convnext_Centernet(bifpn_repeat=1, pretrained=pretrained,gt_enlarge=gt_enlarge,phi=phi)

    # 记录loss
    if local_rank == 0:
        # 获取当前时间，并将其格式化为年_月_日_时_分_秒的字符串格式，以便于作为日志目录的名称。
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        # 创建一个日志目录的路径，将保存目录save_dir和前面生成的时间字符串time_str拼接起来。
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        # 创建一个LossHistory对象,于记录训练过程中的损失历史。
        loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history    = None

    # 是否在训练过程中使用自动混合精度
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()

    # 多卡同步批量归一化
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            #----------------------------#
            #   多卡平行运行
            #----------------------------#
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            # 使用torch.nn.DataParallel包装模型，这将自动处理模型在多个GPU之间的数据并行。
            model_train = torch.nn.DataParallel(model)
            # 设置cudnn.benchmark=True，这允许CuDNN自动寻找最适合当前配置的高效算法，从而提高运行效率。
            cudnn.benchmark = True
            # 将模型model_train移动到GPU上。
            model_train = model_train.cuda()


    # 读取数据集对应的txt
    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    # train_label = pd.read_csv(train_label_path)
    # val_label = pd.read_csv(val_lable_path)
    # num_train = len(train_label)
    # num_val = len(val_label)

    if local_rank == 0:
        show_config(
            model_path = model_path, input_shape = input_shape, gt_enlarge = gt_enlarge, phi = phi, pretrained = pretrained,\
            Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
        )

        #   总训练世代指的是遍历全部数据的总次数
        #   总训练步长指的是梯度下降的总次数
        #   每个训练世代包含若干训练步长，每个训练步长进行一次梯度下降。
        #   此处仅建议最低训练世代，上不封顶，计算时只考虑了解冻部分
        wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
        total_step = num_train // Unfreeze_batch_size * UnFreeze_Epoch
        if total_step <= wanted_step:
            if num_train // Unfreeze_batch_size == 0:
                raise ValueError('数据集过小，无法进行训练，请扩充数据集。')
            wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
            print("\n\033[1;33;44m[Warning] 使用%s优化器时，建议将训练总步长设置到%d以上。\033[0m" % (
            optimizer_type, wanted_step))
            print("\033[1;33;44m[Warning] 本次运行的总训练数据量为%d，Unfreeze_batch_size为%d，共训练%d个Epoch，计算出总训练步长为%d。\033[0m" % (
                num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
            print("\033[1;33;44m[Warning] 由于总训练步长为%d，小于建议总步长%d，建议设置总世代为%d。\033[0m" % (
            total_step, wanted_step, wanted_epoch))

    if True:
        UnFreeze_flag = False
        #   冻结一定部分训练
        if Freeze_Train:
            model.freeze_backbone()
        #   如果不冻结训练的话，直接设置batch_size为Unfreeze_batch_size
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size
        #   判断当前batch_size，自适应调整学习率.这应该也是和冻结训练有关的
        if lr_batch_related:
            nbs = 64
            lr_limit_max = 5e-4 if optimizer_type == 'adam' else 5e-2
            lr_limit_min = 2.5e-4 if optimizer_type == 'adam' else 5e-4
            Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
            Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
        else:
            Init_lr_fit = Init_lr
            Min_lr_fit = Min_lr

        #   根据optimizer_type选择优化器
        optimizer = {
            'adamw':optim.AdamW(model.parameters(), Init_lr_fit,betas=(momentum, 0.999),weight_decay=weight_decay,eps=opt_eps),
            'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
            'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True,
                             weight_decay=weight_decay)
        }[optimizer_type]
        #   获得学习率下降的公式
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        #   判断每一个世代的长度
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        train_dataset = HybridCenternetDataset(train_lines,input_shape,gt_enlarge,hm_radius)
        val_dataset = HybridCenternetDataset(val_lines,input_shape,gt_enlarge,hm_radius)

        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            val_sampler     = None
            shuffle         = True
        # pin_memory为Ture,则Dataloader会将张量复制到CUDA中的固定内存中,加快传输到GPU的速度
        # drop_last丢弃最后一个不完整的批次
        gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=True,
                         drop_last=True, collate_fn=centernet_dataset_collate, sampler=train_sampler,
                         worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
        gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True,
                             drop_last=True, collate_fn=centernet_dataset_collate, sampler=val_sampler,
                             worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

        if local_rank == 0:
            eval_callback   = EvalCallback(model, backbone, input_shape, val_lines, log_dir, Cuda, gt_enlarge,\
                                            eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback   = None

        # 开始模型训练
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            # 如果有冻结学习部分,则解冻,并设置参数
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size
                # 判断当前batch_size，自适应调整学习率
                if lr_batch_related:
                    nbs = 64
                    lr_limit_max = 5e-4 if optimizer_type == 'adam' else 5e-2
                    lr_limit_min = 2.5e-4 if optimizer_type == 'adam' else 5e-4
                    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                    Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                else:
                    Init_lr_fit = Init_lr
                    Min_lr_fit = Min_lr
                # 获得学习率下降公式
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                model.unfreeze_backbone()

                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                if distributed:
                    batch_size = batch_size // ngpus_per_node

                gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last=True, collate_fn=centernet_dataset_collate, sampler=train_sampler,
                                 worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
                gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                     pin_memory=True,
                                     drop_last=True, collate_fn=centernet_dataset_collate, sampler=val_sampler,
                                     worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

                UnFreeze_flag = True

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch,
                          epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, fp16, scaler, backbone,
                          save_period, save_dir, gt_enlarge, local_rank)

        if local_rank == 0:
            loss_history.writer.close()













