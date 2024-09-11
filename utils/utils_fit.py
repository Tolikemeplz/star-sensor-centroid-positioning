import os

import torch
from model.hybrid_centernet_training import focal_loss, reg_l1_loss

from tqdm import tqdm

from utils.utils import get_lr
from utils.utils_getlabel import get_label


def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen,
                  gen_val, Epoch, cuda, fp16, scaler, backbone, save_period, save_dir, gt_enlarge, local_rank=0):
    total_r_loss = 0
    total_c_loss = 0
    total_loss = 0

    val_loss = 0
    val_c_loss = 0
    val_r_loss = 0
    gt_enlarge = 2**gt_enlarge

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        with torch.no_grad():
            if cuda:
                # 将批次中的每个元素（ann）移动到GPU上
                batch = [ann.cuda(local_rank) for ann in batch]
        # batch_hms:(batch_size, output_shape[0], output_shape[1], 1)
        # batch_reg:(batch_size, output_shape[0], output_shape[1], 2)
        # batch_reg_mask:(batch_size, output_shape[0], output_shape[1])
        # labels:(batch_size, num_star, 2)
        # reg_weight:(batch_size, output_shape[0], output_shape[1])
        batch_images, batch_hms, batch_regs, batch_reg_masks, reg_weight= batch

        # print(f'batch_images: {batch_images.shape}')
        # print(f'batch_hms: {batch_hms.shape}')
        # print(f'batch_regs: {batch_regs.shape}')
        # print(f'batch_reg_masks: {batch_reg_masks.shape}')
        # print(f'reg_weight: {reg_weight.shape}')
        # print(f'labels: {labels.shape}')


        # ----------------------#
        #   清零梯度
        # ----------------------#
        optimizer.zero_grad()
        if not fp16:
            if backbone == "resnet50":
                hm, wh, offset = model_train(batch_images)
                c_loss = focal_loss(hm, batch_hms)
                #wh_loss = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
                off_loss = reg_l1_loss(offset, batch_regs, batch_reg_masks, cuda, gt_enlarge)

                loss = c_loss + off_loss

                total_loss += loss.item()
                total_c_loss += c_loss.item()
                total_r_loss +=  off_loss.item()
            elif backbone== "convnext":
                hm, offset= model_train(batch_images)
                c_loss = focal_loss(hm, batch_hms)
                off_loss = 10*reg_l1_loss(offset, batch_regs, batch_reg_masks, reg_weight)

                loss = c_loss + off_loss

                # detect = get_label(hm, batch_regs, )

                total_loss += loss.item()
                total_c_loss += c_loss.item()
                total_r_loss += off_loss.item()


            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                if backbone == "resnet50":
                    hm, wh, offset = model_train(batch_images)
                    c_loss = focal_loss(hm, batch_hms)
                    #wh_loss = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
                    off_loss = reg_l1_loss(offset, batch_regs, batch_reg_masks)

                    loss = c_loss  + off_loss

                    total_loss += loss.item()
                    total_c_loss += c_loss.item()
                    total_r_loss += off_loss.item()


            # ----------------------#
            #   反向传播
            # ----------------------#
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if local_rank == 0:
            pbar.set_postfix(**{'total_r_loss': total_r_loss / (iteration + 1),
                                'total_c_loss': total_c_loss / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break

        with torch.no_grad():
            if cuda:
                batch = [ann.cuda(local_rank) for ann in batch]
            batch_images, batch_hms,  batch_regs, batch_reg_masks, reg_weight= batch

            if backbone == "resnet50":
                hm, wh, offset = model_train(batch_images)
                c_loss = focal_loss(hm, batch_hms)
                off_loss = reg_l1_loss(offset, batch_regs, batch_reg_masks)

                loss = c_loss + off_loss

                val_loss += loss.item()
            elif backbone== "convnext":
                hm, offset= model_train(batch_images)
                c_loss = focal_loss(hm, batch_hms)
                off_loss = 10*reg_l1_loss(offset, batch_regs, batch_reg_masks, reg_weight)

                loss = c_loss + off_loss

                # detect = get_label(hm, batch_regs, )

                val_loss += loss.item()
                val_c_loss += c_loss.item()
                val_r_loss += off_loss.item()
            else:
                outputs = model_train(batch_images)
                index = 0
                loss = 0
                for output in outputs:
                    hm, wh, offset = output["hm"].sigmoid(), output["wh"], output["reg"]
                    c_loss = focal_loss(hm, batch_hms)
                    off_loss = reg_l1_loss(offset, batch_regs, batch_reg_masks)

                    loss += c_loss  + off_loss

                    # detect = get_label(batch_images, batch_hms, cuda, gt_enlarge)


                    index += 1
                val_loss += loss.item() / index

            if local_rank == 0:
                pbar.set_postfix(**{'val_r_loss': val_r_loss / (iteration + 1),
                                    'val_c_loss': val_c_loss / (iteration + 1)
                                    })
                pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        average_true_bias = eval_callback.on_epoch_end(epoch + 1, model_train)
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        if average_true_bias != 'Not eval epoch':
            loss_history.append_bias(average_true_bias)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        if average_true_bias != 'Not eval epoch':
            print('Total train Loss: %.3f || Val Loss: %.3f || average_true_bias: %.3f' % (
                total_loss / epoch_step, val_loss / epoch_step_val, average_true_bias))
        else:
            print('Total train Loss: %.3f || Val Loss: %.3f' % (
            total_loss / epoch_step, val_loss / epoch_step_val))

        # -----------------------------------------------#
        #   保存权值
        # -----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (
            epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_loss_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_loss_epoch_weights.pth"))
            torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer_best_loss_epoch.pth"))


        if average_true_bias != 'Not eval epoch':
            print(f'average_true_bias : {average_true_bias}')
            if len(loss_history.val_bias) <= 1 or(average_true_bias) <= min(loss_history.val_bias):
                print('Save best model to best_bias_epoch_weights.pth')
                torch.save(model.state_dict(), os.path.join(save_dir, "best_bias_epoch_weights.pth"))

                # 用来存放最佳偏差的那个epoch的权值更新的跟踪信息
                with open(os.path.join(save_dir, "best_bias_epoch.txt"), 'a') as f:
                    f.write(f'epoch: {epoch+1},best_bias: {average_true_bias}')
                    f.write("\n")

        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
        # 保存优化器状态
        torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer_last_epoch.pth"))