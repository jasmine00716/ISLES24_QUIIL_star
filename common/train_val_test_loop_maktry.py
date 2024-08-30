import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import matplotlib.pyplot as plt
from utils.common.loss_functions import test_score
from utils.isles_eval_util import compute_dice_f1_instance_difference, compute_absolute_volume_difference


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, init, alpha=0.2):
        self.val = val
        if init:
            self.avg = val
        else:
            self.avg = alpha * val + (1 - alpha) * self.avg


def pad_to_multiple_of_N(x, dim, N=4):
    # 현재 차원 크기
    current_dim = x.size(dim)

    multiply_cnt = current_dim // N
    if current_dim % N == 0:
        return x
    else:
        target_dim = (multiply_cnt + 1) * N
        new_x = torch.zeros((x.shape[0], x.shape[1], target_dim, x.shape[3], x.shape[4]))
        new_x[:, :, :current_dim, :, :] = x

        return new_x


def random_slice_slicing(input_modality, gt_label):
    slice_size = 80
    max_start_index = input_modality.size(2) - slice_size
    start_index = torch.randint(0, max_start_index + 1, (1,)).item()

    new_input = input_modality[:, :, start_index:start_index + slice_size, :, :]
    new_gt_label = gt_label[:, :, start_index:start_index + slice_size, :, :]
    return new_input, new_gt_label


def split_input_voxel(data, splits=2):
    _, _, S, H, W = data.shape

    chunk_size_s = S // splits
    chunk_size_h = H // splits
    chunk_size_w = W // splits

    patches = []
    for i in range(splits):
        for j in range(splits):
            for k in range(splits):
                patch = data[:, :, i * chunk_size_s:(i + 1) * chunk_size_s, j * chunk_size_h:(j + 1) * chunk_size_h, k * chunk_size_w:(k + 1) * chunk_size_w]
                patches.append(patch)

    return patches


def constant_slice_slicing(input_modality, gt_label):
    slice_size = 80
    # if input_modality.shape

    max_start_index = input_modality.size(2) - slice_size
    half_start_index = max_start_index // 2

    new_input = input_modality[:, :, half_start_index:half_start_index + slice_size, :, :]
    new_gt_label = gt_label[:, :, half_start_index:half_start_index + slice_size, :, :]

    return new_input, new_gt_label


def train_model(train_loader, model, loss_seg_fn, optimizer, device):
    size = len(train_loader.dataset)
    model = model.to(device)
    model.train()

    train_loss = AverageMeter()
    tqdm_bar = tqdm(train_loader, total=size, ascii=True)
    iter = 0

    for data in tqdm_bar:
        # ncct, dwi_adc, mask, voxel_volume
        concated_input, sub_ctp, gt_label, voxel_volume = data[0], data[1], data[2], data[3]
        gt_label[gt_label > 0] = 1

        concated_input, gt_label, sub_ctp = concated_input.to(device), gt_label.to(device), sub_ctp.to(device)

        pred_label = model(concated_input, sub_ctp)
        loss = loss_seg_fn(torch.sigmoid(pred_label[0]), gt_label) # stunet
        # loss = loss_seg_fn(torch.sigmoid(pred_label), gt_label) #moret

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.update(loss.mean().detach(), iter == 0)
        if iter % 5 == 0:
            tqdm_bar.set_description('Loss {loss.val:.4f} ({loss.avg:.4f}) '.format(loss=train_loss))
        iter += 1

    return train_loss.avg


# validation, test set calculation
def val_cal(epoch, val_loader, model, loss_fn, threshold, device, with_synthe=False, only_mr=False):
    size = len(val_loader.dataset)
    val_loss = AverageMeter()
    val_f1 = AverageMeter()
    val_lesion_cnt = AverageMeter()
    val_dice = AverageMeter()
    val_volume = AverageMeter()
    tqdm_bar = tqdm(val_loader, total=size, ascii=True)
    model = model.to(device)
    model.eval()
    iter = 0
    with torch.no_grad():
        for data in tqdm_bar:
            concated_input, sub_ctp, gt_label, voxel_volume = data[0], data[1], data[2], data[3]
            gt_label[gt_label > 0] = 1

            concated_input, gt_label, sub_ctp = concated_input.to(device), gt_label.to(device), sub_ctp.to(device)
            pred_label = model(concated_input, sub_ctp)
            pred_label = torch.sigmoid(pred_label[0]) #stunet
            # pred_label = torch.sigmoid(pred_label) # moret

            loss = loss_fn(pred_label, gt_label).mean().item()

            out_cut = np.copy(pred_label.data.cpu().numpy())
            out_cut[out_cut < threshold] = 0.0
            out_cut[out_cut >= threshold] = 1.0
            out_cut = torch.from_numpy(out_cut).squeeze(0).squeeze(0)

            gt_label = gt_label.data.cpu().squeeze(0).squeeze(0)

            f1, lcd, dice = compute_dice_f1_instance_difference(gt_label.data.cpu(), out_cut)
            abs_vol_diff = compute_absolute_volume_difference(gt_label.data.cpu(), out_cut,
                                                                              voxel_volume.numpy())

            val_f1.update(f1, iter == 0)
            val_lesion_cnt.update(lcd, iter == 0)
            val_dice.update(dice, iter == 0)
            val_volume.update(abs_vol_diff, iter == 0)

            val_loss.update(loss, iter == 0)
            # np.save(f"/data2/braindata/ISLES_synthe_seg_train/{patient_num[0]}.npy", out_cut)
            iter += 1

        print(f"[Epoch {epoch + 1}] Val Error: \n F1: {val_f1.avg}, Lesion_cnt : {val_lesion_cnt.avg}, \n DICE: {val_dice.avg} , Vol_diff : {val_volume.avg}, Avg_loss : {val_loss.avg:>9f} \n")
            # print(f"[Epoch {epoch + 1}] Val Error: \n F1: {f1}, Lesion_cnt : {lcd}, \n DICE: {dice} , Vol_diff : {abs_vol_diff}, Avg_loss : {loss:>9f} \n")

    return val_f1.avg, val_lesion_cnt.avg, val_dice.avg, val_volume.avg, val_loss.avg


def pretrained_weights_acc_n_visual(val_loader, model, loss_fn, threshold, device):
    size = len(val_loader.dataset)
    val_loss = AverageMeter()
    val_f1 = AverageMeter()
    val_lesion_cnt = AverageMeter()
    val_dice = AverageMeter()
    val_volume = AverageMeter()
    tqdm_bar = tqdm(val_loader, total=size, ascii=True)
    model = model.to(device)
    model.eval()
    iter = 0
    with torch.no_grad():
        for data in tqdm_bar:
            concated_input, sub_ctp, gt_label, voxel_volume = data[0], data[1], data[2], data[3]
            gt_label[gt_label > 0] = 1

            concated_input, gt_label, sub_ctp = concated_input.to(device), gt_label.to(device), sub_ctp.to(device)
            pred_label = model(concated_input, sub_ctp)
            pred_label = torch.sigmoid(pred_label[0]) #stunet
            # pred_label = torch.sigmoid(pred_label) # moret

            prediction = pred_label.data.cpu().numpy().squeeze(0).squeeze(0)

            out_cut = prediction >= 0.3

            gt_label = gt_label.data.cpu().squeeze(0).squeeze(0)

            f1, lcd, dice = compute_dice_f1_instance_difference(gt_label, out_cut)
            abs_vol_diff = compute_absolute_volume_difference(gt_label, out_cut, voxel_volume.numpy())

            val_f1.update(f1, iter == 0)
            val_lesion_cnt.update(lcd, iter == 0)
            val_dice.update(dice, iter == 0)
            val_volume.update(abs_vol_diff, iter == 0)

            for i in range(gt_label.shape[0]):
                plt.imsave(f"/data2/braindata/ISLES_maktry/result_cnn/pred_{iter}_{i:03d}.png", out_cut[i], cmap='binary')
                plt.imsave(f"/data2/braindata/ISLES_maktry/result_cnn/gt_{iter}_{i:03d}.png", gt_label[i], cmap='binary')
            iter += 1

        print(f"Val Error: \n F1: {val_f1.avg}, Lesion_cnt : {val_lesion_cnt.avg}, \n DICE: {val_dice.avg} , Vol_diff : {val_volume.avg}, Avg_loss : {val_loss.avg:>9f} \n")
            # print(f"[Epoch {epoch + 1}] Val Error: \n F1: {f1}, Lesion_cnt : {lcd}, \n DICE: {dice} , Vol_diff : {abs_vol_diff}, Avg_loss : {loss:>9f} \n")
