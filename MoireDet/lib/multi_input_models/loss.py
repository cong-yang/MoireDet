# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:56
# @Author  : zhoujun
import itertools
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F



class three_part_loss(nn.Module):
    def __init__(self,weights = [0.1,0,1]):
        super().__init__()
        # self.triple_loss = TripleLoss()

        self.triple_loss = SpecifyLayersLoss()

        self.alpha_loss = AlphaLoss()
        self.weights = weights

    def forward(self,pred,alpha,mask,F_I,F_F,F_B):
        Orthogonal_loss, compose_loss = self.triple_loss(F_I,F_F,F_B,alpha)
        regress_loss = self.alpha_loss(pred,alpha,mask)

        losses = [Orthogonal_loss,compose_loss,regress_loss]

        loss = sum(w*v for w,v in zip(self.weights,losses))/sum(self.weights)

        return loss,Orthogonal_loss,compose_loss,regress_loss



class SpecifyLayersLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_gap = 5

        self.channel = 128
        self.smoke_layers = 10
        self.env_layers = 30

        self.max = 5


    def forward(self, F_I,F_F,F_B,alpha):# F_I:the composite of F&B F:Foreground; B:Background

        alpha = F.interpolate(alpha, F_I.size()[-2:], mode='bilinear',
                              align_corners=True)

        mask = torch.where(alpha<0.1,torch.zeros_like(alpha),torch.ones_like(alpha))

        smoke_in_smoke_layers_loss = 0
        env_in_smoke_layers_loss = 0

        smoke_in_env_layers_loss = 0
        env_in_env_layers_loss = 0

        if mask.sum() > 0:

            smoke_in_smoke_layers_loss = F_I[:,:self.smoke_layers,:,:]
            smoke_in_smoke_layers_loss = smoke_in_smoke_layers_loss*mask
            smoke_in_smoke_layers_loss = torch.sum(smoke_in_smoke_layers_loss)/mask.sum()
            smoke_in_smoke_layers_loss = F.relu(self.max-smoke_in_smoke_layers_loss)

            env_in_smoke_layers_loss = F_B[:,:self.smoke_layers,:,:]
            env_in_smoke_layers_loss = env_in_smoke_layers_loss*mask
            env_in_smoke_layers_loss = torch.sum(env_in_smoke_layers_loss)/mask.sum()

            smoke_in_env_layers_loss = F_I[:,self.smoke_layers:self.env_layers,:,:]
            smoke_in_env_layers_loss = smoke_in_env_layers_loss*mask
            smoke_in_env_layers_loss = torch.sum(smoke_in_env_layers_loss)/mask.sum()


            env_in_env_layers_loss = F_B[:,self.smoke_layers:self.env_layers,:,:]
            env_in_env_layers_loss = env_in_env_layers_loss*mask
            env_in_env_layers_loss = torch.sum(env_in_env_layers_loss)/mask.sum()
            env_in_env_layers_loss = F.relu(self.max - env_in_env_layers_loss)

        Orthogonal_loss = smoke_in_smoke_layers_loss+env_in_smoke_layers_loss+smoke_in_env_layers_loss+env_in_env_layers_loss




        compose_loss = F_I - (F_F+F_B*(1-alpha))
        compose_loss = torch.mean(torch.pow(compose_loss, 2))

        return Orthogonal_loss,compose_loss


class SpecifyLayersLoss_bak(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_gap = 5

        self.channel = 128
        self.smoke_layers = 10
        self.env_layers = 30

        self.max = 5


    def forward(self, F_I,F_F,F_B,alpha):# F_I:the composite of F&B F:Foreground; B:Background

        alpha = F.interpolate(alpha, F_I.size()[-2:], mode='bilinear',
                              align_corners=True)

        mask = torch.where(alpha<0.1,torch.zeros_like(alpha),torch.ones_like(alpha))

        smoke_in_smoke_layers_loss = 0
        env_in_smoke_layers_loss = 0


        smoke_in_env_layers_loss = 0
        env_in_env_layers_loss = 0

        if mask.sum() > 0:

            smoke_in_smoke_layers_loss = F_I[:,:self.smoke_layers,:,:]
            smoke_in_smoke_layers_loss = smoke_in_smoke_layers_loss*mask
            smoke_in_smoke_layers_loss = torch.sum(smoke_in_smoke_layers_loss)/mask.sum()
            smoke_in_smoke_layers_loss = F.relu(self.max-smoke_in_smoke_layers_loss)

            env_in_smoke_layers_loss = F_B[:,:self.smoke_layers,:,:]
            env_in_smoke_layers_loss = env_in_smoke_layers_loss*mask
            env_in_smoke_layers_loss = torch.sum(env_in_smoke_layers_loss)/mask.sum()

            smoke_in_env_layers_loss = F_I[:,self.smoke_layers:self.env_layers,:,:]
            smoke_in_env_layers_loss = smoke_in_env_layers_loss*mask
            smoke_in_env_layers_loss = torch.sum(smoke_in_env_layers_loss)/mask.sum()


            env_in_env_layers_loss = F_B[:,self.smoke_layers:self.env_layers,:,:]
            env_in_env_layers_loss = env_in_env_layers_loss*mask
            env_in_env_layers_loss = torch.sum(env_in_env_layers_loss)/mask.sum()
            env_in_env_layers_loss = F.relu(self.max - env_in_env_layers_loss)

        Orthogonal_loss = smoke_in_smoke_layers_loss+env_in_smoke_layers_loss+smoke_in_env_layers_loss+env_in_env_layers_loss




        compose_loss = F_I - (F_F+F_B*(1-alpha))
        compose_loss = torch.mean(torch.pow(compose_loss, 2))

        return Orthogonal_loss,compose_loss


class TripleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_gap = 5

    def forward(self, F_I,F_F,F_B,alpha):# F_I:the composite of F&B F:Foreground; B:Background
        alpha = F.interpolate(alpha, F_I.size()[-2:], mode='bilinear',
                              align_corners=True)

        mask = torch.where(alpha<0.1,torch.zeros_like(alpha),torch.ones_like(alpha))
        Orthogonal_loss = (F_F - F_B)*mask



        Orthogonal_loss = torch.where(torch.abs(Orthogonal_loss.detach()) > self.max_gap,torch.zeros_like(Orthogonal_loss),Orthogonal_loss)

        Orthogonal_loss = torch.abs(torch.sum(Orthogonal_loss,(-1,-2)))
        Orthogonal_loss = -torch.mean(Orthogonal_loss)



        compose_loss = F_I - (F_F+F_B*(1-alpha))
        compose_loss = torch.mean(torch.pow(compose_loss, 2))

        return Orthogonal_loss,compose_loss


class AlphaLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,pred,target,mask):
        pred = torch.sigmoid(pred)
        loss = torch.mean(torch.abs(pred-target)*mask)
        return loss


class PANLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.25, delta_agg=0.5, delta_dis=3, ohem_ratio=3, reduction='mean'):
        """
        Implement PSE Loss.
        :param alpha: loss kernel 前面的系数
        :param beta: loss agg 和 loss dis 前面的系数
        :param delta_agg: 计算loss agg时的常量
        :param delta_dis: 计算loss dis时的常量
        :param ohem_ratio: OHEM的比例
        :param reduction: 'mean' or 'sum'对 batch里的loss 算均值或求和
        """
        super().__init__()
        assert reduction in ['mean', 'sum'], " reduction must in ['mean','sum']"
        self.alpha = alpha
        self.beta = beta
        self.delta_agg = delta_agg
        self.delta_dis = delta_dis
        self.ohem_ratio = ohem_ratio
        self.reduction = reduction

    def forward(self, outputs, labels, densitys,training_masks):
        pred_seg = outputs[:, 0, :, :]
        pred_dens = outputs[:, 1, :, :]


        # 计算 text loss
        selected_masks = self.ohem_batch(pred_seg, labels, training_masks)
        selected_masks = selected_masks.to(outputs.device)

        seg_texts = self.dice_loss(pred_seg, labels, selected_masks)

        dens_loss = self.regress_loss(pred_dens,densitys,training_masks)

        seg_texts = torch.mean(seg_texts)
        dens_loss = torch.mean(dens_loss)

        loss = seg_texts + dens_loss

        return loss,seg_texts,dens_loss

    def agg_dis_loss(self, texts, kernels, gt_texts, gt_kernels, similarity_vectors):
        """
        计算 loss agg
        :param texts: 文本实例的分割结果 batch_size * (w*h)
        :param kernels: 缩小的文本实例的分割结果 batch_size * (w*h)
        :param gt_texts: 文本实例的gt batch_size * (w*h)
        :param gt_kernels: 缩小的文本实例的gt batch_size*(w*h)
        :param similarity_vectors: 相似度向量的分割结果 batch_size * 4 *(w*h)
        :return:
        """
        batch_size = texts.size()[0]
        texts = texts.contiguous().reshape(batch_size, -1)
        kernels = kernels.contiguous().reshape(batch_size, -1)
        gt_texts = gt_texts.contiguous().reshape(batch_size, -1)
        gt_kernels = gt_kernels.contiguous().reshape(batch_size, -1)
        similarity_vectors = similarity_vectors.contiguous().view(batch_size, 4, -1)
        loss_aggs = []
        loss_diss = []
        for text_i, kernel_i, gt_text_i, gt_kernel_i, similarity_vector in zip(texts, kernels, gt_texts, gt_kernels,
                                                                               similarity_vectors):
            text_num = gt_text_i.max().item() + 1
            loss_agg_single_sample = []
            G_kernel_list = []  # 存储计算好的G_Ki,用于计算loss dis
            # 求解每一个文本实例的loss agg
            for text_idx in range(1, int(text_num)):
                # 计算 D_p_Ki
                single_kernel_mask = gt_kernel_i == text_idx
                if single_kernel_mask.sum() == 0 or (gt_text_i == text_idx).sum() == 0:
                    # 这个文本被crop掉了
                    continue
                # G_Ki, shape: 4
                G_kernel = similarity_vector[:, single_kernel_mask].mean(1)  # 4
                G_kernel_list.append(G_kernel)
                # 文本像素的矩阵 F(p) shape: 4* nums (num of text pixel)
                text_similarity_vector = similarity_vector[:, gt_text_i == text_idx]
                # ||F(p) - G(K_i)|| - delta_agg, shape: nums
                text_G_ki = (text_similarity_vector - G_kernel.reshape(4, 1)).norm(2, dim=0) - self.delta_agg
                # D(p,K_i), shape: nums
                D_text_kernel = torch.max(text_G_ki, torch.tensor(0, device=text_G_ki.device, dtype=torch.float)).pow(2)
                # 计算单个文本实例的loss, shape: nums
                loss_agg_single_text = torch.log(D_text_kernel + 1).mean()
                loss_agg_single_sample.append(loss_agg_single_text)
            if len(loss_agg_single_sample) > 0:
                loss_agg_single_sample = torch.stack(loss_agg_single_sample).mean()
            else:
                loss_agg_single_sample = torch.tensor(0, device=texts.device, dtype=torch.float)
            loss_aggs.append(loss_agg_single_sample)

            # 求解每一个文本实例的loss dis
            loss_dis_single_sample = 0
            for G_kernel_i, G_kernel_j in itertools.combinations(G_kernel_list, 2):
                # delta_dis - ||G(K_i) - G(K_j)||
                kernel_ij = self.delta_dis - (G_kernel_i - G_kernel_j).norm(2)
                # D(K_i,K_j)
                D_kernel_ij = torch.max(kernel_ij, torch.tensor(0, device=kernel_ij.device, dtype=torch.float)).pow(2)
                loss_dis_single_sample += torch.log(D_kernel_ij + 1)
            if len(G_kernel_list) > 1:
                loss_dis_single_sample /= (len(G_kernel_list) * (len(G_kernel_list) - 1))
            else:
                loss_dis_single_sample = torch.tensor(0, device=texts.device, dtype=torch.float)
            loss_diss.append(loss_dis_single_sample)
        return torch.stack(loss_aggs), torch.stack(loss_diss)

    def dice_loss(self, input, target, mask):
        input = torch.sigmoid(input)
        target[target <= 0.5] = 0
        target[target > 0.5] = 1
        input = input.contiguous().view(input.size()[0], -1)
        target = target.contiguous().view(target.size()[0], -1)
        mask = mask.contiguous().view(mask.size()[0], -1)

        input = input * mask
        target = target * mask

        a = torch.sum(input * target, 1)
        b = torch.sum(input * input, 1) + 0.001
        c = torch.sum(target * target, 1) + 0.001
        d = (2 * a) / (b + c)
        return 1 - d

    def regress_loss(self, input, target, mask):
        input = torch.sigmoid(input)
        loss = torch.mean(torch.abs(target-input))*mask
        return loss


    def ohem_single(self, score, gt_text, training_mask):
        pos_num = (int)(np.sum(gt_text > 0.5)) - (int)(np.sum((gt_text > 0.5) & (training_mask <= 0.5)))

        if pos_num == 0:
            # selected_mask = gt_text.copy() * 0 # may be not good
            selected_mask = training_mask
            selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
            return selected_mask

        neg_num = (int)(np.sum(gt_text <= 0.5))
        neg_num = (int)(min(pos_num * self.ohem_ratio, neg_num))

        if neg_num == 0:
            selected_mask = training_mask
            selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
            return selected_mask

        neg_score = score[gt_text <= 0.5]
        neg_score_sorted = np.sort(-neg_score)
        threshold = -neg_score_sorted[neg_num - 1]
        selected_mask = ((score >= threshold) | (gt_text > 0.5)) & (training_mask > 0.5)
        selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
        return selected_mask

    def ohem_batch(self, scores, gt_texts, training_masks):
        scores = scores.data.cpu().numpy()
        gt_texts = gt_texts.data.cpu().numpy()
        training_masks = training_masks.data.cpu().numpy()

        selected_masks = []
        for i in range(scores.shape[0]):
            selected_masks.append(self.ohem_single(scores[i, :, :], gt_texts[i, :, :], training_masks[i, :, :]))

        selected_masks = np.concatenate(selected_masks, 0)
        selected_masks = torch.from_numpy(selected_masks).float()

        return selected_masks
