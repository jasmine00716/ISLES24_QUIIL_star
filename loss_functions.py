import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# from sklearn.metrics import f1_score, precision_score


class DiceBCEFocalLoss(nn.Module):
    def __init__(self, alpha = 1, gamma = 2, weight=None, size_average=True):
        super(DiceBCEFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma


    def forward(self, inputs, targets, smooth=1e-2):
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        # targets = targets.view(-1)
        targets = targets.contiguous().view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        bceloss = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_for_focal = F.binary_cross_entropy(inputs, targets, reduction='mean')
        pt = torch.exp(-BCE_for_focal)

        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_for_focal

        # return bceloss + dice_loss + F_loss
        return dice_loss + F_loss

#
# class FocalLoss(nn.Module):
#     def __init__(self, alpha=1, gamma=2):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#
#     def forward(self, inputs, targets):
#         BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
#         pt = torch.exp(-BCE_loss)
#         F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
#         return F_loss.mean()


class total_loss(nn.Module):
    def __init__(self):
        super(total_loss, self).__init__()

    def forward(self, inputs, targets):
        return FocalLoss(inputs, targets) + DiceBCELoss(inputs, targets)


class RBF(nn.Module):  # 가우스 커널 (RBF kernel)
    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth = bandwidth  # RBF 수식에서 sigma에 해당

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)
        return self.bandwidth

    def forward(self, X):  # 거리의 총 합 반환, 유사도를 측정하는 효과
        X = X.view(X.shape[0], -1)
        L2_distances = torch.cdist(X, X) ** 2  # 각 element 끼리의 유클리드 거리 구함, (X rows) X (X rows) 크기 만큼의 matrix

        self.bandwidth_multipliers= self.bandwidth_multipliers.to(self.get_bandwidth(L2_distances).device)

        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)


class MMDLoss(nn.Module):

    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        X = X.view(X.shape[0], -1)
        Y = Y.view(Y.shape[0], -1)

        K = self.kernel(torch.vstack([X, Y]))  # ((X+Y의 rows), (X cols)) 크기의 유사도 matrix

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()  # 데이터 X 내 평균 유사도
        XY = K[:X_size, X_size:].mean()  # 데이터 X, Y 간 평균 유사도
        YY = K[X_size:, X_size:].mean()  # 데이터 Y 내 평균 유사도
        return XX - 2 * XY + YY  # (X-Y)^2


class IntermediateBinaryKLDivergenceLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(IntermediateBinaryKLDivergenceLoss, self).__init__()
        self.temperature = temperature

    def forward(self, feat_S, feat_T):
        # # Temperature를 적용하여 확률 분포 생성
        # t_prob = torch.sigmoid(feat_T / self.temperature)
        # s_prob = torch.sigmoid(feat_S / self.temperature)
        #
        # # KL Divergence 계산
        # return F.kl_div(torch.log(s_prob + 1e-10), t_prob, reduction='mean') * (self.temperature ** 2)
        # Teacher 모델의 soft labels 생성
        soft_labels = F.softmax(feat_T / self.temperature, dim=1)
        # Student 모델의 예측에 temperature scaling 적용
        s_prob = F.log_softmax(feat_S / self.temperature, dim=1)

        # KL Divergence Loss 계산
        kd_loss = F.kl_div(s_prob, soft_labels, reduction='batchmean') * (self.temperature ** 2)
        return kd_loss


class CriterionMSE(nn.Module):
    def __init__(self):
        super(CriterionMSE, self).__init__()

    def forward(self, feat_S, feat_T):
        return ((feat_S - feat_T) ** 2).mean()


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def test_score(preds, labels):
    if labels.sum() == 0 and preds.sum() == 0:
        return 1.0
    # flatten label and prediction tensors
    preds = preds.view(-1)
    # targets = targets.view(-1)
    labels = labels.contiguous().view(-1)

    intersection = (preds * labels).sum()
    smooth = 1e-2
    dice = (2. * intersection + smooth) / (preds.sum() + labels.sum() + smooth)
    # se = (intersection + smooth) / (preds.sum() + labels.sum() + smooth)
    # precision = precision_score(labels, preds)
    # f1 = f1_score(labels, preds)
    return dice