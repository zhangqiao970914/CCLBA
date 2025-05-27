from torch import nn
import torch
import torch.nn.functional as F
import math
import numpy as np
from torch.autograd import Variable

class IoU_loss(torch.nn.Module):
    def __init__(self):
        super(IoU_loss, self).__init__()

    def forward(self, pred, target):
        b = pred.shape[0]
        IoU = 0.0
        for i in range(0, b):
            #compute the IoU of the foreground
            Iand1 = torch.sum(target[i, :, :, :]*pred[i, :, :, :])
            Ior1 = torch.sum(target[i, :, :, :]) + torch.sum(pred[i, :, :, :])-Iand1
            IoU1 = Iand1/(Ior1 + 1e-5)
            #IoU loss is (1-IoU1)
            IoU = IoU + (1-IoU1)

        return IoU/b
        #return IoU


class Scale_IoU(nn.Module):
    def __init__(self):
        super(Scale_IoU, self).__init__()
        self.iou = IoU_loss()

    def forward(self, scaled_preds, gt):
        loss = 0
        for pred_lvl in scaled_preds[0:]:
            loss += self.iou(torch.sigmoid(pred_lvl), gt) + self.iou(1-torch.sigmoid(pred_lvl), 1-gt)
        return loss

        
def compute_distance_map(mask):
    pos_pixel = (mask == 1).nonzero() # 返回mask中值为1的像素的坐标
    neg_pixels = (mask == 0).nonzero() # 返回mask中值为0的像素的坐标
    pos_pixel = pos_pixel.float() # 转换数据类型为float
    neg_pixels = neg_pixels.float() # 转换数据类型为float
    dist = torch.norm(neg_pixels - pos_pixel, dim=1, keepdim=True) # 计算欧几里得距离
    distance_map = torch.zeros_like(mask).float()
    for i in range(pos_pixel.shape[0]):
        x, y = int(pos_pixel[i][0]), int(pos_pixel[i][1])
        distance_map[x, y] = dist[i]
    return distance_map

    
    








def compute_cos_dis(x_sup, x_que):
    x_sup = x_sup.view(x_sup.size()[0], x_sup.size()[1], -1)
    x_que = x_que.view(x_que.size()[0], x_que.size()[1], -1)

    x_que_norm = torch.norm(x_que, p=2, dim=1, keepdim=True)
    x_sup_norm = torch.norm(x_sup, p=2, dim=1, keepdim=True)

    x_que_norm = x_que_norm.permute(0, 2, 1)
    x_qs_norm = torch.matmul(x_que_norm, x_sup_norm)

    x_que = x_que.permute(0, 2, 1)

    x_qs = torch.matmul(x_que, x_sup)
    x_qs = x_qs / (x_qs_norm + 1e-5)
    return x_qs

def sclloss(x, xt, xb):
    cosc = (1 + compute_cos_dis(x, xt)) * 0.5
    cosb = (1 + compute_cos_dis(x, xb)) * 0.5
    loss = -torch.log(cosc + 1e-5) - torch.log(1 - cosb + 1e-5)
    return loss.sum()


def structure_loss(pred, mask):
    """
    loss function (ref: F3Net-AAAI-2020)
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def dice_loss(predict, target):
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()