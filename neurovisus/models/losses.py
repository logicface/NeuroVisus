import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftTargetCrossEntropy(nn.Module):
    """
    标签平滑交叉熵：防止模型过度自信，提高泛化能力
    """
    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()

class OrdinalRegressionLoss(nn.Module):
    """
    序数回归损失：惩罚预测值与真实等级之间的距离
    """
    def __init__(self, num_classes=5):
        super(OrdinalRegressionLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, logits, targets):
        # 将 logits 转换为概率
        probs = F.softmax(logits, dim=1)
        # 生成类别索引 [0, 1, 2, 3, 4]
        levels = torch.arange(self.num_classes).to(logits.device).float()
        
        # 计算期望值 E[y] = sum(p_i * i)
        pred_expectation = torch.sum(probs * levels, dim=1)
        
        # 真实标签已经是 0,1,2,3,4
        targets = targets.float()
        
        # 计算均方误差 (MSE) 作为序数惩罚
        return F.mse_loss(pred_expectation, targets)

class CompositeLoss(nn.Module):
    def __init__(self, ordinal_weight=1.0, smoothing=0.1, num_classes=5):
        super(CompositeLoss, self).__init__()
        self.ordinal_weight = ordinal_weight
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.ce = SoftTargetCrossEntropy()
        self.ordinal = OrdinalRegressionLoss(num_classes)

    def forward(self, logits, targets):
        # 1. 准备平滑标签 (Label Smoothing)
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.data.unsqueeze(1), 1.0 - self.smoothing)
        
        # 2. 计算分类损失
        loss_cls = self.ce(logits, true_dist)
        
        # 3. 计算序数损失
        loss_ord = self.ordinal(logits, targets)
        
        return loss_cls + self.ordinal_weight * loss_ord