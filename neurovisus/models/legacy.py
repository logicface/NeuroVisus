import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import models
# [说明] 这里放置源 model.py 中被注释掉的 ResNet/VGG/LSTM 代码
# 为了保持整洁，这里仅提供基类结构，你需要把原文件里注释的内容取消注释并贴过来。


# =============================================================================
# Section 2: 历史遗留架构 (Legacy/Deprecated) - CNN & LSTM
# [导师注释] 这些代码已注释，保留用于基线对比实验。
# 如需使用，请取消注释并更新 get_model 工厂函数。
# =============================================================================

class BioEncoder(nn.Module):
    """旧版生理信号编码器 (1D CNN) - 弃用，已被 BioSpectrogramConverter 替代"""
    def __init__(self, input_channels=3, feature_dim=128):
        super(BioEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(128, feature_dim)
    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class VisualEncoderResNet(nn.Module):
    """旧版 ResNet18 编码器"""
    def __init__(self, feature_dim=128):
        super(VisualEncoderResNet, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(512, feature_dim)
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class BaseFusionModel(nn.Module):
    def __init__(self, num_classes=5, use_bio=True):
        super(BaseFusionModel, self).__init__()
        self.use_bio = use_bio
        self.register_buffer('levels', torch.tensor([0, 1, 2, 3, 4], dtype=torch.float32))
    def calculate_odi(self, logits):
        probs = torch.softmax(logits, dim=1)
        return torch.sum(probs * self.levels, dim=1) / 4.0

class IPFusionNetResNet(BaseFusionModel):
    def __init__(self, num_classes=5, use_bio=True):
        super(IPFusionNetResNet, self).__init__(num_classes, use_bio)
        self.vis_encoder = VisualEncoderResNet(feature_dim=128)
        self.time_attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        if self.use_bio:
            self.bio_encoder = BioEncoder(input_channels=3, feature_dim=128)
            self.cross_attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
            fusion_dim = 256
        else:
            fusion_dim = 128
        self.classifier = nn.Sequential(nn.Linear(fusion_dim, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, num_classes))
    def forward(self, video, bio=None):
        b, t, c, h, w = video.size()
        vis_feat = self.vis_encoder(video.view(b*t, c, h, w)).view(b, t, -1)
        vis_out, _ = self.time_attn(vis_feat, vis_feat, vis_feat)
        vis_context = torch.mean(vis_out, dim=1)
        if self.use_bio and bio is not None:
            bio_feat = self.bio_encoder(bio)
            attn_out, _ = self.cross_attn(vis_context.unsqueeze(1), bio_feat.unsqueeze(1), bio_feat.unsqueeze(1))
            final_feat = torch.cat([vis_context, attn_out.squeeze(1)], dim=1)
        else:
            final_feat = vis_context
        return self.classifier(final_feat)

class IPFusionNetVGG(BaseFusionModel):
    def __init__(self, num_classes=5, use_bio=True):
        super(IPFusionNetVGG, self).__init__(num_classes, use_bio)
        # 需要引入 VisualEncoderVGG (此处省略详细定义，逻辑同上)
        pass

class IPFusionNetLSTM(BaseFusionModel):
    # LSTM 时序模型 (此处省略详细定义)
    pass


# =============================================================================
# Section 3: 工厂函数 (Factory Function)
# =============================================================================
