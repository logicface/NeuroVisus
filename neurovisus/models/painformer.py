import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x_visual, x_bio):
        # x_visual: [B, Dim]
        # x_bio: [B, Dim] (来自 CrossAttn 的输出)
        
        # 计算门控系数 (决定采纳多少 Bio 信息)
        combined = torch.cat([x_visual, x_bio], dim=-1)
        z = self.gate(combined)
        
        # 融合：Visual + z * Bio
        fused = x_visual + z * x_bio
        return self.norm(fused)

class BioSpectrogramConverter(nn.Module):
    """
    将一维生理信号转换为二维功率谱密度图(PSD)。
    """
    def __init__(self, n_fft=64, hop_length=4):
        super(BioSpectrogramConverter, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer('window', torch.hann_window(n_fft))

    def forward(self, x):
        # x: [B, C, L] -> 这里的 C 可能是 49
        B, C, L = x.size()
        x_flat = x.view(B * C, L) 
        
        stft_res = torch.stft(x_flat, n_fft=self.n_fft, hop_length=self.hop_length, 
                              window=self.window, return_complex=True)
        psd = torch.abs(stft_res).pow(2)
        psd = torch.clamp(psd, min=1e-8)
        psd = 10.0 * torch.log10(psd)
        
        view_psd = psd.reshape(B * C, -1) 
        min_val = view_psd.min(dim=1, keepdim=True)[0]
        max_val = view_psd.max(dim=1, keepdim=True)[0]
        psd_norm = (view_psd - min_val) / (max_val - min_val + 1e-6)
        psd_norm = psd_norm.reshape(*psd.shape)

        psd_img = F.interpolate(psd_norm.unsqueeze(1), size=(224, 224), 
                                mode='bilinear', align_corners=False)
        # 输出: [B, C, 224, 224] (例如 [4, 49, 224, 224])
        return psd_img.view(B, C, 224, 224)

class VisualEncoderViT(nn.Module):
    def __init__(self, feature_dim=128):
        super(VisualEncoderViT, self).__init__()
        # 加载 ImageNet 预训练权重 (默认输入必须是 3 通道)
        self.vit = models.vit_b_32(weights=models.ViT_B_32_Weights.DEFAULT)
        self.vit.heads = nn.Identity() 
        
        self.projector = nn.Sequential(
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, feature_dim)
        )
        
    def forward(self, x):
        feat = self.vit(x) 
        return self.projector(feat)

class PainFormer(nn.Module):
    """
    [修改版] 增加了 bio_adapter 以处理任意通道数的 fNIRS 数据
    """
    def __init__(self, num_classes=5, use_bio=True, bio_channels=3):
        super(PainFormer, self).__init__()
        self.use_bio = use_bio
        self.register_buffer('levels', torch.tensor([0, 1, 2, 3, 4], dtype=torch.float32))

        # --- A. 视觉流 ---
        self.vis_encoder = VisualEncoderViT(feature_dim=128)
        self.temporal_attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)

        # --- B. 生理流 (fNIRS) ---
        if self.use_bio:
            self.bio_converter = BioSpectrogramConverter()
            
            # [关键修复] 通道适配器: 将 N 通道 (如 49) 压缩为 3 通道 (RGB)
            if bio_channels != 3:
                print(f"🔧 [PainFormer] Initializing Bio-Adapter: {bio_channels} -> 3 channels")
                self.bio_adapter = nn.Conv2d(bio_channels, 3, kernel_size=1, bias=False)
            else:
                self.bio_adapter = nn.Identity()

            self.bio_encoder = VisualEncoderViT(feature_dim=128) 
            self.cross_attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
            fusion_dim = 256
        else:
            fusion_dim = 128

        # --- C. 分类头 ---
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def calculate_odi(self, logits):
        probs = torch.softmax(logits, dim=1)
        return torch.sum(probs * self.levels, dim=1) / 4.0

    def freeze_visual_backbone(self):
        for param in self.vis_encoder.vit.parameters(): param.requires_grad = False
        if self.use_bio:
            for param in self.bio_encoder.vit.parameters(): param.requires_grad = False
        print("❄️  ViT Backbone Frozen")

    def unfreeze_visual_backbone(self):
        for param in self.vis_encoder.vit.parameters(): param.requires_grad = True
        if self.use_bio:
            for param in self.bio_encoder.vit.parameters(): param.requires_grad = True
        print("🔥 ViT Backbone Unfrozen")

    def forward(self, video, bio=None):
        b, t, c, h, w = video.size()

        # 时序降采样
        stride = max(1, t // 8) 
        video_sampled = video[:, ::stride, :, :, :]
        if video_sampled.shape[1] > 8: 
            video_sampled = video_sampled[:, :8]
        t_new = video_sampled.shape[1]
        
        # 1. 视觉特征
        vis_feat = self.vis_encoder(video_sampled.reshape(b*t_new, c, h, w)) 
        vis_feat = vis_feat.view(b, t_new, -1)
        vis_out, _ = self.temporal_attn(vis_feat, vis_feat, vis_feat)
        vis_context = torch.mean(vis_out, dim=1) 

        # 2. 生理特征融合
        if self.use_bio and bio is not None:
            # bio: [B, C_bio, L]
            bio_img = self.bio_converter(bio)       # -> [B, 49, 224, 224]
            
            # [关键修复] 维度适配
            bio_img = self.bio_adapter(bio_img)     # -> [B, 3, 224, 224]
            
            bio_feat = self.bio_encoder(bio_img)    # -> [B, 128]
            
            vis_q = vis_context.unsqueeze(1)
            bio_kv = bio_feat.unsqueeze(1)
            attn_out, _ = self.cross_attn(query=vis_q, key=bio_kv, value=bio_kv)
            final_feat = torch.cat([vis_context, attn_out.squeeze(1)], dim=1)
        else:
            if self.use_bio:
                dummy = torch.zeros_like(vis_context)
                final_feat = torch.cat([vis_context, dummy], dim=1)
            else:
                final_feat = vis_context

        return self.classifier(final_feat)

# “”