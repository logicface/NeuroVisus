import torch
import cv2
import numpy as np

class GradCAM:
    """
    梯度加权类激活映射 (Grad-CAM) 实现
   
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # 注册钩子
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def _reshape_vit(self, x):
        """处理 ViT 特有的 Patch 维度"""
        # [B, N, D] -> [B, D, H, W]
        # 假设 N = (224/32)^2 + 1 = 50
        if len(x.shape) == 3:
            x = x[:, 1:, :] # 去掉 CLS Token
            b, n, d = x.size()
            h = w = int(n**0.5)
            return x.permute(0, 2, 1).view(b, d, h, w)
        return x

    def __call__(self, video_tensor, target_class_index=None):
        # 1. 前向传播
        output = self.model(video_tensor)
        if target_class_index is None:
            target_class_index = torch.argmax(output, dim=1).item()
        
        # 2. 反向传播
        self.model.zero_grad()
        one_hot_output = torch.zeros_like(output)
        one_hot_output[0][target_class_index] = 1
        output.backward(gradient=one_hot_output, retain_graph=True)
        
        # 3. 提取特征图与梯度
        grads = self.gradients
        acts = self.activations
        
        # 兼容 ViT
        if len(grads.shape) == 3:
            grads = self._reshape_vit(grads)
            acts = self._reshape_vit(acts)
            
        # 4. 计算 CAM (这里取时间序列的平均或中间帧)
        # 假设输入是 [B*T, C, H, W] 或 [B, C, H, W]
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * acts, dim=1)
        cam = torch.relu(cam)
        
        # 取平均并归一化
        cam = torch.mean(cam, dim=0).detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() + 1e-7)
        
        return cam