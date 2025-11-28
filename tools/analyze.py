import torch
import cv2
import numpy as np
import os
import argparse
from torchvision import transforms

# 引入新架构的组件
from neurovisus.models import get_model
from neurovisus.utils.visualization.cam import GradCAM
from neurovisus.utils.visualization.dashboard import draw_dashboard

def main():
    parser = argparse.ArgumentParser(description="NeuroVisus Offline Analyzer")
    parser.add_argument('--sample', type=str, required=True, help='Path to .pt sample file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to .pth model weights')
    parser.add_argument('--output', type=str, default='analysis_result.mp4')
    parser.add_argument('--arch', type=str, default='painformer')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. 加载模型
    print(f"🏭 Loading model: {args.arch}")
    model = get_model(args.arch, num_classes=5, use_bio=True, bio_channels=49).to(device)
    
    # 加载权重
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
    model.eval()

    # 2. 准备 CAM
    # 挂载到 ViT 最后一层 LayerNorm (针对 PainFormer)
    target_layer = model.vis_encoder.vit.encoder.layers[-1].ln_1
    grad_cam = GradCAM(model, target_layer)
    
    # 3. 加载数据
    print(f"📂 Loading sample: {args.sample}")
    data = torch.load(args.sample)
    video_raw = data['video'].float() / 255.0
    if video_raw.size(1) == 1: video_raw = video_raw.repeat(1, 3, 1, 1)
    
    bio_data = data['bio']
    label = data['label']

    # 预处理输入
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    video_input = torch.stack([normalize(frame) for frame in video_raw])
    
    # 4. 推理
    print("🧠 Running inference...")
    video_tensor = video_input.unsqueeze(0).to(device)
    bio_tensor = bio_data.unsqueeze(0).to(device)
    
    # 获取 CAM Map
    cam_map = grad_cam(video_tensor) # 这里会触发一次前向
    
    # 获取 ODI (需要再跑一次 forward 获取 logits，或者在 GradCAM 里改写，这里分开跑比较清晰)
    with torch.no_grad():
        logits = model(video_tensor, bio=bio_tensor)
        odi = model.calculate_odi(logits).item() * 100

    # 5. 渲染视频
    print(f"🎥 Rendering video to {args.output}...")
    T, C, H, W = video_raw.shape
    
    # 渲染第一帧确定尺寸
    dummy_frame = (video_raw[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    dummy_res = draw_dashboard(dummy_frame, cam_map, bio_data, 0, T, odi, label)
    out_h, out_w = dummy_res.shape[:2]
    
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), 8, (out_w, out_h))
    
    for i in range(T):
        frame_img = (video_raw[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        frame_img = cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR)
        frame_out = draw_dashboard(frame_img, cam_map, bio_data, i, T, odi, label)
        writer.write(frame_out)
        
    writer.release()
    print("✅ Done!")

if __name__ == "__main__":
    main()