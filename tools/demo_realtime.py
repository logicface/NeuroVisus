import torch
import cv2
import dlib
import numpy as np
import collections
import argparse
from torchvision import transforms
import torch.nn.functional as F

from neurovisus.models import get_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='0', help='Path to video or camera index')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='demo_result.mp4')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. 模型初始化
    print("🚀 Initializing Real-time Demo...")
    model = get_model('painformer', num_classes=5, use_bio=True).to(device)
    model.eval()
    
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})

    # CAM Hook
    activations, gradients = {}, {}
    def save_act(name): return lambda m, i, o: activations.update({name: o})
    def save_grad(name): return lambda m, i, o: gradients.update({name: o[0]})
    
    target_layer = model.vis_encoder.vit.encoder.layers[-1].ln_1
    target_layer.register_forward_hook(save_act('act'))
    target_layer.register_full_backward_hook(save_grad('grad'))

    # 2. 视频流准备
    video_src = int(args.video) if args.video.isdigit() else args.video
    cap = cv2.VideoCapture(video_src)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    face_detector = dlib.get_frontal_face_detector()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    buffer = collections.deque(maxlen=32)
    odi_history = collections.deque(maxlen=5)

    print(f"🔴 Recording to {args.output}. Press 'q' to stop.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        display_frame = frame.copy()
        
        # 人脸检测
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 0)
        
        if faces:
            face = faces[0]
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            
            # 提取 ROI 并预处理
            try:
                face_crop = cv2.resize(frame[max(0,y1):y2, max(0,x1):x2], (224, 224))
                tensor = torch.from_numpy(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                tensor = normalize(tensor.permute(2,0,1).float()/255.0)
                buffer.append(tensor)
            except: pass
            
            # 推理条件满足
            if len(buffer) == 32:
                seq = torch.stack(list(buffer)).unsqueeze(0).to(device)
                
                # Forward
                model.zero_grad()
                logits = model(seq, bio=None) # Demo 模式暂无 Bio 信号
                odi = model.calculate_odi(logits).item() * 100
                odi_history.append(odi)
                
                # Backward (CAM)
                score = logits[:, logits.argmax(1)]
                score.backward()
                
                # 简单处理 CAM (只取最后一帧)
                # ViT CAM 处理逻辑需与 utils/cam.py 一致，此处简化用于实时演示
                # ... (此处省略复杂的 reshape，仅显示 ODI 条以保证帧率) ...
                
                avg_odi = sum(odi_history) / len(odi_history)
                
                # 绘制 ODI 条
                color = (0, 255, 0) if avg_odi < 30 else (0, 0, 255)
                cv2.rectangle(display_frame, (x1, y1-30), (x1 + int(avg_odi)*2, y1-10), color, -1)
                cv2.putText(display_frame, f"ODI: {avg_odi:.1f}", (x1, y1-40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

        writer.write(display_frame)
        cv2.imshow('NeuroVisus Live', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()