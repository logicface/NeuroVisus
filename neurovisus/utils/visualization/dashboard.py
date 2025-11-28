import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def draw_dashboard(frame_img, cam_map, bio_data, current_idx, total_frames, odi, label_idx):
    """
    绘制包含 CAM 热力图和生理信号波形的仪表盘
   
    """
    H, W, _ = frame_img.shape
    
    # --- 1. 左侧：人脸 + CAM ---
    if cam_map is not None:
        cam_resized = cv2.resize(cam_map, (W, H))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame_img, 0.7, heatmap, 0.3, 0)
    else:
        overlay = frame_img
    
    # --- 2. 右侧：生理信号波形 ---
    # Bio Data: [3, 512] -> [GSR, ECG, EMG]
    bio_len = bio_data.shape[1]
    current_bio_ptr = int((current_idx / total_frames) * bio_len)
    
    fig, axes = plt.subplots(3, 1, figsize=(5, 4), dpi=80)
    signal_names = ['GSR (Stress)', 'ECG (Heart)', 'EMG (Muscle)']
    colors = ['green', 'red', 'blue']
    
    for i, ax in enumerate(axes):
        sig = bio_data[i, :].numpy()
        ax.plot(sig, color='lightgray', linewidth=1)
        ax.plot(range(current_bio_ptr), sig[:current_bio_ptr], color=colors[i], linewidth=1.5)
        ax.axvline(x=current_bio_ptr, color='orange', linestyle='--')
        ax.set_title(signal_names[i], fontsize=8, pad=2)
        ax.axis('off')
        
    plt.tight_layout()
    
    # Matplotlib -> OpenCV
    canvas = FigureCanvas(fig)
    canvas.draw()
    graph_img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    graph_img = graph_img.reshape(canvas.get_width_height()[::-1] + (3,))
    graph_img = cv2.cvtColor(graph_img, cv2.COLOR_RGB2BGR)
    plt.close(fig)
    
    # 调整大小拼接
    graph_h, graph_w = graph_img.shape[:2]
    scale = H / graph_h
    graph_resized = cv2.resize(graph_img, (int(graph_w * scale), H))
    
    combined = np.hstack((overlay, graph_resized))
    
    # --- 3. 顶部信息 ---
    header = np.zeros((60, combined.shape[1], 3), dtype=np.uint8)
    combined = np.vstack((header, combined))
    
    label_map = {0:'BL1', 1:'PA1', 2:'PA2', 3:'PA3', 4:'PA4'}
    gt_text = f"GT: {label_map.get(int(label_idx), str(label_idx))}"
    odi_text = f"ODI: {odi:.1f}%"
    odi_color = (0, 255, 0) if odi < 40 else (0, 165, 255) if odi < 70 else (0, 0, 255)
    
    cv2.putText(combined, gt_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(combined, odi_text, (250, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, odi_color, 2)
    
    return combined