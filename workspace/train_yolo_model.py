import torch

# 限制 GPU 使用 50% 的显存
torch.cuda.set_per_process_memory_fraction(0.5, device=0)

from ultralytics import YOLO

if __name__ == '__main__':  # 关键：防止 Windows multiprocessing 错误
    model = YOLO("yolo11n.yaml")   # 确保模型文件名正确
    
    model.train(data="african-wildlife.yaml", epochs=1, batch=16, save_period=100, save=True, resume=True)