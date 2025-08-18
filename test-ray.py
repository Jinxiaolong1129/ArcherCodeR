# 调试设备检测问题
import torch

print("=== 设备检测调试 ===")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.current_device() if torch.cuda.is_available() else 'No CUDA'}")

# 检查环境变量
import os
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

# 模拟get_device_name函数的逻辑
def debug_get_device_name():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

print(f"Device name should be: {debug_get_device_name()}")

# 检查Ray是否正确检测GPU
try:
    import ray
    print(f"Ray initialized: {ray.is_initialized()}")
    if ray.is_initialized():
        print(f"Ray resources: {ray.available_resources()}")
except ImportError:
    print("Ray not available")