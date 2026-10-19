import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train_qwen import *

if __name__ == "__main__":
    print("🚀 Đang chạy Bước 4: Huấn luyện mô hình Qwen 8B...")
    train_qwen_8b_quantized_unsloth()
    print("🎉 Hoàn thành Bước 4!")