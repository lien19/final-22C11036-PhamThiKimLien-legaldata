import sys, os
# Đảm bảo luôn tìm thấy thư mục gốc
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluator import *

if __name__ == "__main__":
    print("=" * 65)
    print("🚀 BẮT ĐẦU ĐÁNH GIÁ THỰC NGHIỆM TRÊN TẬP TEST ĐỘC LẬP")
    print("=" * 65)

    base_scores, tuned_scores = run_evaluation_unsloth()
    
    print("\n🎉 ĐÃ HOÀN THÀNH TOÀN BỘ ĐÁNH GIÁ!")

