import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.negative_miner import *
import config

if __name__ == "__main__":
    print("🚀 Đang chạy Bước 3: Đào Hard Negatives...")
    
    # 1. Ép hệ thống không chạy qua Proxy đối với localhost/127.0.0.1
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"

    # 2. Đổi 'localhost' thành '127.0.0.1'
    es = Elasticsearch(
        config.ES_HOST,      # ES_HOST = "http://127.0.0.1:9200"
        request_timeout=30,
        verify_certs=False,
    )

    # 3. In thử thông tin chi tiết (nếu lỗi sẽ hiện rõ nguyên nhân thay vì tự raise ConnectionError)
    try:
        cluster_info = es.info()
        print("🎉 KẾT NỐI ELASTICSEARCH THÀNH CÔNG!")
        print(f"-> Tên Cluster: {cluster_info.get('cluster_name')}")
        print(f"-> Phiên bản: {cluster_info.get('version', {}).get('number')}\n")
    except Exception as e:
        print(f"❌ Vẫn bị lỗi, chi tiết lỗi thật sự là: {e}")
        exit()                    

    
    mine_hard_negatives(
        es_client=es,
        num_negatives=5,  # Lấy tối đa 5 Negatives
        top_k_search=30,  # Quét Top 20 từ ES
    )

    print("🎉 Hoàn thành Bước 3!")



    