import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.qdrant_indexer import *

if __name__ == "__main__":
    print("🚀 Đang chạy Bước 5: Nạp Qdrant...")
    index_corpus_to_qdrant_with_dense()
    # index_corpus_to_qdrant_with_dense_and_bm25()
    print("🎉 Hoàn thành Bước 5!")