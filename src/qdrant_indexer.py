import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import uuid
from qdrant_client import QdrantClient
# ⭐️ THÊM SparseVectorParams và thư viện fastembed
from fastembed import SparseTextEmbedding
from qdrant_client.models import Distance, PointStruct, VectorParams, PayloadSchemaType, SparseVectorParams

from tqdm import tqdm
from unsloth import FastSentenceTransformer
from transformers.trainer_utils import get_last_checkpoint

import config

"""
1.Dùng mô hình bạn vừa train (checkpoint-700) để biến toàn bộ kho luật thành vector.
2.Tự động lấy kích thước vector (dimension) của Qwen 8B.
3.Đẩy từng batch lên Qdrant kèm theo toàn bộ thông tin metadata (Gia phả, nội dung, ngày hiệu lực, sắc thuế).
"""

def index_corpus_to_qdrant_with_dense():
    collection_name = config.COLLECTION_NAME_DENSE

    # -------------------------------------------------------------
    # BƯỚC 1: KẾT NỐI QDRANT & NẠP MODEL
    # -------------------------------------------------------------
    print("-> Đang kết nối tới Qdrant Server...")
    client = QdrantClient(url=config.QDRANT_URL)

    # ⭐️ TỰ ĐỘNG TÌM CHECKPOINT MỚI NHẤT
    latest_checkpoint = get_last_checkpoint(config.MODEL_DIR)

    # Nếu có checkpoint thì lấy checkpoint, nếu đã train xong 100% thì lấy thư mục gốc
    index_model_path = latest_checkpoint if latest_checkpoint else config.MODEL_DIR

    print(f"-> Đang nạp mô hình đã train: {index_model_path}...")
    model = FastSentenceTransformer.from_pretrained(
        model_name=index_model_path,
        max_seq_length=1024,
        load_in_4bit=True,
        load_in_16bit=False,
    )

    # Tự động lấy số chiều vector của mô hình (thường là 4096 hoặc 1536)
    vector_dim = model.get_sentence_embedding_dimension()
    print(f"-> Kích thước Vector của mô hình: {vector_dim} chiều.")

    # -------------------------------------------------------------
    # BƯỚC 2: TẠO COLLECTION TRONG QDRANT
    # -------------------------------------------------------------
    # print(f"-> Đang khởi tạo Collection '{config.COLLECTION_NAME}'...")
    # # Xóa collection cũ nếu đã tồn tại để làm sạch dữ liệu
    # if client.collection_exists(config.COLLECTION_NAME):
    #     client.delete_collection(config.COLLECTION_NAME)
    #     print(f"   (Đã xóa collection cũ '{config.COLLECTION_NAME}')")

    # client.create_collection(
    #     collection_name=config.COLLECTION_NAME,
    #     vectors_config=VectorParams(
    #         size=vector_dim,
    #         distance=Distance.COSINE,
    #     ),
    # )    



    # - Để False: Nếu đứt gánh giữa đường thì tự động CHẠY TIẾP (Resume) từ batch cũ.
    # - Để True: Khi nào muốn XÓA SẠCH để nạp lại model mới tinh từ đầu.
    FORCE_RECREATE = False 
    # Kiểm tra theo công tắc:
    if FORCE_RECREATE and client.collection_exists(collection_name):
        client.delete_collection(collection_name)
        print("-> Đã xóa sạch collection cũ theo yêu cầu FORCE_RECREATE!")

    # Nếu collection chưa có thì tạo mới
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_dim,
                distance=Distance.COSINE,
            ),
        )   
        # Tạo payload index...
        start_index = 0
    else:
        # Nếu đã có sẵn -> Đếm số điểm đã nạp để CHẠY NỐI TIẾP!
        start_index = client.count(collection_name).count
        print(f"-> Phát hiện dữ liệu cũ! Chạy nối tiếp từ văn bản thứ: {start_index}...")


    # -------------------------------------------------------------
    # BƯỚC 3: ĐỌC KHO LUẬT VÀ SINH VECTOR THEO BATCH
    # -------------------------------------------------------------
    print(f"-> Đang đọc kho luật từ {config.CORPUS_FINAL_JSON}...")
    with open(config.CORPUS_FINAL_JSON, "r", encoding="utf-8") as f:
        corpus_data  = json.load(f)

    corpus  = (
        list(corpus_data.values())
        if isinstance(corpus_data, dict)
        else corpus_data
    )
    # ⭐️ TỰ ĐỘNG TẠO PAYLOAD INDEX CHO TẤT CẢ CÁC TRƯỜNG (TRỪ TEXT DÀI)
    # =================================================================
    EXCLUDE_FIELDS = {"hierarchy_context", "content", "search_text", "uuid"}

    for field_name in corpus[0].keys():
        if field_name not in EXCLUDE_FIELDS:
            # Nếu là trường ngày tháng -> Phải tạo kiểu DATETIME để sau này lọc lte, gte
            if "date" in field_name.lower():
                schema = PayloadSchemaType.DATETIME
            else:
                schema = PayloadSchemaType.KEYWORD

            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=schema,
            )

    total_chunks = len(corpus)
    print(f"-> Bắt đầu mã hóa và nạp {total_chunks} văn bản luật vào Qdrant...\n" )

    # Chạy vòng lặp theo batch để không bị nghẽn RAM/VRAM
    for i in tqdm( range(start_index, total_chunks, config.QDRANT_BATCH_SIZE), desc="Đang đẩy lên Qdrant" ):
        batch_docs = corpus[i : i + config.QDRANT_BATCH_SIZE]

        # Lấy trường search_text (đã chứa đủ Gia phả + Nội dung) để encode
        texts_to_encode = [doc.get("search_text", "") for doc in batch_docs]

        # Sinh vector (chuẩn hóa độ dài vector để tính Cosine chính xác)
        embeddings = model.encode( texts_to_encode, normalize_embeddings=True, show_progress_bar=False )

        # Đóng gói từng điểm (Point) gồm: ID, Vector và Toàn bộ Payload (Metadata)
        points = []
        for doc, vec in zip(batch_docs, embeddings):
            # Qdrant yêu cầu ID dạng UUID hợp lệ
            point_id = doc.get("uuid")
            if not point_id:
                point_id = str( uuid.uuid5(uuid.NAMESPACE_DNS, doc.get("chunk_id", "")) )

            points.append(
                PointStruct(
                    id=point_id,
                    vector=vec.tolist(),
                    payload=doc, # ⭐️ PAYLOAD: Nhét toàn bộ thông tin để sau này RAG lấy ra trả lời 
                )
            )
                    
        # Đẩy batch lên Qdrant
        client.upsert(collection_name=collection_name, points=points)

    print("\n" + "=" * 65)
    print(f"🎉 THÀNH CÔNG! ĐÃ NẠP TOÀN BỘ KHO LUẬT VÀO QDRANT!")
    print(f"👉 Kiểm tra ngay trên giao diện web: {config.QDRANT_URL}/dashboard#/collections/{collection_name}" )
    print("=" * 65)




def index_corpus_to_qdrant_with_dense_and_bm25():
    # -------------------------------------------------------------
    # BƯỚC 1: KẾT NỐI QDRANT & NẠP MODEL
    # -------------------------------------------------------------
    print("-> Đang kết nối tới Qdrant Server...")
    client = QdrantClient(url=config.QDRANT_URL)

    # ⭐️ TỰ ĐỘNG TÌM CHECKPOINT MỚI NHẤT
    latest_checkpoint = get_last_checkpoint(config.MODEL_DIR)

    # Nếu có checkpoint thì lấy checkpoint, nếu đã train xong 100% thì lấy thư mục gốc
    index_model_path = latest_checkpoint if latest_checkpoint else config.MODEL_DIR

    print(f"-> Đang nạp mô hình đã train: {index_model_path}...")
    model = FastSentenceTransformer.from_pretrained(
        model_name=index_model_path,
        max_seq_length=1024,
        load_in_4bit=True,
        load_in_16bit=False,
    )

    # Tự động lấy số chiều vector của mô hình (thường là 4096 hoặc 1536)
    vector_dim = model.get_sentence_embedding_dimension()
    print(f"-> Kích thước Vector của mô hình: {vector_dim} chiều.")

    # ⭐️ Nạp mô hình BM25 siêu nhẹ (chạy bằng CPU)
    print("-> Đang nạp mô hình BM25 (FastEmbed)...")
    bm25_model = SparseTextEmbedding("Qdrant/bm25")

    # -------------------------------------------------------------
    # BƯỚC 2: TẠO COLLECTION TRONG QDRANT
    # -------------------------------------------------------------
    # print(f"-> Đang khởi tạo Collection '{config.COLLECTION_NAME}'...")
    # # Xóa collection cũ nếu đã tồn tại để làm sạch dữ liệu
    # if client.collection_exists(config.COLLECTION_NAME):
    #     client.delete_collection(config.COLLECTION_NAME)
    #     print(f"   (Đã xóa collection cũ '{config.COLLECTION_NAME}')")

    # client.create_collection(
    #     collection_name=config.COLLECTION_NAME,
    #     vectors_config=VectorParams(
    #         size=vector_dim,
    #         distance=Distance.COSINE,
    #     ),
    # )    



    # - Để False: Nếu đứt gánh giữa đường thì tự động CHẠY TIẾP (Resume) từ batch cũ.
    # - Để True: Khi nào muốn XÓA SẠCH để nạp lại model mới tinh từ đầu.
    FORCE_RECREATE = True 
    # Kiểm tra theo công tắc:
    if FORCE_RECREATE and client.collection_exists(config.COLLECTION_NAME):
        client.delete_collection(config.COLLECTION_NAME)
        print("-> Đã xóa sạch collection cũ theo yêu cầu FORCE_RECREATE!")

    # Nếu collection chưa có thì tạo mới
    if not client.collection_exists(config.COLLECTION_NAME):
        # ⭐️ SỬA ĐOẠN CREATE: Khai báo ngăn 'dense' và ngăn 'bm25'
        client.create_collection(
            collection_name=config.COLLECTION_NAME,
            vectors_config={"dense": VectorParams(size=vector_dim, distance=Distance.COSINE) }, # Qwen
            sparse_vectors_config={"bm25": SparseVectorParams() },  # ⭐️ Thêm ngăn chứa BM25
        )
        # Tạo payload index...
        start_index = 0
    else:
        # Nếu đã có sẵn -> Đếm số điểm đã nạp để CHẠY NỐI TIẾP!
        start_index = client.count(config.COLLECTION_NAME).count
        print(f"-> Phát hiện dữ liệu cũ! Chạy nối tiếp từ văn bản thứ: {start_index}...")


    # -------------------------------------------------------------
    # BƯỚC 3: ĐỌC KHO LUẬT VÀ SINH VECTOR THEO BATCH
    # -------------------------------------------------------------
    print(f"-> Đang đọc kho luật từ {config.CORPUS_FINAL_JSON}...")
    with open(config.CORPUS_FINAL_JSON, "r", encoding="utf-8") as f:
        corpus_data  = json.load(f)

    corpus  = (
        list(corpus_data.values())
        if isinstance(corpus_data, dict)
        else corpus_data
    )
    # ⭐️ TỰ ĐỘNG TẠO PAYLOAD INDEX CHO TẤT CẢ CÁC TRƯỜNG (TRỪ TEXT DÀI)
    # =================================================================
    EXCLUDE_FIELDS = {"hierarchy_context", "content", "search_text", "uuid"}

    for field_name in corpus[0].keys():
        if field_name not in EXCLUDE_FIELDS:
            # Nếu là trường ngày tháng -> Phải tạo kiểu DATETIME để sau này lọc lte, gte
            if "date" in field_name.lower():
                schema = PayloadSchemaType.DATETIME
            else:
                schema = PayloadSchemaType.KEYWORD

            client.create_payload_index(
                collection_name=config.COLLECTION_NAME,
                field_name=field_name,
                field_schema=schema,
            )    

    total_chunks = len(corpus)
    print(f"-> Bắt đầu mã hóa và nạp {total_chunks} văn bản luật vào Qdrant...\n" )

    # Chạy vòng lặp theo batch để không bị nghẽn RAM/VRAM
    for i in tqdm( range(start_index, total_chunks, config.QDRANT_BATCH_SIZE), desc="Đang đẩy lên Qdrant" ):
        batch_docs = corpus[i : i + config.QDRANT_BATCH_SIZE]

        # Lấy trường search_text (đã chứa đủ Gia phả + Nội dung) để encode
        texts_to_encode = [doc.get("search_text", "") for doc in batch_docs]

        # Sinh vector (chuẩn hóa độ dài vector để tính Cosine chính xác)
        embeddings = model.encode( texts_to_encode, normalize_embeddings=True, show_progress_bar=False )

        # ⭐️ Sinh vector BM25 (Sparse)
        sparse_embeddings = list(bm25_model.embed(texts_to_encode))
    
        # Đóng gói từng điểm (Point) gồm: ID, Vector và Toàn bộ Payload (Metadata)
        points = []
        # ⭐️ SỬA VÒNG LẶP ZIP THÊM sparse_embeddings
        for doc, vec, s_vec in zip(batch_docs, embeddings, sparse_embeddings):
            point_id = doc.get("uuid")
            if not point_id:
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.get("chunk_id", ""))  )

            points.append(
                PointStruct(
                    id=point_id,
                    # ⭐️ Lưu cả 2 loại vector vào cùng 1 Point
                    vector={"dense": vec.tolist(),  # Vector Qwen
                            "bm25": s_vec.as_object(),  # Vector BM25
                    },
                    payload=doc,
                )
            )

        # Đẩy batch lên Qdrant
        client.upsert(collection_name=config.COLLECTION_NAME, points=points)

    print("\n" + "=" * 65)
    print(f"🎉 THÀNH CÔNG! ĐÃ NẠP TOÀN BỘ KHO LUẬT VÀO QDRANT!")
    print(f"👉 Kiểm tra ngay trên giao diện web: {config.QDRANT_URL}/dashboard#/collections/{config.COLLECTION_NAME}" )
    print("=" * 65)


