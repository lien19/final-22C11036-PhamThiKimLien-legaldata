# 🏛️ HỆ THỐNG TRUY XUẤT VĂN BẢN PHÁP LUẬT THUẾ THU NHẬP DOANH NGHIỆP (LEGAL RAG RETRIEVAL)

> **Đề tài tốt nghiệp:** Ứng dụng hỗ trợ truy xuất (Retrieval) văn bản Thuế Thu nhập doanh nghiệp, TNCN, GTGT, Hóa đơn  
> **Mô hình cốt lõi:** Qwen 8B Embedding tinh chỉnh bằng QLoRA & Unsloth  
> **Công nghệ lưu trữ & tìm kiếm:** Elasticsearch 8.x (BM25), Qdrant Vector Database (Cosine 4096-dim)

---

## 📌 1. Giới thiệu tổng quan

    Hệ thống văn bản quy phạm pháp luật về **Thuế Thu nhập Doanh nghiệp (TNDN)** tại Việt Nam có cấu trúc đa tầng (Luật $\rightarrow$ Nghị định $\rightarrow$ Thông tư) với nhiều lần sửa đổi, bổ sung và thay thế theo thời gian. Các công cụ tìm kiếm từ khóa truyền thống thường gặp 3 vấn đề lớn:
    1. **Mất ngữ cảnh (Context Gap):** Các Điểm/Khoản con bị tách rời độc lập, mất câu dẫn của Điều cha.
    2. **Lệch ranh giới sắc thuế:** Dễ nhầm lẫn các quy định có từ ngữ tương đồng giữa Thuế TNDN với Thuế TNCN hay Hóa đơn.
    3. **Bất cập hiệu lực thời gian:** Khó xác định chính xác điều khoản nào còn hiệu lực tại một mốc thời gian cụ thể trong quá khứ (`target_date`).

    Dự án này giải quyết triệt để các hạn chế trên bằng việc xây dựng một đường ống xử lý dữ liệu chuẩn hóa, tiêm ngữ cảnh đa tầng (**Context Injection**), khai phá mẫu đối ngẫu khó (**Hard Negative Mining**) và tinh chỉnh mô hình **Qwen 8B Embedding** để phục vụ bài toán tìm kiếm ngữ nghĩa pháp lý với độ chính xác cao.

---

## ✨ 2. Điểm nổi bật về Kỹ thuật (Technical Highlights)

    * Sử dụng Elasticsearch BM25 tìm các đoạn tương đồng cao nhưng sai đáp án.
    * *Quy tắc 1:* Lọc động theo ngày hiệu lực (`lte`, `gte`).
    * *Quy tắc 2:* Loại bỏ hoàn toàn các `positive_ids`.
    * *Quy tắc 3 (Ranh giới vàng):* Loại bỏ các đoạn cùng Điều cùng Luật với đáp án đúng để tránh false negative.
    * *Quy tắc 4:* Lọc chéo sắc thuế (TNDN vs TNCN/Hóa đơn) và bỏ qua Top-2 ứng viên nghi ngờ "người nhà".
    * **Tối ưu hóa Huấn luyện trên GPU 12GB VRAM:** Ứng dụng **Unsloth (`FastSentenceTransformer`)** và **QLoRA (4-bit)** huấn luyện mô hình 8 Tỷ tham số mượt mà trên card đồ họa phổ thông (NVIDIA GeForce RTX 3060 12GB), không bị lỗi tràn bộ nhớ.
    * **Cơ sở dữ liệu Vector Qdrant:** Lưu trữ vector 4096 chiều, đánh chỉ mục Payload `DATETIME` và `KEYWORD`, hỗ trợ lọc đa điều kiện thời gian thực.
    * **Giao diện Web Demo trực quan:** Xây dựng bằng Flask/HTML/CSS cho phép tra cứu câu hỏi tình huống và hiển thị trích dẫn pháp lý minh bạch kèm số điểm liên quan.

---

## 📂 3. Cấu trúc thư mục dự án

    ```text
    intern-legaldata/
    ├── configs/
    │   └── config.py                  # Cấu hình tập trung đường dẫn, siêu tham số, cổng kết nối
    ├── src/                           # Mã nguồn nghiệp vụ cốt lõi (Core Modules)
    │   ├── crawler.py                 # Thu thập & bóc tách văn bản .doc/.docx
    │   ├── corpus_builder.py          # Lập sổ gia phả, unpivot sửa đổi, tạo final_corpus
    │   ├── query_processor.py         # Làm sạch truy vấn, tạo positive_ids, chia Train/Eval
    │   ├── negative_miner.py          # Elasticsearch BM25 & 4 quy tắc đào Hard Negatives
    │   ├── train_qwen.py              # Huấn luyện mô hình Qwen 8B QLoRA với Unsloth
    │   ├── qdrant_indexer.py          # Sinh vector & nạp vào Qdrant (Offline Indexing)
    │   └── evaluator.py               # Chấm điểm Hit@K, MRR@10 trên tập kiểm thử độc lập
    ├── pipeline/                      # Kịch bản thực thi tuần tự (Pipeline Scripts)
    │   ├── 01_build_corpus.py         # Bước 1: Xây dựng kho tri thức văn bản luật
    │   ├── 02_prepare_queries.py      # Bước 2: Chuẩn bị & chia tập đề thi (Train/Eval)
    │   ├── 03_mine_negatives.py       # Bước 3: Đào mẫu âm tính khó với Elasticsearch
    │   ├── 04_train_model.py          # Bước 4: Huấn luyện mô hình Embedding
    │   ├── 05_index_and_search.py     # Bước 5: Nạp vector vào Qdrant & kiểm thử truy vấn
    │   └── 05_evaluate.py             # Bước phụ: Đánh giá định lượng (Base vs Tuned)
    ├── data/                          # Dữ liệu dự án
    │   ├── raw/                       # Dữ liệu văn bản thô, file ngày hiệu lực
    │   ├── processed/                 # File kho luật chuẩn (3-final_legal_corpus.json)
    │   ├── queries/                   # File câu hỏi (train_queries, eval_queries)
    │   └── outputs/                   # Checkpoints mô hình, file CSV đối chiếu
    ├── templates/                     # Giao diện HTML của Web Demo
    │   └── index.html
    ├── static/                        # CSS & JavaScript của Web Demo
    │   ├── css/style.css
    │   └── js/app.js
    ├── app.py                         # Ứng dụng Web Tra cứu (Flask Server)
    ├── requirements.txt               # Danh sách thư viện phụ thuộc
    └── README.md                      # Tài liệu hướng dẫn dự án


⚙️ 4. Yêu cầu hệ thống & Cài đặt
    Yêu cầu phần cứng khuyến nghị:
    Hệ điều hành: Windows 10/11 64-bit hoặc Linux Ubuntu 22.04.
    GPU: NVIDIA GPU ≥  12GB VRAM (đã thử nghiệm thành công trên RTX 3060 12GB).
    RAM hệ thống: ≥ 16GB (khuyến nghị 32GB).

    Bước 1: Khởi động các dịch vụ phụ trợ (Docker)
    Cài đặt Docker Desktop và khởi chạy 2 container cho Elasticsearch và Qdrant:
        # 1. Khởi chạy Elasticsearch 8.12 (Cổng 9200) - Tắt SSL/Security cho môi trường Local
        docker run -d --name es-local -p 9200:9200 -e "discovery.type=single-node" -e "xpack.security.enabled=false" -e "ES_JAVA_OPTS=-Xms1g -Xmx1g" docker.elastic.co/elasticsearch/elasticsearch:8.12.0

        # 2. Khởi chạy Qdrant Vector Database (Cổng 6333)
        docker run -d --name qdrant-local -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant:latest
    
    Bước 2: Cài đặt môi trường Python
    Khuyến nghị sử dụng Python 3.11 – 3.13:
        # Cài đặt PyTorch hỗ trợ CUDA 12.1+
        pip install torch --index-url https://download.pytorch.org/whl/cu121

        # Cài đặt toàn bộ các thư viện còn lại
        pip install -r requirements.txt

🚀 5. Hướng dẫn chạy hệ thống (Pipeline Execution)
Bạn có thể chạy tuần tự từng bước từ thư mục gốc của dự án:
    # BƯỚC 1: Xây dựng kho tri thức văn bản luật (Đã có sẵn data/3-final_legal_corpus.json có thể bỏ qua)
    python pipeline/01_build_corpus.py

    # BƯỚC 2: Chuẩn hóa truy vấn và phân chia tập Train (80%) / Eval (20%)
    python pipeline/02_prepare_queries.py

    # BƯỚC 3: Đào Hard Negatives bằng Elasticsearch BM25
    python pipeline/03_mine_negatives.py

    # BƯỚC 4: Huấn luyện mô hình Qwen 8B QLoRA với Unsloth (Tự động lưu checkpoint mỗi 100 steps)
    python pipeline/04_train_model.py

    # BƯỚC 5: Mã hóa và nạp 9.284 đoạn luật vào Qdrant Vector Database
    python pipeline/05_index_and_search.py

    # BƯỚC ĐÁNH GIÁ (Tùy chọn): Chạy kiểm thử đối sánh mô hình Gốc vs Mô hình Fine-tuned
    python pipeline/05_evaluate.py

🌐 6. Khởi chạy Giao diện Web Tra cứu (Web Demo)
Sau khi dữ liệu đã được nạp hoàn tất vào Qdrant:
python app.py
    Mở trình duyệt web và truy cập: http://localhost:5000
    Tính năng:
    Chọn các câu hỏi tình huống thực tế về Thuế TNDN.
    Tự động sinh vector và truy vấn ngữ nghĩa song song với bộ lọc ngày hiệu lực.
    Hiển thị Top 3 căn cứ pháp lý liên quan nhất kèm nhãn Điều/Khoản, mức độ tương đồng Cosine và trạng thái hiệu lực.

📊 7. Kết quả thực nghiệm (Evaluation Benchmark)
    Đánh giá trên tập kiểm thử độc lập gồm các câu hỏi tình huống thực tế về Thuế TNDN:
    Chỉ số (Metric),Mô hình Gốc (Base),Sau Fine-tune (Tuned),Tăng trưởng
    Hit@1 (Chính xác Top 1),47.80%,64.29%,+ 16.48%
    Hit@5 (Nằm trong Top 5),85.44%,92.86%,+ 7.42%
    Hit@10 (Nằm trong Top 10),92.03%,97.25%,+ 5.22%
    MRR@10 (Thứ hạng trung bình),62.12%,76.27%,+ 14.16%
    NDCG@10 (Độ liên quan tổng thể),69.37%,81.44%,+ 12.07%

