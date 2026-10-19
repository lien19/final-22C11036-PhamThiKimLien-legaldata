import os

# =====================================================================
# 1. QUẢN LÝ TOÀN BỘ ĐƯỜNG DẪN THƯ MỤC TRONG DỰ ÁN (DIRECTORIES)
# =====================================================================
# Thư mục gốc của toàn bộ dự án
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))

# ⭐️ Khai báo đầy đủ tất cả các folder trong dự án:
CONFIG_DIR      = os.path.join(BASE_DIR, "config")
SRC_DIR         = os.path.join(BASE_DIR, "src")
PIPELINE_DIR    = os.path.join(BASE_DIR, "pipeline")
DATA_DIR        = os.path.join(BASE_DIR, "data")

# Các folder giao diện Web Demo:
TEMPLATES_DIR   = os.path.join(BASE_DIR, "templates")
STATIC_DIR      = os.path.join(BASE_DIR, "static")
STATIC_CSS_DIR  = os.path.join(STATIC_DIR, "css")
STATIC_JS_DIR   = os.path.join(STATIC_DIR, "js")

# Folder chứa model đã train:
MODEL_DIR       = os.path.join(BASE_DIR, "qwen8b_legal_unsloth_lora")
CHECKPOINT_DIR  = os.path.join(MODEL_DIR, "checkpoint-700")

# Tự động tạo các folder cần thiết nếu chưa có
for folder in [DATA_DIR, TEMPLATES_DIR, STATIC_CSS_DIR, STATIC_JS_DIR]:
    os.makedirs(folder, exist_ok=True)


# =====================================================================
# 2. ĐƯỜNG DẪN CÁC FILE DỮ LIỆU CỤ THỂ (TRONG FOLDER DATA)
# =====================================================================
# File dữ liệu văn bản luật (Corpus)
RAW_LEGAL_CSV       = os.path.join(DATA_DIR, "3-thong ke - luat.csv")
CORPUS_BASE_JSONL   = os.path.join(DATA_DIR, "3-thong ke - luat-jsonl.jsonl")
CORPUS_BASE_JSON    = os.path.join(DATA_DIR, "3-thong ke - luat-json.json")
CORPUS_ID_CSV       = os.path.join(DATA_DIR, "3-thong ke - luat-id-uiud.csv")
# unpivot_group_data
UNPIVOT_CSV         = os.path.join(DATA_DIR, "3-thong ke - luat-output.csv")
EDIT_MAP_JSON       = os.path.join(DATA_DIR, "3-thong ke - luat-unpivoted_edit_map.json")
# convert_metadata_csv_to_json
EFFECTIVE_DATE_CSV  = os.path.join(DATA_DIR, "3-thong ke - luat-effectivedate.csv")
METADATA_JSON       = os.path.join(DATA_DIR, "3-thong ke - luat-effectivedate-json.json")
# merge_corpus_data
CORPUS_FINAL_JSON   = os.path.join(DATA_DIR, "3-final_legal_corpus.json")
CORPUS_FINAL_JSONL  = os.path.join(DATA_DIR, "3-final_legal_corpus.jsonl")
CORPUS_MAP_FINAL_JSON = os.path.join(DATA_DIR, "3-final_legal_expanded_content_map.json")
# =====================================================================
# File dữ liệu câu hỏi (Queries)
RAW_QUERY_CSV       = os.path.join(DATA_DIR, "3-query_check_5.csv")
TRAIN_QUERIES_JSONL = os.path.join(DATA_DIR, "3-train_queries_for_mining.jsonl")
EVAL_QUERIES_JSONL  = os.path.join(DATA_DIR, "3-eval_queries.jsonl")
# =====================================================================
# File dữ liệu sau khi đào Hard Negatives 
MINED_TRAIN_JSONL   = os.path.join(DATA_DIR, "3-train_queries_for_embedding_with_neg.jsonl")
INSPECT_CSV_PATH    = os.path.join(DATA_DIR, "3-inspect_mined_negatives.csv")
# =====================================================================
EVAL_RESULTS_JSON       = os.path.join(DATA_DIR, "eval_results_json.json")
EVAL_RESULTS_TABLE_JSON = os.path.join(DATA_DIR, "eval_results_table_json.json")
EVAL_RESULTS_TABLE_CSV  = os.path.join(DATA_DIR, "eval_results_table_csv.csv")


# =====================================================================
# 3. CẤU HÌNH ELASTICSEARCH (ĐÀO HARD NEGATIVES)
# =====================================================================
ES_HOST            = "http://127.0.0.1:9200"
# ES_INDEX_NAME      = "vietnamese_legal_corpus"
# ES_INDEX_NAME      = "vietnamese_legal_corpus_clean"
# ES_INDEX_NAME      = "vietnamese_legal_corpus_clean_1"
# ES_INDEX_NAME      = "vietnamese_legal_corpus_clean_2"
# ES_INDEX_NAME      = "vietnamese_legal_corpus_clean_3"
# ES_INDEX_NAME      = "vietnamese_legal_corpus_clean_4"
ES_INDEX_NAME      = "vietnamese_legal_corpus_clean_5"
ES_REQUEST_TIMEOUT = 30


# =====================================================================
# 4. CẤU HÌNH HUẤN LUYỆN QWEN 8B (UNSLOTH QLORA)
# =====================================================================
BASE_MODEL_ID      = "Qwen/Qwen3-Embedding-8B"  # hoặc Alibaba-NLP/gte-Qwen2-7B-instruct

# Siêu tham số (Hyperparameters)
MAX_SEQ_LENGTH     = 1024
TRAIN_BATCH_SIZE   = 4
GRAD_ACCUM_STEPS   = 4
NUM_TRAIN_EPOCHS   = 3
LEARNING_RATE      = 1e-4
WARMUP_RATIO       = 0.1
SAVE_STEPS         = 100
SAVE_TOTAL_LIMIT   = 2


# =====================================================================
# 5. CẤU HÌNH VECTOR DATABASE (QDRANT)
# =====================================================================
QDRANT_URL         = "http://localhost:6333"
# COLLECTION_NAME    = "vietnamese_legal_rag"
COLLECTION_NAME_DENSE = "vietnamese_legal_rag_1_dense"
COLLECTION_NAME    = "vietnamese_legal_rag_1"
QDRANT_BATCH_SIZE  = 16   # Nạp 16 chunks/batch vừa khít RTX 3060 12GB
FORCE_RECREATE     = False # Đổi True nếu muốn xóa sạch collection để nạp lại từ đầu