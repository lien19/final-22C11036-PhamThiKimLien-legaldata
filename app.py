import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date
from flask import Flask, jsonify, render_template, request
from qdrant_client import QdrantClient, models
from unsloth import FastSentenceTransformer
import json
import os
import traceback
from transformers.trainer_utils import get_last_checkpoint

import config

app = Flask(__name__)

# =====================================================================
# CẤU HÌNH HỆ THỐNG
# =====================================================================
COLLECTION_NAME = config.COLLECTION_NAME  # "vietnamese_legal_rag_1"
TOP_K           = 10

# ⭐️ TỰ ĐỘNG TÌM CHECKPOINT MỚI NHẤT
latest_checkpoint = get_last_checkpoint(config.MODEL_DIR)

# Nếu có checkpoint thì lấy checkpoint, nếu đã train xong 100% thì lấy thư mục gốc
index_model_path = latest_checkpoint if latest_checkpoint else config.MODEL_DIR

print("-> Đang nạp mô hình Qwen 8B LoRA...")
model = FastSentenceTransformer.from_pretrained(
    model_name=index_model_path,  # "qwen8b_legal_unsloth_lora/checkpoint-700"
    max_seq_length=config.MAX_SEQ_LENGTH,
    load_in_4bit=True,
    load_in_16bit=False,
)

print("-> Đang kết nối Qdrant...")
qdrant = QdrantClient(url=config.QDRANT_URL)  # "http://localhost:6333"

# Reranker dự phòng
reranker = None
try:
  from sentence_transformers import CrossEncoder

  print("-> Đang nạp Reranker (CPU)...")
  reranker = CrossEncoder("BAAI/bge-reranker-base", device="cpu")
  print("   (Reranker sẵn sàng)")
except Exception as e:
  print(f"⚠️ Chưa cài sentence_transformers CrossEncoder ({e}) -> Dùng RRF bù.")



# NẠP BẢN ĐỒ MỞ RỘNG GIA PHẢ (KÈM CẢNH BÁO ĐƯỜNG DẪN)
expanded_content_map = {}
if os.path.exists(config.CORPUS_MAP_FINAL_JSON):  # "3-final_legal_expanded_content_map.json"
  with open(config.CORPUS_MAP_FINAL_JSON, "r", encoding="utf-8") as f:  
    expanded_content_map = json.load(f)
  print( f"-> Đã nạp bản đồ phân cấp mở rộng ({len(expanded_content_map)} chunks)"  )
else:
  # In cảnh báo chữ to rõ ràng trên terminal nếu gõ sai đường dẫn:
  print("\n" + "!" * 65)
  print(f"⚠️ CẢNH BÁO: KHÔNG TÌM THẤY FILE: '{config.CORPUS_MAP_FINAL_JSON}'!")
  print("👉 Hệ thống sẽ tự động dùng 'content' gốc trong Qdrant thay thế.")
  print("👉 Hãy kiểm tra lại xem tên file trong folder data/ có phải là"
      " '3-final_legal_expanded_content_map.json' không nhé!"
  )
  print("!" * 65 + "\n")
  

@app.route("/")
def index():
  return render_template("index.html")


@app.route("/api/questions", methods=["GET"])
def get_questions():
  questions = []
  if os.path.exists(config.EVAL_QUERIES_JSONL):
    with open(config.EVAL_QUERIES_JSONL, "r", encoding="utf-8") as f:
      for line in f:
        if line.strip():
          data = json.loads(line)
          # ⭐️ LẤY NGUYÊN VẸN TOÀN BỘ QUERY (BAO GỒM CẢ CONTEXT + CÂU HỎI)
          q = data.get("query", "").strip()
          if q and q not in questions:
            questions.append(q)

  if not questions:
    questions = [
        "Thu nhập chịu thuế là gì?",
        "Chi phí mua hàng hóa 5 triệu đồng bằng tiền mặt có được tính vào chi"
        " phí được trừ thuế TNDN không?",
        "Doanh nghiệp công nghệ cao có được ưu đãi thuế không?",
        "Khoản chi nào không được trừ khi xác định thu nhập chịu thuế TNDN?",
        "Doanh nghiệp thành lập mới được hưởng thuế suất ưu đãi bao nhiêu %?",
    ]
  return jsonify(questions[:50])


# ⭐️ ĐÃ SỬA CHUẨN TÊN THAM SỐ: score
def format_point_xóa(hit, score=None):
  p        = hit.payload or {}
  chunk_id = p.get("chunk_id", "")
  law      = p.get("law_no", "")
  art      = p.get("article_id", "")
  badge    = f"Điều {art} · {law}" if art else law

  # ⭐️ TỰ ĐỘNG TRA CỨU NỘI DUNG MỞ RỘNG (O(1) trong RAM):
  # - Nếu là Điều: tự động lấy toàn bộ Khoản + Điểm
  # - Nếu là Khoản: tự động lấy Khoản + Điểm con của nó
  # - Nếu là Điểm: lấy nguyên Điểm đó
  display_content = expanded_content_map.get(chunk_id, p.get("content", ""))

  return {
      "badge": badge,
      "tax_type": str(p.get("tax_type", "")).upper(),
      "hierarchy": p.get("hierarchy_context", ""),
      "content": display_content,  # 👈 Nội dung đã được mở rộng phân cấp chuẩn!
      "score": (
          round(float(score), 4)
          if score is not None
          else round(float(hit.score), 4)
      ),
  }


# ⭐️ SỬA HÀM format_point: THÊM LOGIC TRỪ ĐIỂM THEO NGÀY THUẦN TÚY
def format_point(hit, score=None, target_date=None):
  p = hit.payload or {}
  chunk_id = p.get("chunk_id", "")
  law = p.get("law_no", "")
  art = p.get("article_id", "")
  badge = f"Điều {art} · {law}" if art else law

  raw_score = float(score if score is not None else hit.score)

  # 1. Xác định target_date (mặc định lấy hôm nay nếu không truyền)
  if not target_date:
    target_date = date.today().strftime("%Y-%m-%d")

  # 2. Lấy ngày hiệu lực và ngày hết hạn (làm sạch chuỗi rỗng/None)
  def clean_d(k):
    v = str(p.get(k, "")).strip()
    return None if v.lower() in ["", "none", "nan", "null"] else v

  eff_date = clean_d("effective_date")
  exp_date = clean_d("expire_date")

  # -------------------------------------------------------------
  # ⭐️ SO SÁNH TOÁN HỌC NGÀY THÁNG (BỎ HOÀN TOÀN TRƯỜNG STATUS CỐ ĐỊNH)
  # -------------------------------------------------------------
  is_expired = False
  status_label = "còn hiệu lực"

  if eff_date and eff_date > target_date:
    # Chưa tới ngày áp dụng tại mốc target_date
    is_expired = True
    status_label = "chưa có hiệu lực"
  elif exp_date and exp_date <= target_date:
    # Đã quá ngày hết hạn tại mốc target_date
    is_expired = True
    status_label = "hết hiệu lực"

  # ⭐️ CÔNG THỨC PHẠT ĐIỂM: Giảm 25% nếu không còn hiệu lực tại target_date
  if is_expired:
    adjusted_score = raw_score * 0.75
    display_status = f"{status_label} (đã trừ 25% điểm)"
  else:
    adjusted_score = raw_score
    display_status = "còn hiệu lực"

  display_content = expanded_content_map.get(chunk_id, p.get("content", ""))

  return {
      "badge": badge,
      "tax_type": str(p.get("tax_type", "TNDN")).upper(),
      "hierarchy": p.get("hierarchy_context", ""),
      "content": display_content,
      "score": round(adjusted_score, 4),  # 👈 Điểm đã bị trừ phạt
      "raw_score": round(raw_score, 4),  # 👈 Điểm gốc ban đầu
      "status": display_status,
      "is_expired": is_expired,
      "effective_date": eff_date or "Chưa rõ",
      "expire_date": exp_date or "Vô thời hạn",
  }

# ⭐️ BỔ SUNG: Nạp model BM25 của FastEmbed để phục vụ Cột 3 & Cột 4
try:
  from fastembed import SparseTextEmbedding

  print("-> Đang nạp mô hình BM25 (CPU)...")
  bm25_model = SparseTextEmbedding("Qdrant/bm25")
  print("   (BM25 sẵn sàng)")
except Exception as e:
  print(f"⚠️ Chưa cài fastembed ({e}) -> Cần pip install fastembed")
  bm25_model = None


# @app.route("/api/search", methods=["POST"])
def search_xóa():
  try:
    data  = request.json or {}
    query = data.get("query", "").strip()
    if not query:
      return jsonify({"error": "Vui lòng nhập hoặc chọn câu hỏi"}), 400

    prompt_query = f"Instruct: Truy xuất điều khoản pháp luật trả lời câu hỏi sau\nQuery: {query}"

    # 1. CỘT 2: TUNED MODEL-
    tuned_vec  = model.encode([prompt_query], normalize_embeddings=True)[0].tolist()
    tuned_hits = qdrant.query_points( collection_name=COLLECTION_NAME, query=tuned_vec, limit=TOP_K ).points
    tuned_results = [format_point(h) for h in tuned_hits]

    # 2. CỘT 1: BASE MODEL (TẮT ADAPTER AN TOÀN)
    base_results = []
    base_error = None

    try:
        # Chỉ khi nào tắt adapter thành công 100% mới chạy
        if hasattr(model[0].auto_model, "disable_adapter"):
            with model[0].auto_model.disable_adapter():
                base_vec = model.encode([prompt_query], normalize_embeddings=True)[0].tolist()
            
            base_hits    = qdrant.query_points( collection_name=COLLECTION_NAME, query=base_vec, limit=TOP_K ).points
            base_results = [format_point(h) for h in base_hits]
        else:
            base_error = "Mô hình hiện tại không hỗ trợ hàm disable_adapter() để tháo lớp LoRA."
    except Exception as e:
        base_error = f"Không thể tắt LoRA Adapter để nạp Bản Gốc trên cùng phiên GPU ({str(e)})."

    # 3. CỘT 3: RERANK
    if reranker:
      candidate_pairs   = [ [query, h.payload.get("search_text", "")] for h in tuned_hits ]
      rerank_scores     = reranker.predict(candidate_pairs)
      scored_candidates = sorted( zip(tuned_hits, rerank_scores), key=lambda x: x[1], reverse=True  )
      rerank_results    = [ format_point(h, score=s) for h, s in scored_candidates[:TOP_K] ]
    else:
      # Thuật toán giả lập dựa trên từ khóa nếu chưa có Reranker
      rerank_results = sorted(
          tuned_results,
          key=lambda x: ( 1 if any(w in x["content"].lower() for w in query.lower().split())
                          else 0,
                          x["score"],
          ),
          reverse=True,
      )

    return jsonify({
        "base": base_results,
        "base_error": base_error,   # 👈 Gửi lý do lỗi nếu không tắt được LoRA
        "tuned": tuned_results,
        "rerank": rerank_results,
    })

  except Exception as e:
    traceback.print_exc()  # In lỗi chi tiết ra Terminal
    return jsonify({"error": f"Lỗi phía Server: {str(e)}"}), 500

@app.route("/api/search", methods=["POST"])
def search():
  try:
    data = request.json or {}
    query = data.get("query", "").strip()

    # ⭐️ Lấy target_date người dùng truyền lên (hoặc để None tự lấy hôm nay)
    target_date = data.get("target_date")

    if not query:
      return jsonify({"error": "Vui lòng nhập hoặc chọn câu hỏi"}), 400

    prompt_query = f"Instruct: Truy xuất điều khoản pháp luật trả lời câu hỏi sau\nQuery: {query}"

    # Sinh vector cho câu hỏi (Dense & Sparse BM25)
    tuned_vec = model.encode([prompt_query], normalize_embeddings=True)[0].tolist()
    query_sparse = ( list(bm25_model.embed([query]))[0] if bm25_model else None  )


    # ⭐️ HÀM TIỆN ÍCH: Format và tự động Re-sort tụt hạng tài liệu bị trừ điểm
    def process_and_rank(hits, score_list=None):
      if score_list is not None:  # ⭐️ Thêm "is not None" để không bị lỗi NumPy Array
        formatted = [ format_point(h, score=s, target_date=target_date) for h, s in zip(hits, score_list) ]
      else:
        formatted = [ format_point(h, target_date=target_date) for h in hits ]
      # Sắp xếp lại theo điểm mới sau khi trừ phạt:
      formatted.sort(key=lambda x: x["score"], reverse=True)
      return formatted[:TOP_K]
    
    # =============================================================
    # CỘT 1: BASE MODEL (TẮT LORA ADAPTER)
    # =============================================================
    col1_base = []
    base_error = None
    try:
      # Chỉ khi nào tắt adapter thành công 100% mới chạy
      if hasattr(model[0].auto_model, "disable_adapter"):
        with model[0].auto_model.disable_adapter():
          base_vec = model.encode([prompt_query], normalize_embeddings=True)[0].tolist()

        base_hits = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=base_vec,
            using="dense",  # 👈 Chỉ định ngăn dense
            limit=TOP_K,
        ).points
        # base_results = [format_point(h) for h in base_hits]
        col1_base = process_and_rank(base_hits)
      else:
        base_error = "Mô hình không hỗ trợ disable_adapter()."
    except Exception as e:
      base_error = f"Lỗi tắt LoRA: {str(e)}"

    # =============================================================
    # CỘT 2: FINE-TUNED (DENSE THUẦN - TẮT BM25)
    # =============================================================
    tuned_dense_hits = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=tuned_vec,
        using="dense",  # 👈 Chỉ tìm trong ngăn dense, bỏ qua BM25
        limit=TOP_K,
    ).points
    # col2_tuned_dense = [format_point(h) for h in tuned_dense_hits]
    col2_tuned_dense = process_and_rank(tuned_dense_hits)
    # =============================================================
    # Chuẩn bị truy vấn prefetch cho Hybrid (dùng chung cho Cột 3 và 4)
    prefetch_hybrid = []
    if query_sparse:
      prefetch_hybrid = [
          models.Prefetch( query=tuned_vec, using="dense", limit=20),  # Top 20 Dense
          models.Prefetch( 
              query=models.SparseVector( indices=query_sparse.indices, values=query_sparse.values ),
              using="bm25",
              limit=20,  # Top 20 BM25
          ),
      ]

    # =============================================================
    # CỘT 3: HYBRID VỚI THUẬT TOÁN RRF (Thứ hạng)
    # =============================================================
    if prefetch_hybrid:
      rrf_hits = qdrant.query_points(
          collection_name=COLLECTION_NAME,
          prefetch=prefetch_hybrid,
          query=models.FusionQuery(fusion=models.Fusion.RRF),  # 👈 Thuật toán RRF
          limit=TOP_K,
      ).points
      # col3_hybrid_rrf = [format_point(h) for h in rrf_hits]
      col3_hybrid_rrf = process_and_rank(rrf_hits)
    else:
      col3_hybrid_rrf = col2_tuned_dense
      print(f"Mô hình HYBRID VỚI THUẬT TOÁN RRF không hoạt động")

    # =============================================================
    # CỘT 4: HYBRID VỚI THUẬT TOÁN DBSF (Phân phối điểm số)
    # =============================================================
    if prefetch_hybrid:
      dbsf_hits = qdrant.query_points(
          collection_name=COLLECTION_NAME,
          prefetch=prefetch_hybrid,
          query=models.FusionQuery( fusion=models.Fusion.DBSF ),  # 👈 Thuật toán DBSF
          limit=TOP_K,
      ).points
      # col4_hybrid_dbsf = [format_point(h) for h in dbsf_hits]
      col4_hybrid_dbsf = process_and_rank(dbsf_hits)
    else:
      col4_hybrid_dbsf = col2_tuned_dense
      print(f"Mô hình HYBRID VỚI THUẬT TOÁN DBSF không hoạt động")

    # =============================================================
    # CỘT 5: RERANKER (Soi lại Top ứng viên từ Hybrid RRF)
    # =============================================================
    # Lấy nguồn ứng viên chất lượng nhất từ Hybrid RRF để Rerank
    source_hits_for_rerank = ( rrf_hits if prefetch_hybrid else tuned_dense_hits )

    if reranker:
      candidate_pairs = [ [query, h.payload.get("search_text", "")]
                          for h in source_hits_for_rerank  ]
      rerank_scores = reranker.predict(candidate_pairs)
      # scored_candidates = sorted( zip(source_hits_for_rerank, rerank_scores),key=lambda x: x[1],reverse=True, )
      # col5_rerank = [ format_point(h, score=s) for h, s in scored_candidates[:TOP_K] ]
      col5_rerank = process_and_rank(source_hits_for_rerank, score_list=rerank_scores)
    else:
      col5_rerank = col3_hybrid_rrf
      print(f"RERANKER (Soi lại Top ứng viên từ Hybrid RRF) không hoạt động")


    # Trả về đủ 5 cột cho giao diện UI
    return jsonify({
        "base": col1_base,
        "base_error": base_error,
        "tuned_dense": col2_tuned_dense,
        "hybrid_rrf": col3_hybrid_rrf,
        "hybrid_dbsf": col4_hybrid_dbsf,
        "rerank": col5_rerank,
    })

  except Exception as e:
    traceback.print_exc()
    return jsonify({"error": f"Lỗi phía Server: {str(e)}"}), 500


if __name__ == "__main__":
  print("\n🚀 Web Tra Cứu đang chạy tại: http://localhost:5000")
  app.run(host="0.0.0.0", port=5000, debug=False)