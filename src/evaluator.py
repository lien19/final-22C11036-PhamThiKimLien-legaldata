import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unsloth import FastSentenceTransformer
import json
import torch
import gc
import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator
from transformers import BitsAndBytesConfig
from transformers.trainer_utils import get_last_checkpoint

from peft import PeftModel, PeftConfig
import config


# =====================================================================
# 1. HÀM CHUẨN BỊ DỮ LIỆU ĐÁNH GIÁ
# =====================================================================
def prepare_eval_data():
    print("-> Đang nạp kho luật (Corpus)...")
    with open(config.CORPUS_FINAL_JSON, "r", encoding="utf-8") as f:
        corpus_records = json.load(f)

    # 1. Toàn bộ kho luật
    corpus = {doc["chunk_id"]: doc["search_text"] for doc in corpus_records}

    queries = {}
    relevant_docs = {}
    query_prefix = "Instruct: Truy xuất điều khoản pháp luật trả lời câu hỏi sau\nQuery: "

    print(f"-> Đang nạp tập câu hỏi thi từ {config.EVAL_QUERIES_JSONL}...")
    with open(config.EVAL_QUERIES_JSONL, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            data = json.loads(line)
            qid = data.get("query_id", f"eval_q_{idx:04d}")
            q_text = data["query"].strip()
            pos_ids = data.get("positive_ids", [])

            valid_pos = [pid for pid in pos_ids if pid in corpus]
            if q_text and valid_pos:
                # Gắn tiền tố instruction chuẩn như lúc train
                queries[qid]       = f"{query_prefix}{q_text}"
                relevant_docs[qid] = set(valid_pos)

    print(
        f"-> Sẵn sàng chấm thi: {len(queries)} câu hỏi trên kho {len(corpus)}"
        " chunks!\n"
    )
    return corpus, queries, relevant_docs


# =====================================================================
# 2. HÀM CHẠY ĐÁNH GIÁ (LOAD 4-BIT CHỐNG NỔ VRAM)
# =====================================================================
def save_eval_scores_to_json(scores, model_name, json_path=config.EVAL_RESULTS_JSON):
  """Tự động thêm/cập nhật điểm số của model vào file JSON tổng
  (Nếu file chưa có thì tạo mới, nếu có rồi thì giữ nguyên điểm cũ và chèn thêm điểm mới).
  """
  data = {}

  # 1. Nếu file đã tồn tại -> đọc dữ liệu của các model cũ lên trước
  if os.path.exists(json_path) and os.path.getsize(json_path) > 0:
    try:
      with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    except json.JSONDecodeError:
      data = {}

  # 2. INSERT điểm của model hiện tại vào
  data[model_name] = scores

  # 3. Ghi lại toàn bộ vào file JSON
  with open(json_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

  print(f"💾 Đã lưu/chèn điểm của '{model_name}' vào file: {json_path}")


def run_evaluation_unsloth():
    # 1. Chuẩn bị dữ liệu
    corpus, queries, relevant_docs = prepare_eval_data()

    # 2. Tạo Evaluator
    # ⭐️ 1. ĐẶT BATCH_SIZE = 8 
    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="vietnamese-legal-eval",
        mrr_at_k=[1, 5, 10],
        ndcg_at_k=[5, 10],
        accuracy_at_k=[1, 3, 5, 10],
        batch_size=8,  # Để batch 8 để encode không bị nghẽn VRAM
        show_progress_bar=True,
    )

    # -------------------------------------------------------------
    # BƯỚC 1: ĐÁNH GIÁ MÔ HÌNH GỐC
    # -------------------------------------------------------------
    print("=" * 65)
    print(f"1. ĐANG ĐÁNH GIÁ MÔ HÌNH GỐC: {config.BASE_MODEL_ID}")
    print("=" * 65)

    base_model = FastSentenceTransformer.from_pretrained(
        model_name=config.BASE_MODEL_ID,
        max_seq_length=1024,
        load_in_4bit=True,
        load_in_16bit=False,
    )
    base_scores = evaluator(base_model)
    save_eval_scores_to_json(base_scores, "base_model", json_path=config.EVAL_RESULTS_JSON)

    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    # -------------------------------------------------------------
    # BƯỚC 2: ĐÁNH GIÁ MÔ HÌNH FINE-TUNE
    # -------------------------------------------------------------
    print("=" * 65)
    print(f"2. ĐANG ĐÁNH GIÁ MÔ HÌNH FINE-TUNE: {config.CHECKPOINT_DIR}")
    print("=" * 65)

    # ⭐️ TỰ ĐỘNG TÌM CHECKPOINT MỚI NHẤT
    latest_checkpoint = get_last_checkpoint(config.MODEL_DIR)

    # Nếu có checkpoint thì lấy checkpoint, nếu đã train xong 100% thì lấy thư mục gốc
    eval_model_path = latest_checkpoint if latest_checkpoint else config.MODEL_DIR

    print(f"\n=======================================================")
    print(f"-> TỰ ĐỘNG BỐC MÔ HÌNH MỚI NHẤT RA CHẤM THI: {eval_model_path}")
    print(f"=======================================================")

    tuned_model = FastSentenceTransformer.from_pretrained(
        model_name=eval_model_path,
        max_seq_length=1024,
        load_in_4bit=True,
        load_in_16bit=False,
    )
    tuned_scores = evaluator(tuned_model)
    save_eval_scores_to_json(tuned_scores, "tuned_model", json_path=config.EVAL_RESULTS_JSON)

    del tuned_model
    gc.collect()
    torch.cuda.empty_cache()

    # -------------------------------------------------------------
    # ⭐️ LƯU THẲNG RA FILE JSON
    # -------------------------------------------------------------
    # final_output = {
    #     "base_model": base_scores,
    #     "tuned_model": tuned_scores
    # }
    # with open(config.EVAL_RESULTS_JSON, "w", encoding="utf-8") as f:
    #     json.dump(tuned_scores, f, ensure_ascii=False, indent=4)
    print(f"\n💾 ĐÃ LƯU KẾT QUẢ ĐẦY ĐỦ VÀO FILE: {config.EVAL_RESULTS_JSON}")

    # -------------------------------------------------------------
    # BƯỚC 3: IN BẢNG BÁO CÁO (ĐÃ THÊM _cosine_ CHUẨN XÁC)
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"{'CHỈ SỐ ĐO LƯỜNG (METRICS)':<28} | {'MÔ HÌNH GỐC':<13} | {'SAU FINE-TUNE':<13} | {'TĂNG TRƯỞNG'}")
    print("=" * 70)

    # ⭐️ ĐÃ CÓ CHỮ _cosine_
    metrics = [
        ("Hit@1 (Chính xác Top 1)",        "vietnamese-legal-eval_cosine_accuracy@1"),
        ("Hit@5 (Nằm trong Top 5)",        "vietnamese-legal-eval_cosine_accuracy@5"),
        ("Hit@10 (Nằm trong Top 10)",      "vietnamese-legal-eval_cosine_accuracy@10"),
        ("MRR@10 (Thứ hạng trung bình)",   "vietnamese-legal-eval_cosine_mrr@10"),
        ("NDCG@10 (Độ liên quan tổng thể)","vietnamese-legal-eval_cosine_ndcg@10"),
    ]

    table_rows = []
    for label, key in metrics:
        base_val  = base_scores.get(key, 0.0) * 100
        tuned_val = tuned_scores.get(key, 0.0) * 100
        diff      = tuned_val - base_val
        sign      = "+" if diff >= 0 else ""
        print(f"{label:<28} | {base_val:>11.2f}% | {tuned_val:>11.2f}% | {sign}{diff:>6.2f}%")
        
        table_rows.append({
            "Chỉ số (Metric)": label,
            "Mô hình Gốc (Base)": f"{base_val:.2f}%",
            "Sau Fine-tune (Tuned)": f"{tuned_val:.2f}%",
            "Tăng trưởng": f"{sign} {abs(diff):.2f}%",
        })
    print("=" * 70)
    # 3. Xuất ra file JSON dạng bảng đẹp
    with open(config.EVAL_RESULTS_TABLE_JSON, "w", encoding="utf-8") as f:
        json.dump(table_rows, f, ensure_ascii=False, indent=4)

    # 4. Xuất ra file CSV mở thẳng bằng Excel
    df = pd.DataFrame(table_rows)
    df.to_csv(config.EVAL_RESULTS_TABLE_CSV, index=False, encoding="utf-8-sig")

    print(f"🎉 XONG! Đã lưu bảng so sánh đẹp mắt vào:")
    print(f"📁 1. {config.EVAL_RESULTS_TABLE_JSON}")
    print(f"📁 2. {config.EVAL_RESULTS_TABLE_CSV}")
    return base_scores, tuned_scores




# =====================================================================
# evaluation report
# =====================================================================
# INPUT_FILE = config.EVAL_RESULTS_JSON
# OUTPUT_TABLE_JSON = config.EVAL_RESULTS_TABLE_JSON
# OUTPUT_TABLE_CSV = config.EVAL_RESULTS_TABLE_CSV

# # 1. Đọc dữ liệu thô
# with open(INPUT_FILE, "r", encoding="utf-8") as f:
#     raw_data = json.load(f)

# base = raw_data.get("base_model", {})
# tuned = raw_data.get("tuned_model", {})


# # Hàm tìm giá trị theo đuôi metric (bất kể prefix dài dòng phía trước)
# def get_val(data_dict, suffix):
#     for k, v in data_dict.items():
#         if k.endswith(suffix):
#             return v
#     return 0.0


# # 2. Trích xuất đúng 5 chỉ số vàng trên bảng của bạn
# metrics_config = [
#     ("Hit@1 (Chính xác Top 1)", "_accuracy@1"),
#     ("Hit@5 (Nằm trong Top 5)", "_accuracy@5"),
#     ("Hit@10 (Nằm trong Top 10)", "_accuracy@10"),
#     ("MRR@10 (Thứ hạng trung bình)", "_mrr@10"),
#     ("NDCG@10 (Độ liên quan tổng thể)", "_ndcg@10"),
# ]

# table_rows = []
# for label, suffix in metrics_config:
#     b_val = get_val(base, suffix) * 100
#     t_val = get_val(tuned, suffix) * 100
#     diff = t_val - b_val
#     sign = "+" if diff >= 0 else "-"

#     table_rows.append({
#         "Chỉ số (Metric)": label,
#         "Mô hình Gốc (Base)": f"{b_val:.2f}%",
#         "Sau Fine-tune (Tuned)": f"{t_val:.2f}%",
#         "Tăng trưởng": f"{sign} {abs(diff):.2f}%",
#     })

# # 3. Xuất ra file JSON dạng bảng đẹp
# with open(config.EVAL_RESULTS_TABLE_JSON, "w", encoding="utf-8") as f:
#     json.dump(table_rows, f, ensure_ascii=False, indent=4)

# # 4. Xuất ra file CSV mở thẳng bằng Excel
# df = pd.DataFrame(table_rows)
# df.to_csv(config.EVAL_RESULTS_TABLE_CSV, index=False, encoding="utf-8-sig")

# print(f"🎉 XONG! Đã lưu bảng so sánh đẹp mắt vào:")
# print(f"📁 1. {config.EVAL_RESULTS_TABLE_JSON}")
# print(f"📁 2. {config.EVAL_RESULTS_TABLE_CSV}")




