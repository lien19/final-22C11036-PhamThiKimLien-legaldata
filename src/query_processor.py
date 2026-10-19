import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, datetime
import json
import re
import pandas as pd
import config


# ===============================================================
# 1.
# ===============================================================

def process_query_dataset_to_json(
    input_file_path,  # File .xlsx hoặc .csv của bạn
    output_jsonl_path,  # Đường dẫn xuất file .jsonl (cho Mining)
    output_json_path=None,  # Đường dẫn xuất file .json (đọc bằng mắt)
    # ⭐️ TOÀN BỘ CỘT ĐỀU ĐƯỢC THAM SỐ HÓA (KHÔNG HARDCODE)
    law_no_col="no",  # Tên cột văn bản
    hierarchy_cols=["Điều", "Khoản", "Điểm"],  # Danh sách cột phân cấp đích
    context_col="Context",  # Tên cột tình huống
    question_col="Question",  # Tên cột câu hỏi
    metadata_cols=None,  # Các cột phụ muốn giữ lại (Tax Type, Difficulty...)
):
  """Chuyển đổi bảng câu hỏi sang JSON/JSONL:
  - Ghép Context + Question thành Query hoàn chỉnh.
  - Gom các dòng trùng Query lại để gom mảng positive_ids (Multi-positive).
  - Không hardcode tên cột.

  return
  {
    "query_id": "q_001",
    "query": "Công ty A thu mua xoài tươi...\nHoạt động sản xuất xoài sấy dẻo...",
    "positive_ids": [
        "320/2025/NĐ-CP_4_1_b",
        "320/2025/NĐ-CP_5_2"
    ],
    "Tax Type": "CIT",
    "Difficulty": "Medium",
    "Intent": "Lookup"
    }
  """
  # 1. Đọc file (hỗ trợ cả Excel lẫn CSV)
  if input_file_path.endswith((".xlsx", ".xls")):
    df = pd.read_excel(input_file_path, keep_default_na=False)
  else:
    df = pd.read_csv(input_file_path, keep_default_na=False, low_memory=False)

  # Chuẩn hóa khoảng trắng tên cột
  df.columns = [str(c).strip() for c in df.columns]

  # Chuẩn hóa ND-CP thành NĐ-CP cho khớp với Corpus
  if law_no_col in df.columns:
    df[law_no_col] = (
        df[law_no_col]
        .astype(str)
        .str.replace(r"ND-CP", "NĐ-CP", case=False, regex=True)
    )

  # 2. Tạo mã positive_id cho từng dòng theo đúng công thức chunk_id của Corpus
  id_cols = [law_no_col] + hierarchy_cols
  pos_ids_per_row = []

  for vals in zip(*(df[c].astype(str).str.strip() for c in id_cols)):
    # Bỏ qua ô rỗng, ghép bằng dấu gạch dưới
    pid = "_".join(
        v
        for v in vals
        if v and v.lower() not in ["", "nan", "none", "-", "null"]
    ).replace(" ", "_")
    pos_ids_per_row.append(pid)

  df["_temp_pos_id"] = pos_ids_per_row

  # 3. Tạo câu query hoàn chỉnh = Context + Question
  full_queries = []
  for ctx, q in zip(df[context_col].astype(str).str.strip(),
                    df[question_col].astype(str).str.strip(),
                ):
    if ctx and q and ctx != q:
        full_query = re.sub(r"\s+", " ", f"{ctx} {q}").strip()
    else:
        full_query = re.sub(r"\s+", " ", q or ctx).strip()
    full_queries.append(full_query)

  df["_temp_full_query"] = full_queries

  # 4. Xác định các cột metadata phụ trợ cần giữ lại
  system_cols = id_cols + [
      context_col,
      question_col,
      "_temp_pos_id",
      "_temp_full_query",
  ]
  if metadata_cols is None:
    # Tự động lấy tất cả các cột còn lại trong bảng (STT, Tax Type, Intent, Difficulty...)
    extra_cols = [c for c in df.columns if c not in system_cols]
  else:
    extra_cols = [c for c in metadata_cols if c in df.columns]

  # -----------------------------------------------------------------
  # 5. ⭐️ GOM NHÓM THEO CÂU HỎI (Xử lý các dòng trùng Query nhưng khác Điều/Khoản)
  # -----------------------------------------------------------------
  queries_data = []
  query_idx = 1

  for query_text, grp in df.groupby("_temp_full_query", sort=False):
    if not query_text.strip():
      continue

    first_row = grp.iloc[0]

    # Gom toàn bộ positive_id của các dòng trùng câu hỏi này (khử trùng lặp)
    raw_pos = [pid for pid in grp["_temp_pos_id"] if pid]
    pos_list = list(dict.fromkeys(raw_pos))

    # Lấy metadata từ dòng đầu tiên
    extra_meta = {col: first_row[col] for col in extra_cols if str(first_row[col]).strip() }

    item = {
        "query_id": f"q_{query_idx:03d}",
        "query": query_text,
        "positive_ids": pos_list,  # ⭐️ Mảng chứa tất cả các đáp án đúng của câu hỏi này
        **extra_meta,
    }
    queries_data.append(item)
    query_idx += 1

  # 6. Xuất file .jsonl (Chuẩn nạp cho khâu Mining)
  with open(output_jsonl_path, "w", encoding="utf-8") as f_jsonl:
    for item in queries_data:
      f_jsonl.write(json.dumps(item, ensure_ascii=False) + "\n")

  # 7. Xuất file .json (Dễ đọc bằng mắt nếu cần)
  if output_json_path:
    with open(output_json_path, "w", encoding="utf-8") as f_json:
      json.dump(queries_data, f_json, ensure_ascii=False, indent=4)

  print(f"--- XỬ LÝ QUERY THÀNH CÔNG ---")
  print(f"Tổng số dòng gốc ban đầu: {len(df)}")
  print(f"Tổng số Query sau khi gom nhóm khử trùng: {len(queries_data)}")
  print(f"File JSONL lưu tại: {output_jsonl_path}")

  return queries_data


# process_query_dataset_to_json(
#     input_file_path="3_query_current_law.csv",  # File Excel của bạn
#     output_jsonl_path="3_queries_for_mining.jsonl",
#     output_json_path="3_queries_for_mining.json",
#     # ⭐️ Truyền chính xác tên các cột trong file của bạn:
#     law_no_col="no",
#     hierarchy_cols=["Điều", "Khoản", "Điểm"],
#     context_col="Context",
#     question_col="Question",
#     # Tự động gom luôn các cột phụ:
#     metadata_cols=["Tax Type", "Intent", "Legal Issue", "Difficulty"],
# )




# ===============================================================
# 2.
# ===============================================================

# Hàm làm sạch chuỗi, khử lỗi float .0 của Pandas
def clean_str(v):
  s = str(v).strip()
  if s.endswith(".0"):
    s = s[:-2]
  return s


# Hàm chuẩn hóa ngày sang chuẩn YYYY-MM-DD
def clean_date_iso(val):
  s = str(val).strip()
  if not s or s.lower() in ["nan", "none", "-", "null"]:
    return ""
  try:
    return pd.to_datetime(s, format="%d/%m/%Y").strftime("%Y-%m-%d")
  except Exception:
    try:
      return pd.to_datetime(s).strftime("%Y-%m-%d")
    except Exception:
      return s


def check_and_enrich_queries(
    query_input_path,  # File câu hỏi (.csv hoặc .xlsx)
    corpus_json_path,  # File kho luật (.json)
    output_csv_path,  # File CSV kết quả
    output_jsonl_path,  # (Tùy chọn) Xuất luôn file JSONL cho Mining ES
    # ⭐️ 2 CHẾ ĐỘ GOM NHÓM: 'query' HOẶC 'q_no'
    group_by,
    target_date_col,  # Tên cột ngày trong file query
    law_no_col,
    hierarchy_cols,    #["Điều", "Khoản", "Điểm"]
    question_col,
    q_no_col=None,  # Tên cột mã câu hỏi khi chọn group_by='q_no' (fallback STT)
    context_col=None,
):
  print(f"--- BẮT ĐẦU XỬ LÝ QUERY (Chế độ gom: {group_by}) ---")

  # =================================================================
  # 1. ĐỌC KHO CORPUS JSON VÀO RAM
  # =================================================================
  with open(corpus_json_path, "r", encoding="utf-8") as f:
    corpus_data = json.load(f)

  records = ( corpus_data if isinstance(corpus_data, list) else corpus_data.values() )
  corpus_lookup = {
      str(doc.get("chunk_id", "")).strip(): doc
      for doc in records
      if doc.get("chunk_id")
  }

  print(f"-> Đã nạp {len(corpus_lookup)} chunks từ kho Corpus vào bộ nhớ.")

  # =================================================================
  # 2. ĐỌC FILE QUERY (CSV HOẶC EXCEL)
  # =================================================================
  if query_input_path.endswith((".xlsx", ".xls")):
    df_query = pd.read_excel(query_input_path, keep_default_na=False)
  else: 
    df_query = pd.read_csv( query_input_path, keep_default_na=False, low_memory=False)

  df_query.columns = [str(c).strip() for c in df_query.columns]

  # 2.1 Chuẩn hóa target_date sang YYYY-MM-DD
  if target_date_col in df_query.columns:
    df_query[target_date_col] = df_query[target_date_col].apply(clean_date_iso)

  # 2.2 Tạo mã positive_id tạm thời cho từng dòng
  id_cols = [law_no_col] + hierarchy_cols
  temp_pos_ids = []
  for vals in zip(*(df_query[c] for c in id_cols)):
    clean_vals = [clean_str(v) for v in vals]
    pid = "_".join(
        v
        for v in clean_vals
        if v and v.lower() not in ["", "nan", "none", "-", "null"]
    ).replace(" ", "_")
    temp_pos_ids.append(pid)
  df_query["_temp_pos_id"] = temp_pos_ids

    
  # 2.3 Tạo câu query: Tự động thích ứng có Context hoặc KHÔNG có Context
  # =================================================================
  has_context = context_col in df_query.columns

  if has_context:
    full_queries = []
    for ctx, q in zip(df_query[context_col], df_query[question_col]):
      ctx_s = str(ctx).strip()
      q_s = str(q).strip()
      # Nếu có cả tình huống lẫn câu hỏi thì ghép lại
      if ctx_s and q_s and ctx_s != q_s:
        full_queries.append(f"{ctx_s}\n{q_s}")
      else:
        full_queries.append(q_s if q_s else ctx_s)
    df_query["_temp_full_query"] = full_queries
  else:
    # ⭐️ Bảng này KHÔNG CÓ cột Context -> Lấy thẳng 100% cột Question:
    df_query["_temp_full_query"] = [
        str(q).strip() for q in df_query[question_col]
    ]

  # =================================================================
  # 3. THIẾT LẬP CHẾ ĐỘ GOM NHÓM
  # =================================================================
  if group_by == "q_no":
    # Tìm cột mã câu hỏi thực tế trong file
    actual_q_col = (
        q_no_col
        if q_no_col in df_query.columns
        else ("STT" if "STT" in df_query.columns else df_query.columns[0])
    )
    print(f"-> Gom nhóm theo cột mã: '{actual_q_col}'")
    group_iterator = df_query.groupby(actual_q_col, sort=False)
  else:
    # Mặc định: Gom theo câu hỏi
    print("-> Gom nhóm theo nội dung câu hỏi (_temp_full_query)")
    group_iterator = df_query.groupby("_temp_full_query", sort=False)

  # =================================================================
  # 4. DUYỆT TỪNG NHÓM, TRÍCH XUẤT CORPUS VÀ GOM POSITIVE_IDS
  # =================================================================
  output_rows = []
  jsonl_records = []
  query_idx = 1

  for grp_key, grp in group_iterator:
    first_row  = grp.iloc[0].to_dict()
    full_query = first_row["_temp_full_query"]

    if not full_query.strip():
      continue

    # Gom toàn bộ positive_ids của nhóm (khử trùng lặp)
    raw_pos = [pid for pid in grp["_temp_pos_id"] if pid]
    pos_ids = list(dict.fromkeys(raw_pos))

    # Lấy thông tin ngày mục tiêu
    t_date = str(first_row.get(target_date_col, "")).strip()

    # Tra cứu các mã trong kho Corpus # ⭐️ Phải có ít nhất 1 văn bản đúng là TNDN (hoặc CIT) thì mới cho True:
    matched_docs = [ corpus_lookup[pid] for pid in pos_ids if pid in corpus_lookup ]
    is_in_corpus = any( str(d.get("tax_type") or d.get("Tax Type") or "").strip().upper()
                        in ["TNDN", "CIT"]
                        for d in matched_docs
    )

    # Bốc hierarchy_context và content từ kho luật
    contexts = [d.get("hierarchy_context", "")
                for d in matched_docs
                if d.get("hierarchy_context")
    ]
    contents = [d.get("content", "") for d in matched_docs if d.get("content")]

    # 4.1 Đóng gói dòng CSV (GIỮ NGUYÊN 100% CÁC CỘT GỐC BAN ĐẦU)
    row_out = {**first_row}
    row_out["positive_ids"] = "; ".join(pos_ids)
    row_out["is_in_corpus"] = is_in_corpus
    row_out["corpus_hierarchy_context"] = "\n---\n".join(contexts)
    row_out["corpus_content"] = "\n\n=====\n\n".join(contents)

    # Xóa 2 cột tạm nội bộ
    row_out.pop("_temp_pos_id", None)
    row_out.pop("_temp_full_query", None)
    output_rows.append(row_out)

    # 4.2 Đóng gói bản ghi JSONL (nếu cần xuất cho khâu Mining ES)
    qid = (
        f"q_{grp_key}"
        if group_by == "q_no"
        else f"q_{first_row.get('STT', query_idx)}"
    )
    jsonl_records.append({
        "query_id": str(qid),
        "query": full_query,
        "positive_ids": pos_ids,
        "target_date": t_date,  # ⭐️ Đã đưa vào metadata
        "target_status": "còn hiệu lực" if not t_date else None,
    })
    query_idx += 1

  # =================================================================
  # 5. XUẤT FILE KẾT QUẢ
  # =================================================================
  df_out = pd.DataFrame(output_rows)
  df_out.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
  print(f"📁 1. Đã lưu file CSV đầy đủ cột tại: {output_csv_path}")

  if output_jsonl_path:
    with open(output_jsonl_path, "w", encoding="utf-8") as f_jl:
      for r in jsonl_records:
        f_jl.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"📁 2. Đã lưu file JSONL sẵn sàng cho Mining: {output_jsonl_path}")

  print(f"-> Tổng số dòng câu hỏi sau khi gom: {len(df_out)}")
  print(f"✅ Số câu khớp trọn vẹn trong Corpus: {sum(df_out['is_in_corpus'])}")

  return df_out


# check_and_enrich_queries(
#     query_input_path="3_query_current_law.csv",
#     corpus_json_path="3-final_legal_corpus.json",
#     output_csv_path="3-query_enriched_with_corpus_current_law.csv",
#     output_jsonl_path="3-queries_mining_current_law.jsonl",  # Xuất luôn JSONL để dùng
#     group_by="query",  # ⭐️ Gom theo câu hỏi
    
#     target_date_col="target_date",  # Cột ngày
#     law_no_col="no",
#     hierarchy_cols=["Điều", "Khoản", "Điểm"],
#     context_col="Context",
#     question_col="Question",
# )                   #Gom nhóm theo Câu hỏi (group_by="query")

# check_and_enrich_queries(
#     query_input_path="3_query_old_law.csv",
#     corpus_json_path="3-final_legal_corpus.json",
#     output_csv_path="3-query_enriched_with_corpus_old_law.csv",
#     output_jsonl_path="3-queries_mining__old_law.jsonl",
#     group_by="q_no",  # ⭐️ Gom theo mã câu hỏi
    
#     q_no_col="q_no",  # (hoặc q_no_col="STT")
#     target_date_col="target_date",
#     law_no_col="number",
#     hierarchy_cols=["article", "khoan", "diem"],    
#     question_col="Question",
# )                    #Gom nhóm theo Mã câu hỏi (group_by="q_no")




# ===============================================================
# 3.
# ===============================================================
def parse_or_default_date_xóa(date_val):
    """
    Xử lý target_date:
    - Nếu để trống / NaN / '-' -> Tự động lấy ngày hôm nay (YYYY-MM-DD).
    - Nếu có giá trị DD/MM/YYYY -> Chuyển thành YYYY-MM-DD.
    """
    s = str(date_val).strip()

    # 1. Nếu để trống -> Mặc định là ngày hôm nay
    if not s or s.lower() in ["nan", "none", "-", "null", ""]:
        return date.today().strftime("%Y-%m-%d")

    # 2. Xóa các ký tự đuôi float vô tình bị dính (như .0)
    if s.endswith(".0"):
        s = s[:-2]

    # 3. Ép chuẩn format DD/MM/YYYY sang YYYY-MM-DD
    try:
        return datetime.strptime(s, "%d/%m/%Y").strftime("%Y-%m-%d")
    except ValueError:
        # Dự phòng trường hợp đã lỡ điền sẵn dạng YYYY-MM-DD hoặc định dạng ngày khác
        try:
            return pd.to_datetime(s).strftime("%Y-%m-%d")
        except Exception:
            return s  # Nếu là chuỗi lạ thì giữ nguyên


def export_and_split_queries_for_mining_xóa(
    input_csv_path,
    output_csv_with_id,
    output_train_jsonl="data/train_queries_for_mining.jsonl",
    output_eval_jsonl="data/eval_queries.jsonl",
    eval_ratio=0.2,  # 80% Train, 20% Eval
    random_seed=42,  # Cố định seed chia ngẫu nhiên
    result_col="Kiểm tra trích dẫn (ok / gợi ý)",
    tax_neg_col="tax_type_for_negative",
    target_date_col="target_date",
    context_col="Context",
    question_col="Question",
    positive_id_col="positive_ids",
):
    print("--- BẮT ĐẦU CHUYỂN ĐỔI VÀ CHIA TẬP TRAIN/EVAL CHO MINING ---")

    # 1. Đọc file CSV
    df = pd.read_csv(input_csv_path, keep_default_na=False, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    print(f"-> Tổng số dòng trong file CSV gốc: {len(df)}")

    # -------------------------------------------------------------
    # BƯỚC 1: LỌC DÒNG ĐẠT CHUẨN
    # -------------------------------------------------------------
    mask_corpus = df["is_in_corpus"].astype(str).str.upper() == "TRUE"
    mask_ok = (df[result_col].astype(str).str.strip().str.lower().str.startswith("ok") )

    df_ok = df[mask_corpus & mask_ok].copy()
    print(f"-> Số dòng đạt chuẩn 'ok' và có trong Corpus: {len(df_ok)} dòng.")

    if df_ok.empty:
        print("❌ Không có dòng nào đạt chuẩn 'ok'! Kiểm tra lại file.")
        return

    # -------------------------------------------------------------
    # BƯỚC 2: TẠO QUERY HOÀN CHỈNH VÀ GOM NHÓM
    # -------------------------------------------------------------
    has_ctx = context_col in df_ok.columns
    contexts = (
        df_ok[context_col].astype(str).str.strip()
        if has_ctx
        else [""] * len(df_ok)
    )
    questions = df_ok[question_col].astype(str).str.strip()

    # Ghép Context + Question
    full_queries = []
    for ctx, q in zip(contexts, questions):
        if ctx and q and ctx != q:
            full_queries.append(f"{ctx}\n{q}")
        else:
            full_queries.append(q or ctx)
    df_ok["_full_query"] = full_queries

    # Cột chứa mã positive (fallback sang check_positive_id nếu tên cột khác)
    actual_pos_col = (
        positive_id_col
        if positive_id_col in df_ok.columns
        else "check_positive_id"
    )

    grouped_queries = []

    # Gom nhóm theo câu hỏi duy nhất
    for query_text, grp in df_ok.groupby("_full_query", sort=False):
        if not query_text.strip():
            continue

        first_row = grp.iloc[0]

        # 2.1 Gom toàn bộ positive_ids của câu hỏi (1 câu có thể trích nhiều Điều/Khoản)
        pos_list = []
        for p_val in grp[actual_pos_col]:
            for p in str(p_val).split(";"):
                cleaned = p.strip()
                if cleaned and cleaned not in pos_list:
                    pos_list.append(cleaned)

        if not pos_list:
            continue

        # 2.2 Xử lý target_date: Điền tay thì format lại Y-m-d, để trống thì lấy hôm nay
        raw_date = first_row.get(target_date_col, "")
        t_date = parse_or_default_date(raw_date)

        # 2.3 Lấy danh sách sắc thuế dùng để đào negative (loại người nhà)
        tax_for_neg = str(first_row.get(tax_neg_col, "")).strip()

        grouped_queries.append({
            "query": query_text,
            "positive_ids": pos_list,
            "tax_type_for_negative": tax_for_neg,
            "target_date": t_date,
        })

    df_final = pd.DataFrame(grouped_queries)
    print(f"-> Tổng số câu hỏi duy nhất sau khi gom: {len(df_final)}")

    # -------------------------------------------------------------
    # BƯỚC 3: CHIA TẬP TRAIN (80%) VÀ EVAL (20%)
    # -------------------------------------------------------------
    eval_df = df_final.sample(frac=eval_ratio, random_state=random_seed)
    train_df = df_final.drop(eval_df.index)


    # =============================================================
    # ⭐️ BỔ SUNG: XUẤT BẢNG INPUT ĐÃ GẮN MÃ 5 CHỮ SỐ VÀ TẬP TRAIN/EVAL
    # =============================================================
    # 1. Tạo từ điển tra mã query_id (5 chữ số) từ danh sách đã gom
    query_to_id = {item["query"]: f"q_{idx+1:05d}" for idx, item in enumerate(grouped_queries)}
    train_query_set = set(train_df["query"])

    # 2. Gán mã query_id và phân loại train/eval ngược lại vào bảng input
    df_ok["query_id"] = df_ok["_full_query"].map(query_to_id)
    df_ok["split"] = df_ok["_full_query"].apply(lambda q: "train" if q in train_query_set else "eval")

    # 3. Đưa 2 cột này lên đầu bảng, bỏ cột tạm _full_query
    other_cols = [c for c in df_ok.columns if c not in ["query_id", "split", "_full_query"]]
    df_input_exported = df_ok[["query_id", "split"] + other_cols]

    # 4. Lưu ra file CSV
    df_input_exported.to_csv(output_csv_with_id, index=False, encoding="utf-8-sig")
    print(f"📁 Đã lưu bảng input gắn mã 5 chữ số tại: {output_csv_with_id}")
    # =============================================================


    # -------------------------------------------------------------
    # BƯỚC 4: XUẤT RA 2 FILE JSONL ĐỊNH DẠNG 5 CHỮ SỐ (:05d)
    # -------------------------------------------------------------

    # 4.1 File Train (Dùng cho khâu đào Negative của ES)
    with open(output_train_jsonl, "w", encoding="utf-8") as f_train:
        for idx, row in train_df.reset_index(drop=True).iterrows():
            item = {
                "query_id": f"train_q_{idx + 1:05d}",  # Chuẩn 5 chữ số
                "query": row["query"],
                "positive_ids": row["positive_ids"],
                "tax_type_for_negative": row["tax_type_for_negative"],
                "target_date": row["target_date"],
            }
            f_train.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 4.2 File Eval (Dùng để thi Test chấm điểm MRR / Hit@10)
    with open(output_eval_jsonl, "w", encoding="utf-8") as f_eval:
        for idx, row in eval_df.reset_index(drop=True).iterrows():
            eval_item = {
                "query_id": f"eval_q_{idx + 1:05d}",  # Chuẩn 5 chữ số
                "query": row["query"],
                "positive_ids": row["positive_ids"],
                "target_date": row[
                    "target_date"
                ],  # Giữ lại để lọc hiệu lực khi chấm thi
            }
            f_eval.write(json.dumps(eval_item, ensure_ascii=False) + "\n")

    print("\n" + "=" * 55)
    print("🎉 ĐÃ XUẤT XONG 2 FILE JSONL ĐẦY ĐỦ THÔNG TIN!")
    print(f"1. File Train (Dùng cho Mining ES): {output_train_jsonl}")
    print(f"   - Số lượng: {len(train_df)} câu hỏi")
    print(f"2. File Eval (Dùng để thi Test): {output_eval_jsonl}")
    print(f"   - Số lượng: {len(eval_df)} câu hỏi")
    print("=" * 55)








# =====================================================================
# HÀM BỔ TRỢ: XỬ LÝ NGÀY MỤC TIÊU
# =====================================================================
def parse_or_default_date(date_val):
  s = str(date_val).strip()
  if not s or s.lower() in ["nan", "none", "-", "null", ""]:
    return date.today().strftime("%Y-%m-%d")
  if s.endswith(".0"):
    s = s[:-2]
  try:
    return datetime.strptime(s, "%d/%m/%Y").strftime("%Y-%m-%d")
  except ValueError:
    try:
      return pd.to_datetime(s).strftime("%Y-%m-%d")
    except Exception:
      return s


# =====================================================================
# HÀM CHÍNH: LỌC, TẠO ID ĐỒNG BỘ VÀ CHIA TẬP TRAIN/EVAL
# =====================================================================
def export_and_split_queries_for_mining(
    input_csv_path,  # Nhận cả file .csv lẫn .xlsx / .xls
    output_csv_with_id,
    output_train_jsonl="data/train_queries_for_mining.jsonl",
    output_eval_jsonl="data/eval_queries.jsonl",
    eval_ratio=0.2,  # 80% Train, 20% Eval
    random_seed=42,  # Cố định hạt giống ngẫu nhiên
    result_col="Kiểm tra trích dẫn (ok / gợi ý)",
    tax_neg_col="tax_type_for_negative",
    target_date_col="target_date",
    context_col="Context",
    question_col="Question",
    positive_id_col="positive_id",
):
  print("--- BẮT ĐẦU CHUYỂN ĐỔI VÀ CHIA TẬP TRAIN/EVAL CHO MINING ---")

  # -------------------------------------------------------------
  # 1. ĐỌC FILE LINH HOẠT (HỖ TRỢ CẢ EXCEL LẪN CSV)
  # -------------------------------------------------------------
  ext = os.path.splitext(input_csv_path)[1].lower()
  if ext in [".xlsx", ".xls"]:
    df = pd.read_excel(input_csv_path)
  else:
    df = pd.read_csv(input_csv_path, keep_default_na=False, low_memory=False)

  # Chuẩn hóa khoảng trắng ở tên cột
  df.columns = [str(c).strip() for c in df.columns]
  print(f"-> Tổng số dòng trong file gốc: {len(df)}")

  # -------------------------------------------------------------
  # BƯỚC 1: LỌC DÒNG ĐẠT CHUẨN (CÓ THÊM ĐIỀU KIỆN TAX_TYPE_FOR_NEGATIVE)
  # -------------------------------------------------------------
  mask_corpus = df["is_in_corpus"].astype(str).str.upper() == "TRUE"
  mask_ok = (df[result_col].astype(str).str.strip().str.lower().str.startswith("ok") )

  # ⭐️ ĐIỀU KIỆN MỚI: tax_type_for_negative KHÔNG ĐƯỢC RỖNG / NAN / NONE
  tax_neg_clean = df[tax_neg_col].fillna("").astype(str).str.strip().str.lower()   # ⭐️ DÙNG fillna("") ĐỂ BIẾN TẤT CẢ Ô TRỐNG THÀNH CHUỖI RỖNG TRƯỚC:
  mask_tax_neg = ~tax_neg_clean.isin(["", "nan", "none", "null", "-", "<na>"]) # Thêm "<na>" vào danh sách loại trừ cho chắc ăn 100%:

  # Kết hợp cả 3 điều kiện
  df_ok = df[mask_corpus & mask_ok & mask_tax_neg].copy()
  print(f"-> Số dòng đạt chuẩn ('ok' + có trong Corpus + có tax_neg):"
        f" {len(df_ok)} dòng."  )

  if df_ok.empty:
    print("❌ Không có dòng nào đạt chuẩn! Vui lòng kiểm tra lại dữ liệu.")
    return

  # -------------------------------------------------------------
  # BƯỚC 2: TẠO QUERY HOÀN CHỈNH VÀ GOM NHÓM THEO CÂU HỎI
  # -------------------------------------------------------------
  has_ctx = context_col in df_ok.columns
  contexts = (
      df_ok[context_col].astype(str).str.strip()
      if has_ctx
      else [""] * len(df_ok)
  )
  questions = df_ok[question_col].astype(str).str.strip()

  # Ghép Context + Question
  full_queries = []
  for ctx, q in zip(contexts, questions):
    if ctx and q and ctx != q:
      full_queries.append(f"{ctx}\n{q}")
    else:
      full_queries.append(q or ctx)
  df_ok["_full_query"] = full_queries

  # Nhận diện cột positive_ids
  actual_pos_col = (positive_id_col if positive_id_col in df_ok.columns else "check_positive_id" )

  grouped_queries = []

  # Gom nhóm theo câu hỏi duy nhất (Multi-positive)
  for query_text, grp in df_ok.groupby("_full_query", sort=False):
    if not query_text.strip():
      continue

    first_row = grp.iloc[0]

    # Gom các positive_id (khử trùng lặp)
    pos_list = []
    for p_val in grp[actual_pos_col]:
      for p in str(p_val).split(";"):
        cleaned = p.strip()
        if cleaned and cleaned not in pos_list:
          pos_list.append(cleaned)

    if not pos_list:
      continue

    t_date = parse_or_default_date(first_row.get(target_date_col, ""))
    tax_for_neg = str(first_row.get(tax_neg_col, "")).strip()

    grouped_queries.append({
        "query": query_text,
        "positive_ids": pos_list,
        "tax_type_for_negative": tax_for_neg,
        "target_date": t_date,
    })

  df_final = pd.DataFrame(grouped_queries)
  print(f"-> Tổng số câu hỏi duy nhất sau khi gom: {len(df_final)}")

  # -------------------------------------------------------------
  # BƯỚC 3: CHIA TẬP TRAIN (80%) VÀ EVAL (20%)
  # -------------------------------------------------------------
  eval_df = df_final.sample(frac=eval_ratio, random_state=random_seed)
  train_df = df_final.drop(eval_df.index)

  # -------------------------------------------------------------
  # ⭐️ BƯỚC 4: SINH MÃ ID ĐỒNG BỘ 100% GIỮA CSV VÀ JSONL
  # -------------------------------------------------------------
  # Tạo mã train_q_00001, train_q_00002... và eval_q_00001, eval_q_00002...
  query_to_id = {}
  query_to_split = {}

  # Gán ID cho tập Train
  train_items = []
  for idx, row in train_df.reset_index(drop=True).iterrows():
    qid = f"train_q_{idx + 1:05d}"
    query_to_id[row["query"]] = qid
    query_to_split[row["query"]] = "train"
    train_items.append({
        "query_id": qid,
        "query": row["query"],
        "positive_ids": row["positive_ids"],
        "tax_type_for_negative": row["tax_type_for_negative"],
        "target_date": row["target_date"],
    })

  # Gán ID cho tập Eval
  eval_items = []
  for idx, row in eval_df.reset_index(drop=True).iterrows():
    qid = f"eval_q_{idx + 1:05d}"
    query_to_id[row["query"]] = qid
    query_to_split[row["query"]] = "eval"
    eval_items.append({
        "query_id": qid,
        "query": row["query"],
        "positive_ids": row["positive_ids"],
        "target_date": row["target_date"],
    })

  # =============================================================
  # 5. XUẤT BẢNG CSV GẮN MÃ TRÙNG KHỚP VỚI JSONL
  # =============================================================
  # Gán thẳng mã train_q_... / eval_q_... vào bảng CSV
  df_ok["query_id"] = df_ok["_full_query"].map(query_to_id)
  df_ok["split"] = df_ok["_full_query"].map(query_to_split)

  # Đưa query_id và split lên 2 cột đầu tiên
  other_cols = [
      c for c in df_ok.columns if c not in ["query_id", "split", "_full_query"]
  ]
  df_input_exported = df_ok[["query_id", "split"] + other_cols]

  os.makedirs(os.path.dirname(output_csv_with_id), exist_ok=True)
  df_input_exported.to_csv(
      output_csv_with_id, index=False, encoding="utf-8-sig"
  )
  print(f"📁 Đã lưu bảng CSV (ID đồng bộ với JSONL) tại: {output_csv_with_id}")

  # =============================================================
  # 6. XUẤT 2 FILE JSONL ĐỒNG BỘ MÃ ID
  # =============================================================
  os.makedirs(os.path.dirname(output_train_jsonl), exist_ok=True)
  with open(output_train_jsonl, "w", encoding="utf-8") as f_train:
    for item in train_items:
      f_train.write(json.dumps(item, ensure_ascii=False) + "\n")

  os.makedirs(os.path.dirname(output_eval_jsonl), exist_ok=True)
  with open(output_eval_jsonl, "w", encoding="utf-8") as f_eval:
    for item in eval_items:
      f_eval.write(json.dumps(item, ensure_ascii=False) + "\n")

  print("\n" + "=" * 60)
  print("🎉 ĐÃ HOÀN TẤT CHUYỂN ĐỔI VÀ PHÂN CHIA:")
  print(f"1. File Train JSONL: {output_train_jsonl} ({len(train_items)} câu)")
  print(f"2. File Eval JSONL : {output_eval_jsonl} ({len(eval_items)} câu)")
  print(f"3. File CSV gắn mã  : {output_csv_with_id}")
  print("=" * 60)




# export_and_split_queries_for_mining(
#     input_csv_path="3-query_check_5.csv",
#     output_csv_with_id="3-query_check_5_with_id_5digit.csv",
#     output_train_jsonl="3-train_queries_for_mining.jsonl",
#     output_eval_jsonl="3-eval_queries.jsonl",
#     eval_ratio=0.2,
# )


# df = pd.read_excel("data/queries_with_tax_type_for_negative.xlsx")  # hoặc file của bạn

# print("1. Số dòng is_in_corpus = TRUE :", (df["is_in_corpus"].astype(str).str.upper() == "TRUE").sum())
# print("2. Bắt đầu bằng 'ok' (Python)    :", df["Kiểm tra trích dẫn (ok / gợi ý)"].astype(str).str.strip().str.lower().str.startswith("ok").sum())
# print("3. Bằng ĐÚNG chữ 'ok' (Excel)   :", (df["Kiểm tra trích dẫn (ok / gợi ý)"].astype(str).str.strip().str.lower() == "ok").sum())
# print("4. tax_neg KHÁC rỗng            :", (~df["tax_type_for_negative"].astype(str).str.strip().str.lower().isin(["", "nan", "none", "-"])).sum())

# # ⭐️ Đảm bảo trỏ đúng file bạn đang mở trên Excel:

# col = "tax_type_for_negative"

# print("1. Số dòng đọc được:", len(df))
# print(
#     "2. Số dòng THỰC SỰ RỖNG:",
#     (df[col].isna() | (df[col].astype(str).str.strip() == "")).sum(),
# )
# print("3. Các giá trị xuất hiện nhiều nhất:\n", df[col].value_counts().head(5))