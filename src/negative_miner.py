import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from elasticsearch import Elasticsearch, helpers
from datetime import date
import pandas as pd
import config

# =====================================================================
# 1. HÀM TẠO INDEX VÀ NẠP CORPUS VÀO ELASTICSEARCH
# =====================================================================
def setup_and_index_corpus(
    es_client, 
    corpus_records
):
  """Tạo index với mapping tối ưu và nạp toàn bộ corpus vào Elasticsearch."""
  # Nếu index đã tồn tại thì xóa đi tạo mới (để làm sạch dữ liệu)
  if es_client.indices.exists(index=config.ES_INDEX_NAME):
    print(f"-> Index '{config.ES_INDEX_NAME}' đã tồn tại, đang xóa để làm mới...")
    es_client.indices.delete(index=config.ES_INDEX_NAME)

  # Định nghĩa Mapping: Phân tách rõ content và hierarchy_context
  mapping = {
      "mappings": {
          "properties": {
              "uuid": {"type": "keyword"},
              "chunk_id": {"type": "keyword"},
              "law_no": {"type": "keyword"},
              "article_id": {"type": "keyword"},
              "doc_type": {"type": "keyword"},
              "status": {"type": "keyword"},
              "tax_type": {"type": "keyword"},  # ⭐️ BẮT BUỘC PHẢI CÓ DÒNG NÀY
              "effective_date": {"type": "date", "format": "yyyy-MM-dd"},
              "expire_date": {"type": "date", "format": "yyyy-MM-dd"},
              # Trường dùng để BM25 search
              "hierarchy_context": {"type": "text"},
              "content": {"type": "text"},
              "search_text": {"type": "text"},
          }
      }
  }

  es_client.indices.create(index=config.ES_INDEX_NAME, body=mapping)
  print(f"1. Đã tạo index '{config.ES_INDEX_NAME}' thành công!")

  # Nạp dữ liệu bằng Bulk API
  def bulk_actions():
    for doc in corpus_records:
      yield {"_index": config.ES_INDEX_NAME, "_id": doc["chunk_id"], "_source": doc}

  # success_count, _ = helpers.bulk(es_client, bulk_actions())
  # Ép ES đồng bộ dữ liệu vào đĩa ngay lập tức
  # es_client.indices.refresh(index=config.ES_INDEX_NAME)
  # print(f"2. Đã nạp thành công {success_count} chunks vào Elasticsearch!")

  try:
    success_count, _ = helpers.bulk(es_client, bulk_actions())
    es_client.indices.refresh(index=config.ES_INDEX_NAME)
    print(f"2. Đã nạp thành công {success_count} chunks vào Elasticsearch!")
    print(es_client.count(index="vietnamese_legal_corpus_clean_4")["count"])

  except helpers.BulkIndexError as e:
    print("\n" + "=" * 60)
    print(f"❌ TỔNG CỘNG CÓ {len(e.errors)} DÒNG BỊ LỖI. ĐÂY LÀ LÝ DO CỦA DÒNG ĐẦU TIÊN:")
    # In trực tiếp lý do từ Elasticsearch:
    print(e.errors[0])
    print("=" * 60 + "\n")


# =====================================================================
# 2. HÀM CHÍNH: MINING HARD NEGATIVES TỪ QUERY
# =====================================================================
TODAY_STR = date.today().strftime("%Y-%m-%d")


def mine_hard_negatives(
    es_client,
    num_negatives=5,
    top_k_search=30,
    skip_top_k=0,
    c_chunk_id_col="chunk_id",
    c_tax_type_col="tax_type",
    c_context_col="hierarchy_context",
    c_content_col="content",
):
  print("--- BẮT ĐẦU QUY TRÌNH MINING & XUẤT CSV KIỂM TRA ---")

  # -------------------------------------------------------------
  # BƯỚC 0: NẠP CORPUS VÀO RAM
  # -------------------------------------------------------------
  with open(config.CORPUS_FINAL_JSON, "r", encoding="utf-8") as f:
    corpus_data = json.load(f)

  records      = (corpus_data if isinstance(corpus_data, list) else corpus_data.values() )
  corpus_cache = { str(doc.get(c_chunk_id_col, "")).strip(): doc for doc in records }
  print(f"-> Đã nạp {len(corpus_cache)} chunks từ kho Corpus vào RAM.")


  # -------------------------------------------------------------
  # BƯỚC 1: KIỂM TRA XEM INDEX ELASTICSEARCH CÓ DỮ LIỆU KHÔNG
  # -------------------------------------------------------------
  need_index = False
  if not es_client.indices.exists(index=config.ES_INDEX_NAME):
      need_index = True
  elif es_client.count(index=config.ES_INDEX_NAME)["count"] == 0:
      need_index = True

  if need_index:
      print(f"-> Index '{config.ES_INDEX_NAME}' chưa có dữ liệu. Đang tự động nạp kho luật vào Elasticsearch...")
      # ⭐️ GỌI HÀM NẠP TỰ ĐỘNG Ở ĐÂY:
      setup_and_index_corpus(es_client, records)
      print("-> Đã nạp xong toàn bộ kho luật vào Elasticsearch!")
  else:
      doc_count = es_client.count(index=config.ES_INDEX_NAME)["count"]
      print(f"-> Index ES '{config.ES_INDEX_NAME}' đã có sẵn {doc_count} tài liệu. Bỏ qua bước nạp!")

 
  # -------------------------------------------------------------
  # BƯỚC 2: ĐỌC FILE QUERIES
  # -------------------------------------------------------------
  queries_data = []
  with open(config.TRAIN_QUERIES_JSONL, "r", encoding="utf-8") as f:
    for line in f:
      if line.strip():
        queries_data.append(json.loads(line))
  print(f"-> Đã nạp {len(queries_data)} câu hỏi cần đào Negatives.")

  train_records = []
  audit_rows = []
  skipped_pos_missing = 0

  # -------------------------------------------------------------
  # BƯỚC 3: DUYỆT TỪNG QUERY
  # -------------------------------------------------------------
  for idx, item in enumerate(queries_data):
    qid = item.get("query_id", f"q_{idx+1}")
    q_text = item.get("query", "").strip()
    pos_ids = item.get("positive_ids", [])

    if not q_text or not pos_ids:
      continue

    valid_pos_ids = [pid for pid in pos_ids if pid in corpus_cache]
    if not valid_pos_ids:
      skipped_pos_missing += 1
      continue

    # -------------------------------------------------------------
    # 3.1 Gom nội dung Positive
    # -------------------------------------------------------------
    pos_contexts = [corpus_cache[pid].get(c_context_col, "") for pid in valid_pos_ids]
    pos_contents = [corpus_cache[pid].get(c_content_col, "") for pid in valid_pos_ids]
    pos_contexts_str = "\n---\n".join(c for c in pos_contexts if c)
    pos_contents_str = "\n\n=====\n\n".join(c for c in pos_contents if c)

    # -------------------------------------------------------------
    # 3.2 Vùng cấm kỵ: Cùng Văn bản + Cùng Điều
    # -------------------------------------------------------------
    protected_articles = set()
    for pid in valid_pos_ids:
      p_doc = corpus_cache[pid]
      p_law = p_doc.get("law_no", "")
      p_art = p_doc.get("article_id", "")
      if p_law and p_art:
        protected_articles.add((p_law, p_art))

    # -------------------------------------------------------------
    # ⭐️ 3.3 SỬA LỖI LỌC NGÀY: CHỈ LỌC KHI CÓ NGÀY HỢP LỆ
    # (Tuyệt đối không ép ngày hôm nay nếu query không yêu cầu)
    # -------------------------------------------------------------
    filter_conditions = []
    t_date = item.get("target_date")

    if t_date and str(t_date).strip().lower() not in ["none","nan","","null",]:
      clean_date = str(t_date).strip()
      filter_conditions.append(
          {"range": {"effective_date": {"lte": clean_date}}}
      )
      filter_conditions.append({
          "bool": {
              "should": [
                  {"bool": {"must_not": {"exists": {"field": "expire_date"}}}},
                  {"range": {"expire_date": {"gte": clean_date}}},
              ]
          }
      })

    # -------------------------------------------------------------
    # ⭐️ 3.4 SỬA LỖI TAX_TYPE: KHỬ CHỮ "NONE" / "NAN"
    # -------------------------------------------------------------
    raw_tax_neg = item.get("tax_type_for_negative")
    allowed_taxes = []

    if raw_tax_neg and str(raw_tax_neg).strip().lower() not in ["none","nan","","null",]:
      allowed_taxes = [ t.strip() for t in str(raw_tax_neg).split(";")
                                  if t.strip() and t.strip().lower() not in ["none", "nan", "null"]   ]

    if allowed_taxes:
      filter_conditions.append({"terms": {"tax_type": allowed_taxes}})

    # -------------------------------------------------------------
    # 3.5 BẮN QUERY VÀO ELASTICSEARCH
    # -------------------------------------------------------------
    bool_query = {
        "must": {
            "multi_match": {
                "query": q_text,
                "fields": ["content^3", "hierarchy_context^1"],
            }
        }
    }
    if filter_conditions:
      bool_query["filter"] = filter_conditions

    search_query = {"query": {"bool": bool_query}, "size": top_k_search}

    response = es_client.search(index=config.ES_INDEX_NAME, body=search_query)
    hits = response["hits"]["hits"]

    # In kiểm tra câu đầu tiên xem ES có nhả kết quả không
    if idx == 0:
      print(f"\n🔍 [DEBUG CÂU ĐẦU TIÊN]:")
      print(f"   - Số kết quả ES tìm thấy: {len(hits)} hits")
      if filter_conditions:
        print(f"   - Bộ lọc áp dụng: {filter_conditions}")
      else:
        print("   - Tìm kiếm TỰ DO (Không bị chặn bởi filter nào).")

    # -------------------------------------------------------------
    # 3.6 ĐÀO NEGATIVES
    # -------------------------------------------------------------
    neg_texts = []
    skipped_count = 0
    rank_idx = 1

    for hit in hits:
      cand_id = hit["_id"]
      cand_src = hit["_source"]
      neg_score = hit.get("_score", 0.0)

      # Bỏ qua nếu là positive
      if cand_id in valid_pos_ids:
        continue

      cand_law = cand_src.get("law_no", "")
      cand_art = cand_src.get("article_id", "")

      # Bỏ qua nếu cùng Điều với positive
      if (cand_law, cand_art) in protected_articles:
        continue

      if skipped_count < skip_top_k:
        skipped_count += 1
        continue

      if cand_id in corpus_cache:
        neg_item = corpus_cache[cand_id]
        neg_texts.append(neg_item.get("search_text", ""))

        if config.INSPECT_CSV_PATH:
          audit_rows.append({
              "query_id": qid,
              "query": q_text,
              "target_date": t_date if t_date else "",
              "tax_filter": ( "; ".join(allowed_taxes) if allowed_taxes else "TỰ DO" ),
              "positive_ids": "; ".join(valid_pos_ids),
              "pos_hierarchy_context": pos_contexts_str,
              "pos_content": pos_contents_str,
              "neg_rank": rank_idx,
              "neg_score": round(float(neg_score), 4),
              "neg_chunk_id": neg_item.get(c_chunk_id_col, ""),
              "neg_tax_type": neg_item.get(c_tax_type_col, ""),
              "neg_hierarchy_context": neg_item.get(c_context_col, ""),
              "neg_content": neg_item.get(c_content_col, ""),
          })
        rank_idx += 1

      if len(neg_texts) >= num_negatives:
        break

    # 3.7 Đóng gói file JSONL đem train
    if neg_texts:
      for pid in valid_pos_ids:
        pos_text = corpus_cache[pid].get("search_text", "")
        train_records.append( {"query": q_text, "pos": [pos_text], "neg": neg_texts} )

  # -------------------------------------------------------------
  # 4 & 5. XUẤT FILE KẾT QUẢ
  # -------------------------------------------------------------
  with open(config.MINED_TRAIN_JSONL, "w", encoding="utf-8") as f_out:
    for r in train_records:
      f_out.write(json.dumps(r, ensure_ascii=False) + "\n")

  if config.INSPECT_CSV_PATH and audit_rows:
    df_audit = pd.DataFrame(audit_rows)
    df_audit.to_csv(config.INSPECT_CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"📁 1. Đã lưu file CSV đối chiếu tại: {config.INSPECT_CSV_PATH}")

  print("\n" + "=" * 55)
  print(f"🎉 HOÀN THÀNH MINING!")
  print(f"Tổng số mẫu train tạo được: {len(train_records)}")
  if skipped_pos_missing > 0:
    print(f"⚠️ Số câu bị bỏ qua do không tìm thấy positive_id: {skipped_pos_missing}")
  print(f"📁 2. Đã lưu file JSONL train tại: {config.MINED_TRAIN_JSONL}")
  print("=" * 55)



  