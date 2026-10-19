import json
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config 


def generate_expanded_hierarchy_map():
  """
  Mã cấp Điều : nhồi 100% đầy đủ toàn bộ Khoản và Điểm con bên dưới nó.
  Mã cấp Khoản: nhồi câu dẫn của Khoản + toàn bộ Điểm con trực thuộc nó.
  Mã cấp Điểm : Giữ nguyên văn bản cô đọng của Điểm đó.
  {  
    "320/2025/nd-cp_10": "Điều 10. Các khoản chi không được trừ khi xác định thu nhập chịu thuế\n\n1. Các khoản chi không đáp ứng đủ các điều kiện quy định tại khoản 1 Điều 9, trừ các trường hợp sau:\n\na) Khoản chi thực tế phát sinh liên quan đến hoạt động sản xuất, kinh doanh của doanh nghiệp;\n\nb) Khoản chi có đủ hoá đơn, chứng từ hợp pháp theo quy định của pháp luật.\n\n2. Chi tiền lương, tiền công, tiền thưởng cho người lao động nhưng thực tế không chi trả.",
  
    "320/2025/nd-cp_10_1": "1. Các khoản chi không đáp ứng đủ các điều kiện quy định tại khoản 1 Điều 9, trừ các trường hợp sau:\n\na) Khoản chi thực tế phát sinh liên quan đến hoạt động sản xuất, kinh doanh của doanh nghiệp;\n\nb) Khoản chi có đủ hoá đơn, chứng từ hợp pháp theo quy định của pháp luật.",
    
    "320/2025/nd-cp_10_1_a": "a) Khoản chi thực tế phát sinh liên quan đến hoạt động sản xuất, kinh doanh của doanh nghiệp;",
    
    "320/2025/nd-cp_10_1_b": "b) Khoản chi có đủ hoá đơn, chứng từ hợp pháp theo quy định của pháp luật.",
    
    "320/2025/nd-cp_10_2": "2. Chi tiền lương, tiền công, tiền thưởng cho người lao động nhưng thực tế không chi trả."
  }
  """
  print("-> Đang đọc kho luật từ:", config.CORPUS_FINAL_JSON)              # "data/3-final_legal_corpus.json"
  with open(config.CORPUS_FINAL_JSON, "r", encoding="utf-8") as f:
    corpus_data = json.load(f)

  corpus_list = (
      list(corpus_data.values())
      if isinstance(corpus_data, dict)
      else corpus_data
  )

  # 1. Gom nhóm văn bản theo từng Điều: (law_no, article_id)
  articles_tree = {}
  for doc in corpus_list:
    law = str(doc.get("law_no", "")).strip()
    art = str(doc.get("article_id", "")).strip()
    if not law or not art:
      continue

    art_key = (law, art)
    if art_key not in articles_tree:
      articles_tree[art_key] = []
    articles_tree[art_key].append(doc)

  # 2. Xây dựng bản đồ nội dung mở rộng: {chunk_id: expanded_content}
  expanded_map = {}

  for art_key, docs in articles_tree.items():
    # -------------------------------------------------------------
    # A. CẤP ĐIỀU: Ghép toàn bộ nội dung của Điều đó (Tất cả Khoản + Điểm)
    # -------------------------------------------------------------
    # TÌM "chunk_id" CỦA CẤP ĐIỀU ĐỂ LÀM "KEY" LƯU VÀO TỪ ĐIỂN! (clause_id rỗng và point_id rỗng)
    article_doc = next( ( d for d in docs  if str(d.get("clause_id", "")).strip() == "" 
                                          and str(d.get("point_id", "")).strip() == ""  ), None, )
    
    if article_doc:
      art_chunk_id = article_doc.get("chunk_id")
      # Gom toàn bộ content của mọi phần tử trong Điều này theo đúng thứ tự
      all_contents = [ d.get("content", "").strip() for d in docs if d.get("content") ]

      # Loại bỏ các đoạn trùng lặp (nếu có) nhưng giữ nguyên thứ tự
      unique_contents = list(dict.fromkeys(all_contents))
      expanded_map[art_chunk_id] = "\n\n".join(unique_contents)

    # -------------------------------------------------------------
    # B. CẤP KHOẢN: Ghép câu dẫn của Khoản + toàn bộ Điểm con của nó
    # -------------------------------------------------------------
    # Lấy danh sách các Khoản tồn tại trong Điều này
    clause_ids = set( str(d.get("clause_id", "")).strip()
                      for d in docs if str(d.get("clause_id", "")).strip() != ""    )

    for c_id in clause_ids:
      # TÌM "chunk_id" CỦA CẤP Khoản ĐỂ LÀM "KEY" LƯU VÀO TỪ ĐIỂN! (point_id rỗng)
      clause_doc = next( ( d for d in docs  if str(d.get("clause_id", "")).strip() == c_id
                                          and str(d.get("point_id", "")).strip() == ""  ), None,  )
      if clause_doc:
        clause_chunk_id = clause_doc.get("chunk_id")
        # Gom nội dung của Khoản này và tất cả các Điểm trực thuộc Khoản này
        clause_and_points = [ d.get("content", "").strip()
                              for d in docs
                              if str(d.get("clause_id", "")).strip() == c_id and d.get("content")
        ]
        unique_clause_contents = list(dict.fromkeys(clause_and_points))
        expanded_map[clause_chunk_id] = "\n\n".join(unique_clause_contents)

    # -------------------------------------------------------------
    # C. CẤP ĐIỂM: Giữ nguyên nội dung của chính Điểm đó
    # -------------------------------------------------------------
    for d in docs:
      if str(d.get("point_id", "")).strip() != "":
        point_chunk_id = d.get("chunk_id")
        expanded_map[point_chunk_id] = d.get("content", "").strip()

  # 3. Xuất ra file JSON tĩnh
  os.makedirs(os.path.dirname(config.CORPUS_MAP_FINAL_JSON), exist_ok=True)  # tạo folder nếu chưa có
  with open(config.CORPUS_MAP_FINAL_JSON, "w", encoding="utf-8") as f_out:
    json.dump(expanded_map, f_out, ensure_ascii=False, indent=2)

  print( f"🎉 THÀNH CÔNG! Đã tạo bản đồ mở rộng cho {len(expanded_map)} chunks pháp lý."  )
  print( f"📁 Lưu tại: {config.CORPUS_MAP_FINAL_JSON}" )


if __name__ == "__main__":
  generate_expanded_hierarchy_map()