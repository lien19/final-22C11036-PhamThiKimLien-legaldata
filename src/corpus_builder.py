import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import uuid
import pandas as pd
from datetime import date, datetime
import itertools
import re

import config

#===============================================================
# 1.
#===============================================================
def universal_legal_corpus_processor(
    csv_path=config.RAW_LEGAL_CSV,
    output_jsonl_path=config.CORPUS_BASE_JSONL,
    output_json_path=config.CORPUS_BASE_JSON,
    output_csv_path=config.CORPUS_ID_CSV,
    law_no_column="law_no",
    hierarchy_columns=["article_id", "clause_id", "point_id"],
    content_column="content",
    tax_type_column="tax_type",
):
  """[CONTEXT] Tự động gom nhóm đa tầng ngữ cảnh theo đúng quy tắc kế thừa:
  - Cấp Điều: Văn bản > Điều X                                  ==> Văn bản: 320/2025/nd-cp > Điều 2
  - Cấp Khoản: Văn bản > Điều X: [Nội dung Điều] > Khoản Y      ==> Văn bản: 80/2021/tt-btc > Điều 73. Các tổ chức, cá nhân... > Khoản 3
  - Cấp Điểm: Văn bản > Điều X: [Nội dung Điều] > Khoản Y: [Nội dung Khoản] >  Điểm Z    ==> Văn bản: 80/2021/tt-btc > Điều 73. Các tổ chức... > Khoản 3: 3. Tổ chức, đại lý thuế... > Điểm a
  Đồng thời, hàm trên đã tích hợp sẵn lệnh list(dict.fromkeys(...)) ở mọi bước, xóa sạch vĩnh viễn lỗi dính đúp - Điều 4... - Điều 4... bị lặp chữ trước đây!
  """
  
  """
  Tự động gom nhóm đa tầng ngữ cảnh cho mọi loại văn bản pháp luật,
  áp dụng kỹ thuật Parent-Child phẳng (Context Injection) cho Elasticsearch và LLM.
  
  # Lần 1:  lập "SỔ GIA PHẢ" (chỉ ghi lại Tiêu đề của Cha và Ông nội, KHÔNG lấy nội dung của Điểm)
      # 0 là tầng Điều:
      article_context_map= {
          ('78/2014/TT-BTC', '2'): "Người nộp thuế",
          ('78/2014/TT-BTC', '3'): "Thu nhập chịu thuế",
          ('78/2014/TT-BTC', '4'): "Thu nhập được miễn thuế"
      },
      # 1 là tầng Khoản:
      clause_context_map = {
          ('78/2014/TT-BTC', '2', '1'): "Người nộp thuế TNDN là tổ chức... bao gồm:"
      }
  
  # Lần 2: Đi đến từng Điểm/Khoản chi tiết, tra cuốn sổ tạm đó để lấy tên "cha/ông" gắn vào đầu, rồi xuất ra JSON chuẩn chỉnh cho Elasticsearch và RAG.
      Ở level_idx = 0 (Cột article_id = '5'):       Lấy chìa khóa ('Luật 78', '5') mở cuốn Sổ cấp 0. Tra ra: "Quyền hạn của cơ quan Thống kê".
      →   Append vào list: "Điều 5: Quyền hạn của cơ quan Thống kê".
      Ở level_idx = 1 (Cột clause_id = '1'):        Lấy chìa khóa ('Luật 78', '5', '1') mở cuốn Sổ cấp 1. Tra ra: "Khi thu thập dữ liệu, có quyền sau:".
      →   Append vào list: "Khoản 1: Khi thu thập dữ liệu, có quyền sau:".
      Ở level_idx = 2 (Cột point_id = 'a'):         Sổ cấp 2 không có tiêu đề   → lấy mặc định tên nhãn.
      →   Append vào list: "Điểm a".
  """
  
  # -----------------------------------------------------------------
  # BƯỚC 1: Đọc file chống lỗi kiểu dữ liệu
  # -----------------------------------------------------------------
  df = pd.read_csv(csv_path, keep_default_na=False, low_memory=False)

  # -----------------------------------------------------------------
  # BƯỚC 2: Chuẩn hóa dữ liệu
  # -----------------------------------------------------------------
  all_id_cols = [law_no_column] + hierarchy_columns
  for col in all_id_cols:
    if col in df.columns:
      df[col] = df[col].astype(str).str.strip()

  df[content_column] = df[content_column].astype(str).str.strip()

  # BỔ SUNG ID VÀO DF GỐC & LƯU CSV
  df["chunk_id"] = [ "_".join( str(val).strip() for val in vals if str(val).strip() != "" ).replace(" ", "_")
                                                for vals in zip(*(df[col] for col in all_id_cols))  ]

  df["uuid"] = [str(uuid.uuid5(uuid.NAMESPACE_DNS, cid)) if cid else ""
                for cid in df["chunk_id"]  ]

  cols_order = ["uuid", "chunk_id"] + [c for c in df.columns if c not in ["uuid", "chunk_id"]]
  df = df[cols_order]
  df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
  print(f"1. Đã lưu file CSV tại: {output_csv_path}")

  # -----------------------------------------------------------------
  # BƯỚC 3: Lập "SỔ GIA PHẢ" (Tách bạch tiêu đề Điều và câu dẫn Khoản)
  # -----------------------------------------------------------------
  art_col = hierarchy_columns[0]  # Thường là article_id
  clau_col = (hierarchy_columns[1] if len(hierarchy_columns) > 1 else None )  # clause_id
  point_col = (hierarchy_columns[2] if len(hierarchy_columns) > 2 else None)  # point_id
  # -----------------------------------------------------------------
  # 3.1 Sổ cấp 0: Lưu tiêu đề / câu mở đầu của ĐIỀU (khoản và điểm phải rỗng)
  article_context_map = {}
  cond_art = df[art_col] != ""
  if clau_col:
    cond_art = cond_art & (df[clau_col] == "")
  if point_col:
    cond_art = cond_art & (df[point_col] == "")

  for keys, g in df[cond_art].groupby([law_no_column, art_col], sort=False):
    # Lọc trùng lặp chuỗi trước khi nối
    raw_texts = [
        str(c).strip()
        for c in g[content_column].dropna()
        if str(c).strip() and str(c).strip().lower() not in ["nan", "none"]
    ]
    unique_texts = list(dict.fromkeys(raw_texts))
    article_context_map[keys] = " - ".join(unique_texts)
  # -----------------------------------------------------------------
  # 3.2 Sổ cấp 1: Lưu câu dẫn / nội dung của KHOẢN (điểm phải rỗng)
  clause_context_map = {}
  if clau_col:
    cond_clau = (df[art_col] != "") & (df[clau_col] != "")
    if point_col:
      cond_clau = cond_clau & (df[point_col] == "")

    for keys, g in df[cond_clau].groupby( [law_no_column, art_col, clau_col], sort=False ):
      raw_texts = [
          str(c).strip()
          for c in g[content_column].dropna()
          if str(c).strip() and str(c).strip().lower() not in ["nan", "none"]
      ]
      unique_texts = list(dict.fromkeys(raw_texts))
      clause_context_map[keys] = " - ".join(unique_texts)

  # -----------------------------------------------------------------
  # BƯỚC 4: TẠO DOCUMENT VỚI NGỮ CẢNH CHUẨN XÁC TỪNG CẤP
  # -----------------------------------------------------------------
  processed_records = []

  for keys, group in df.groupby(all_id_cols, sort=False):
    row_data = dict(zip(all_id_cols, keys))
    first_row = group.iloc[0]
    tax_type = str(first_row.get(tax_type_column, "")).strip()

    # Bỏ qua dòng rác nếu tất cả cấp phân cấp đều rỗng
    if all(row_data[col] == "" for col in hierarchy_columns):
      continue
    # -----------------------------------------------------------------
    # Lấy nội dung chi tiết của chính chunk này (khử trùng lặp)
    raw_contents = [
        str(c).strip()
        for c in group[content_column].dropna()
        if str(c).strip() and str(c).strip().lower() not in ["nan", "none"]
    ]
    unique_contents = list(dict.fromkeys(raw_contents))
    if not unique_contents:
      continue
    full_content = "\n\n".join(unique_contents)
    # -----------------------------------------------------------------
    # Trích xuất các giá trị ID
    law_val = row_data[law_no_column]
    art_val = row_data[art_col]
    clau_val = row_data.get(clau_col, "") if clau_col else ""
    point_val = row_data.get(point_col, "") if point_col else ""

    # =============================================================
    # ⭐️ [CONTEXT] BẮT ĐẦU RÁP NGỮ CẢNH, PHÂN BIỆT RẠCH RÒI 3 CẤP
    # =============================================================
    # TRƯỜNG HỢP 1: Chunk là CẤP ĐIỀU (Khoản rỗng VÀ Điểm rỗng)
    if clau_val == "" and point_val == "":
      # ✅ Cấp Điều CHỈ CẦN nhãn Điều, TUYỆT ĐỐI KHÔNG lấy tiêu đề Điều vào context
      context_parts = [f"Văn bản: {law_val}", f"Điều {art_val}"]

    # TRƯỜNG HỢP 2: Chunk là CẤP KHOẢN (Khoản có chữ, nhưng Điểm rỗng)
    elif clau_val != "" and point_val == "":
      art_text = article_context_map.get((law_val, art_val), "").strip()
      art_part = f"Điều {art_val}: {art_text}" if art_text else f"Điều {art_val}"
      # ✅ Cấp Khoản: Tiêm nội dung của Điều cha (art_part) + Nhãn Khoản
      context_parts = [f"Văn bản: {law_val}", art_part, f"Khoản {clau_val}"]

    # TRƯỜNG HỢP 3: Chunk là CẤP ĐIỂM (Điểm có chữ)
    elif point_val != "":
      # Lấy câu dẫn của Khoản cha
      clau_text = clause_context_map.get( (law_val, art_val, clau_val), "").strip()
      clau_part = (f"Khoản {clau_val}: {clau_text}" if clau_text else f"Khoản {clau_val}" )
      # ✅ Cấp Điểm: Tiêm nội dung Điều ông nội (art_part) + Nội dung Khoản cha (clau_part) + Nhãn Điểm
      context_parts = [
          f"Văn bản: {law_val}",
          art_part,
          clau_part,
          f"Điểm {point_val}",
      ] 


    # Tạo chuỗi ngữ cảnh hoàn chỉnh
    full_hierarchy_context = " > ".join(context_parts)
    search_text = (f"{full_hierarchy_context}\nNội dung chi tiết:\n{full_content}" )

    chunk_id = group["chunk_id"].iloc[0]
    doc_uuid = group["uuid"].iloc[0]

    doc = {
        "uuid": doc_uuid,
        "chunk_id": chunk_id,
        **row_data,
        "hierarchy_context": full_hierarchy_context,
        "content": full_content,
        "search_text": search_text,
        "tax_type": tax_type,
    }
    processed_records.append(doc)

  # -----------------------------------------------------------------
  # BƯỚC 5: XUẤT FILE
  # -----------------------------------------------------------------
  with open(output_json_path, "w", encoding="utf-8") as f_json:
    json.dump(processed_records, f_json, ensure_ascii=False, indent=4)

  with open(output_jsonl_path, "w", encoding="utf-8") as f_jsonl:
    for item in processed_records:
      f_jsonl.write(json.dumps(item, ensure_ascii=False) + "\n")

  print("--- THÀNH CÔNG ---")
  print(f"2. Đã tạo file JSON tại: {output_json_path}")
  print(f"3. Đã tạo file JSONL tại: {output_jsonl_path}")
  print(f"Tổng số chunks hợp nhất: {len(processed_records)}")



#===============================================================
# 2.
#===============================================================
def get_col_not_in_groups_csvtable(df, field_groups):   # Tạo list chứa tall field trong all group # Trả về list các cột nếu nó KHÔNG khớp với pattern  
    """ Đã ktra format field_groups ở hàm unpivot_group_data """
    import itertools
    all_fields = list(itertools.chain.from_iterable(field_groups))
    pattern    = re.compile(rf"^({'|'.join(all_fields)})(\d+)$")       
    return [col for col in df.columns if not pattern.match(col)]


def get_group_numbers_csvtable(df, fields):  # ví dụ field (article, clause, point)
    pattern = re.compile(rf"^({'|'.join(fields)})(\d+)$")
    return sorted(list(
                        set(int(m.group(2)) for col in df.columns if (m := pattern.match(col)))
                    ))


def unpivot_group_data(
    field_groups=["law_edited_", "article_edited_", "clause_edited_", "point_edited_"],  # sắp xếp theo thứ tự phân cấp
    csv_legal_corpus=config.CORPUS_ID_CSV,
    output_csv_path=config.UNPIVOT_CSV,
    output_edit_json=config.EDIT_MAP_JSON,
):
    import pandas as pd
    import itertools
    """
    Hàm tổng quát dùng để rải hàng (unpivot) các nhóm cột có đuôi số giống nhau.
    
    Tham số:
    - df: DataFrame gốc đọc từ CSV
    - field_groups: List các danh sách trường (Ví dụ: [['Dienthoai'], ['Maytinh']])
    - group_combinations: Danh sách các tổ hợp số đuôi (Ví dụ: [(1,1), (1,2)])
    - col_not_in_groups: Danh sách các cột chung cần giữ lại
    """
    
    # Nếu người dùng truyền vào 1 danh sách đơn lẻ, ví dụ: ['Dienthoai', 'Maytinh']
    # nhưng phần tử đầu tiên KHÔNG phải là một list, ta tự bọc nó lại thành List của List.

    if field_groups and not isinstance(field_groups[0], list):
        field_groups = [field_groups]

    df = pd.read_csv(csv_legal_corpus, keep_default_na=False, low_memory=False)

    # 1. Tìm cụm số của từng group # Ví dụ:    {[1, 2],[1, 2, 3]}      
    group_numbers = [get_group_numbers_csvtable(df, fields) for fields in field_groups]
   
    # 2. Tạo tất cả tổ hợp giữa các group # kết quả: (1, 1)  (1, 2) (1, 3) (2, 1) (2, 2)  (2, 3)
    group_combinations = list(itertools.product(*group_numbers))
    print( f"Kiểm tra group_combinations:{group_combinations}" )

    if not group_combinations:
      print(  f"⚠️ CẢNH BÁO: Không tìm thấy cột nào có đuôi số khớp với {field_groups}!"     )
      return df
    # 3. Tìm các cột không thuộc bất kỳ group nào
    col_not_in_groups = get_col_not_in_groups_csvtable(df, field_groups)
    

    # 4. Danh sách chứa các dataframe con sau khi tách theo tổ hợp
    dfs_to_concat = []
    i = 0
    for combination in group_combinations:
        # 1. Chọn ra các cột cần giữ lại cho tổ hợp này
        cols_to_select = list(col_not_in_groups)
        rename_dict = {}
        for fields, group_n in zip(field_groups, combination):
            for field in fields:
                old_col = f"{field}{group_n}"
                if i==0: print(f"Kiểm tra tên cột:{old_col}")
                if old_col in df.columns:
                    cols_to_select.append(old_col)
                    rename_dict[old_col] = field # Đổi tên về tên gốc (nếu cần)
                i = i+1       
        # 2. Cắt DataFrame theo các cột đã chọn và đổi tên cột
        sub_df = df[cols_to_select].rename(columns=rename_dict)
        dfs_to_concat.append(sub_df)

    # 5. Gộp tất cả lại thành DataFrame mới chạy chỉ mất vài mili giây
    final_df = pd.concat(dfs_to_concat, ignore_index=True)
    
    # Xóa ngay danh sách các bảng con để Python thu hồi RAM lập tức
    del dfs_to_concat 
    import gc
    gc.collect() # Lệnh ép hệ thống dọn rác RAM

    final_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"-> Đã lưu file CSV unpivot tại: {output_csv_path}")
    
    # =================================================================
    # BƯỚC 6: TẠO JSON QUAN HỆ SỬA ĐỔI (THEO CHUNK_ID)
    # =================================================================
    # 1. Tự động lấy cột từ field_groups và tạo nhãn tương ứng
    edit_cols = [f for grp in field_groups for f in grp if f in final_df.columns]
    labels = ["Văn bản: "   if "law" in c.lower()
                            else ("Điều " if "art" in c.lower()
                                            else ("Khoản " if "clau" in c.lower()
                                                else ("Điểm " if "point" in c.lower() else f"{c}: "))
                            )
                            for c in edit_cols
    ]

    # 2. Tạo cột Phụ "_target_path"...Ghép chuỗi bằng zip (Chạy thẳng bằng C, dòng không sửa sẽ tự ra chuỗi rỗng "")
    final_df["_target_path"] = [" > ".join( f"{lbl}{clean_val}" 
                                            for lbl, v in zip(labels, vals) #GHÉP CẶP NGANG (Dán nhãn vào chữ)
                                            if (clean_val := str(v).strip()) and clean_val.lower() not in ["", "nan", "none"]
                                            )
                                for vals in zip(*(final_df[c] for c in edit_cols)) #QUÉT DỌC (Duyệt từng dòng)
    ]

    # 3. Lọc dòng có chữ và gom nhóm thành Dict đúng 1 dòng lệnh (khử trùng lặp luôn)
    valid_paths = final_df[final_df["_target_path"] != ""]
    edit_map    = (valid_paths.groupby("chunk_id", sort=False)["_target_path"]
                  .agg(lambda s: list(dict.fromkeys(s)))
                  .to_dict()
    )
    """{Ví dụ:  "96_2015_3_1": [  # <-- Đây là chunk_id
                    "Văn bản 78 > Điều 2",  # <-- Mớ này là do s sinh ra
                    "Văn bản 78 > Điều 5",
    ]"""

    final_df.drop(columns=["_target_path"], inplace=True)  # Xóa cột tạm


    # 6.5 Xuất ra file JSON để lưu trữ / kiểm tra bằng mắt
    import json

    with open(output_edit_json, "w", encoding="utf-8") as f:
      json.dump(edit_map, f, ensure_ascii=False, indent=4)

    print(f" Đã trích xuất {len(edit_map)} chunks có quan hệ sửa đổi/bổ sung!")
    print(f" File JSON sửa đổi lưu tại: {output_edit_json}")

    return edit_map  # Trả về Dictionary để bơm ngay vào Base JSON
    
   

#===============================================================
# 3
#===============================================================
def parse_date(date_val):
  """Chuyển DD/MM/YYYY sang YYYY-MM-DD."""
  s = str(date_val).strip()
  if not s or s.lower() in ["nan", "none", "-", "còn hiệu lực"]:
    return None
  try:
    return datetime.strptime(s, "%d/%m/%Y").strftime("%Y-%m-%d")
  except ValueError:
    return s


# Lấy ngày hôm nay theo chuẩn YYYY-MM-DD
today = date.today().strftime("%Y-%m-%d")
def get_status_by_date(eff_date, exp_date):
  """Tự động xác định trạng thái hiệu lực theo ngày hôm nay."""
  # 1. Chưa tới ngày có hiệu lực
  if eff_date and eff_date > today:
    return "chưa có hiệu lực"

  # 2. Đã qua ngày hết hiệu lực
  if exp_date and exp_date <= today:
    return "hết hiệu lực"

  # 3. Còn lại: exp_date để trống (None) HOẶC chưa tới ngày hết hạn
  return "còn hiệu lực"

def convert_metadata_csv_to_json(
    csv_path=config.EFFECTIVE_DATE_CSV,
    output_json_path=config.METADATA_JSON,
    original_cols=None,  # Tùy chọn: truyền [cột_luật, cột_hiệu_lực, cột_hết_hạn]
    amending_cols=None,  # Tùy chọn: truyền [cột_luật, cột_hiệu_lực, cột_hết_hạn]
):
  """ return
  "14/2008/qh12": {
        "doc_type": "gốc",
        "effective_date": "2009-01-01",
        "expire_date": "2025-01-10",
        "status": "hết hiệu lực",
        "amending_laws": [
            "32/2013/qh13",
            "71/2014/qh13"
        ],
        "original_laws": []
    },
  """
  df = pd.read_csv(csv_path, keep_default_na=False, low_memory=False)
  metadata = {}

  # =================================================================
  # ⭐️ TỰ ĐỘNG LẤY TÊN CỘT (KHÔNG CÒN HARDCODE)
  # =================================================================
  # Nếu không truyền tên cột, tự động lấy 3 cột đầu và 3 cột sau
  c_orig_law, c_orig_eff, c_orig_exp = (original_cols if original_cols else df.columns[0:3] )
  c_amend_law, c_amend_eff, c_amend_exp = (amending_cols if amending_cols else df.columns[3:6] )

  # =================================================================
  # 1. GOM NHÓM VĂN BẢN GỐC (Dùng biến cột động)
  # =================================================================
  mask_orig = ( (df[c_orig_law].astype(str).str.strip() != "")
                & (~df[c_orig_law].astype(str).str.strip().isin(["-", "nan"]))
  )
  df_orig = df[mask_orig]

  for orig_law, grp in df_orig.groupby(c_orig_law, sort=False):
    orig_law = str(orig_law).strip()
    first_row = grp.iloc[0]

    eff_date = parse_date(first_row.get(c_orig_eff, ""))
    exp_date = parse_date(first_row.get(c_orig_exp, ""))

    status = get_status_by_date(eff_date, exp_date)

    # Gom toàn bộ các luật sửa đổi con (loại bỏ rỗng, dấu gạch và khử trùng lặp)
    raw_amending = [
        str(x).strip()
        for x in grp[c_amend_law]
        if str(x).strip() and str(x).strip() not in ["-", "nan", "none"]
    ]
    amending_laws = list(dict.fromkeys(raw_amending))

    metadata[orig_law] = {
        "doc_type": "gốc",
        "effective_date": eff_date,
        "expire_date": exp_date,
        "status": status,
        "amending_laws": amending_laws,
        "original_laws": [],
    }

  # =================================================================
  # 2. GOM NHÓM VĂN BẢN SỬA ĐỔI (Dùng biến cột động)
  # =================================================================
  mask_amend = ((df[c_amend_law].astype(str).str.strip() != "")
                & (~df[c_amend_law].astype(str).str.strip().isin(["-", "nan"]))
  )
  df_amend = df[mask_amend]

  for amend_law, grp in df_amend.groupby(c_amend_law, sort=False):
    amend_law = str(amend_law).strip()
    first_row = grp.iloc[0]

    eff_date = parse_date(first_row.get(c_amend_eff, ""))
    exp_date = parse_date(first_row.get(c_amend_exp, ""))
    status = "hết hiệu lực" if exp_date else "còn hiệu lực"

    # Gom toàn bộ các luật gốc mà văn bản này sửa đổi
    raw_orig = [
        str(x).strip()
        for x in grp[c_orig_law]
        if str(x).strip() and str(x).strip() not in ["-", "nan", "none"]
    ]
    original_laws = list(dict.fromkeys(raw_orig))

    metadata[amend_law] = {
        "doc_type": "sửa đổi",
        "effective_date": eff_date,
        "expire_date": exp_date,
        "status": status,
        "amending_laws": [],
        "original_laws": original_laws,
    }

  # =================================================================
  # 3. XUẤT FILE JSON
  # =================================================================
  with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=4)

  print(f"--- CHUYỂN ĐỔI METADATA THÀNH CÔNG ---")
  print(f"Tổng số văn bản đã trích xuất: {len(metadata)}")
  print(f"File JSON lưu tại: {output_json_path}")

  return metadata



#===============================================================
# 4
#===============================================================
def merge_corpus_data(
    base_json_path,
    merge_json_path,
    output_final_json_path,
    output_final_jsonl_path,
    match_key="law_no",  # 'law_no' HOẶC 'chunk_id'
    target_field=None,  # Gán tên trường nếu dữ liệu là List (ví dụ: 'edited_targets')
    update_search_text=False,  # Bơm vào search_text hay không (True/False)
):
  print(f"--- BẮT ĐẦU GHÉP DỮ LIỆU TỪ: {merge_json_path} ---")

  # 1. Đọc file Base JSON
  with open(base_json_path, "r", encoding="utf-8") as f:
    corpus_data = json.load(f)
  records = (corpus_data if isinstance(corpus_data, list) else corpus_data.values() )

  # 2. Đọc file Merge JSON
  with open(merge_json_path, "r", encoding="utf-8") as f:
    merge_map = json.load(f)

  # ⭐️ ÉP TOÀN BỘ KEY VỀ CHỮ THƯỜNG & GỌT KHOẢNG TRẮNG ĐỂ KHÔNG BỊ LỆCH
  clean_merge_map = {str(k).strip().lower(): v for k, v in merge_map.items() if k }
  print(f"-> Đã nạp {len(clean_merge_map)} keys từ file cần ghép.")

  matched_count = 0

  # 3. Duyệt và ghép dữ liệu
  for doc in records:
    raw_k = str(doc.get(match_key, "")).strip().lower()
    val = clean_merge_map.get(raw_k)

    targets = []

    # TRƯỜNG HỢP 1: Dữ liệu là Dict (ví dụ: file ngày hiệu lực)
    if isinstance(val, dict):
      doc.update(val)
      matched_count += 1

    # TRƯỜNG HỢP 2: Dữ liệu là List (ví dụ: file edit_map sửa đổi)
    elif target_field:
      targets = val if val else []
      doc[target_field] = targets
      if val:
        matched_count += 1

    # Cập nhật search_text nếu được yêu cầu (Context Injection)
    if update_search_text and targets:
      targets_text = "\n".join([f"- {t}" for t in targets])
      doc["search_text"] = (
          f"{doc.get('hierarchy_context', '')}\n"
          f"[SỬA ĐỔI, BỔ SUNG, BÃI BỎ CHO:\n{targets_text}]\n"
          f"Nội dung chi tiết:\n{doc.get('content', '')}"
      )

  # 4. Xuất file kết quả
  with open(output_final_json_path, "w", encoding="utf-8") as f_json:
    json.dump(records, f_json, ensure_ascii=False, indent=4)

  with open(output_final_jsonl_path, "w", encoding="utf-8") as f_jsonl:
    for doc in records:
      f_jsonl.write(json.dumps(doc, ensure_ascii=False) + "\n")

  print("\n" + "=" * 55)
  print(f"🎉 KẾT QUẢ GHÉP NỐI:")
  print(f"Tổng số tài liệu Corpus: {len(records)}")
  print(f"👉 ĐÃ GHÉP THÀNH CÔNG VÀO: {matched_count} / {len(records)} văn bản!")
  if matched_count == 0:
    print(f"❌ CẢNH BÁO: Khớp 0 dòng! Hãy kiểm tra xem match_key '{match_key}' có tồn tại trong doc không!" )
  print(f"📁 Đã lưu file tại: {output_final_json_path}")
  print("=" * 55)



#===============================================================
# 5
#===============================================================
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


