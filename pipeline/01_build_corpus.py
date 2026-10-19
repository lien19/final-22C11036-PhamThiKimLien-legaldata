import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.corpus_builder import *
import config


def build_corpus():
    print("=" * 65)
    print("🚀 BẮT ĐẦU CHẠY TRỌN VẸN PIPELINE XÂY DỰNG KHO LUẬT")
    print("=" * 65)

    # -----------------------------------------------------------------
    # CẤU HÌNH ĐƯỜNG DẪN DỮ LIỆU (Thêm 'data/' nếu file nằm trong folder data)
    # -----------------------------------------------------------------
    RAW_LEGAL_CSV       = "data/3-thong ke - luat.csv"
    EFFECTIVE_DATE_CSV  = "data/3-thong ke - luat-effectivedate.csv"

    # Các file trung gian sinh ra trong quá trình chạy
    CORPUS_BASE_JSON    = "data/3-thong ke - luat-json.json"
    CORPUS_BASE_JSONL   = "data/3-thong ke - luat-jsonl.jsonl"
    CORPUS_ID_CSV       = "data/3-thong ke - luat-id-uiud.csv"
    UNPIVOT_CSV         = "data/3-thong ke - luat-output.csv"
    EDIT_MAP_JSON       = "data/3-thong ke - luat-unpivoted_edit_map.json"
    METADATA_JSON       = "data/3-thong ke - luat-effectivedate-json.json"

    # File kho luật hoàn chỉnh cuối cùng
    CORPUS_FINAL_JSON   = "data/3-final_legal_corpus.json"
    CORPUS_FINAL_JSONL  = "data/3-final_legal_corpus.jsonl"

    # =================================================================
    # BƯỚC 1: XỬ LÝ NGỮ CẢNH GIA PHẢ & SINH MÃ CHUNK_ID, UUID
    # =================================================================
    print("\n👉 [BƯỚC 1/4] Đang lập sổ gia phả & làm phẳng văn bản...")
    universal_legal_corpus_processor()

    # =================================================================
    # BƯỚC 2: RẢI HÀNG (UNPIVOT) CÁC ĐIỀU KHOẢN SỬA ĐỔI, BỔ SUNG
    # =================================================================
    print("\n👉 [BƯỚC 2/4] Đang trích xuất quan hệ sửa đổi/bổ sung...")
    # Lấy đúng file CSV có chunk_id vừa sinh ra ở Bước 1 nạp vào:
    unpivot_group_data()

    # =================================================================
    # BƯỚC 3: CHUYỂN ĐỔI NGÀY HIỆU LỰC & TRẠNG THÁI VĂN BẢN
    # =================================================================
    print("\n👉 [BƯỚC 3/4] Đang chuyển đổi metadata ngày hiệu lực...")
    convert_metadata_csv_to_json()

    # =================================================================
    # BƯỚC 4: HỢP NHẤT TOÀN BỘ VÀO FILE CORPUS CUỐI CÙNG
    # =================================================================
    print("\n👉 [BƯỚC 4/4] Đang hợp nhất dữ liệu vào final_legal_corpus...")
    # 4.1 Bơm quan hệ sửa đổi vào Base JSON
    merge_corpus_data(
        base_json_path=config.CORPUS_BASE_JSON, # Lấy kết quả của lần 1
        merge_json_path=config.EDIT_MAP_JSON,   # File sửa đổi
        output_final_json_path=config.CORPUS_FINAL_JSON,   # File JSON đích cuối cùng
        output_final_jsonl_path=config.CORPUS_FINAL_JSONL, # File JSONL đích cuối cùng
        match_key="chunk_id", # Nối theo chunk_id
        target_field="edited_targets", # Nhét danh sách vào trường này
        update_search_text=True, # Bơm thẳng vào search_text
    )

    # 4.2 Bơm ngày hiệu lực vào file cuối
    merge_corpus_data(
        base_json_path=config.CORPUS_FINAL_JSON,
        merge_json_path=config.METADATA_JSON,
        output_final_json_path=config.CORPUS_FINAL_JSON,
        output_final_jsonl_path=config.CORPUS_FINAL_JSONL,
        match_key="law_no",
    )

    print("\n" + "=" * 65)
    print("🎉 CHÚC MỪNG! ĐÃ TẠO XONG KHO LUẬT HOÀN CHỈNH 1 LÈO!")
    print(f"📁 File kết quả cuối cùng: {CORPUS_FINAL_JSON}")
    print("=" * 65)









if __name__ == "__main__":
    print("🚀 Đang chạy Bước 1: Xây dựng kho luật...")
    build_corpus()
    print("🎉 Hoàn thành Bước 1!")