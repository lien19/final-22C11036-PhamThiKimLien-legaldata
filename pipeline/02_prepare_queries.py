import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.query_processor import *

if __name__ == "__main__":
    print("🚀 Đang chạy Bước 2: Chuẩn bị bộ câu hỏi...")

    # export_and_split_queries_for_mining(
    #     input_csv_path="data/3-query_check_5.csv",
    #     output_csv_with_id="data/3-query_check_5_with_id_5digit.csv",
    #     output_train_jsonl="data/3-train_queries_for_mining.jsonl",
    #     output_eval_jsonl="data/3-eval_queries.jsonl",
    #     eval_ratio=0.2,
    # )

    export_and_split_queries_for_mining(
        input_csv_path="data/queries_with_tax_type_for_negative.xlsx",
        output_csv_with_id="data/3-query_check_5_with_id_5digit.csv",
        output_train_jsonl="data/3-train_queries_for_mining.jsonl",
        output_eval_jsonl="data/3-eval_queries.jsonl",
        eval_ratio=0.2,
    )

    print("🎉 Hoàn thành Bước 2!")