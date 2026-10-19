import json
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

FILES_TO_CLEAN = [  config.CORPUS_FINAL_JSON,
                    config.CORPUS_MAP_FINAL_JSON,
]


def clean_text_duplicates(text, separator):
  """Hàm lọc bỏ các đoạn văn bản bị lặp lại liên tiếp."""
  if not text or not isinstance(text, str):
    return text
  parts = [p.strip() for p in text.split(separator) if p.strip()]
  # Khử trùng lặp nhưng giữ nguyên thứ tự xuất hiện
  unique_parts = list(dict.fromkeys(parts))
  return separator.join(unique_parts)


def clean_corpus():
  for file_path in FILES_TO_CLEAN:
    if not os.path.exists(file_path):
      continue

    print(f"-> Đang làm sạch trùng lặp trong: {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
      data = json.load(f)

    # 1. Nếu là file 3-final_legal_corpus.json (dạng List)
    if isinstance(data, list):
      for doc in data:
        # Làm sạch content: Khử các đoạn văn \n hoặc \n\n bị nhân đôi
        if "content" in doc:
          doc["content"] = clean_text_duplicates(doc["content"], "\n\n")
          doc["content"] = clean_text_duplicates(doc["content"], "\n")

        # Làm sạch hierarchy_context: Khử các cụm lặp nối bằng dấu ' - '
        if "hierarchy_context" in doc:
          # Tách theo từng cấp > trước
          levels = doc["hierarchy_context"].split(" > ")
          cleaned_levels = []
          for lvl in levels:
            cleaned_levels.append(clean_text_duplicates(lvl, " - "))
          doc["hierarchy_context"] = " > ".join(cleaned_levels)

        # Cập nhật lại search_text cho đồng bộ
        if "search_text" in doc:
          doc["search_text"] = (
              f"{doc.get('hierarchy_context', '')}\nNội dung chi"
              f" tiết:\n{doc.get('content', '')}"
          )

    # 2. Nếu là file legal_expanded_content_map.json (dạng Dict {chunk_id: text})
    elif isinstance(data, dict):
      for k, v in data.items():
        if isinstance(v, str):
          cleaned = clean_text_duplicates(v, "\n\n")
          data[k] = clean_text_duplicates(cleaned, "\n")

    # Lưu lại file đã được lọc sạch
    with open(file_path, "w", encoding="utf-8") as f:
      json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"   (Đã làm sạch xong {file_path})")

  print("\n🎉 HOÀN TẤT! Dữ liệu đã sạch bóng, không còn một chữ nào bị lặp lại!")


if __name__ == "__main__":
  clean_corpus()