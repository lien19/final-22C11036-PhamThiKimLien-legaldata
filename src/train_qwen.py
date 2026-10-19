import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import os
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
import torch
from transformers import BitsAndBytesConfig

from transformers.trainer_utils import get_last_checkpoint
# ⭐️ IMPORT LỚP TỐI ƯU CHO EMBEDDING CỦA UNSLOTH
from unsloth import FastSentenceTransformer
import config

# =====================================================================
# 1. HÀM NẠP DATASET
# =====================================================================
def prepare_dataset_from_mined_jsonl():
  print(f"-> Đang tải dữ liệu từ {config.MINED_TRAIN_JSONL}...")
  triples = []
  query_prefix = ( "Instruct: Truy xuất điều khoản pháp luật trả lời câu hỏi sau\nQuery: "  )

  with open(config.MINED_TRAIN_JSONL, "r", encoding="utf-8") as f:
    # ⭐️ ĐỌC THỬ DÒNG ĐẦU TIÊN ĐỂ "BẮT BỆNH"
    first_line = f.readline()
    if not first_line:
        raise ValueError(f"❌ LỖI: File {config.MINED_TRAIN_JSONL} bị rỗng!")
    
    sample = json.loads(first_line)
    
    # 1. KIỂM TRA CHÌA KHÓA (KEYS)
    if "pos" not in sample or "neg" not in sample:
        raise KeyError(
            f"\n❌ BÁO ĐỘNG: BẠN ĐANG LẤY NHẦM FILE ĐẦU VÀO CỦA KHÂU MINING!"
            f"\nFile này chỉ có keys: {list(sample.keys())}."
            f"\n👉 Hãy kiểm tra lại: Bạn cần truyền file kết quả SAU KHI ĐÀO (có 'pos' và 'neg')."
        )
        
    # 2. KIỂM TRA NỘI DUNG (Tránh trường hợp 'pos' chỉ chứa mỗi cái mã ID cụt lủn)
    first_pos_text = sample["pos"][0]
    if len(first_pos_text) < 30 or " " not in first_pos_text:
        raise ValueError(
            f"\n❌ BÁO ĐỘNG: Trường 'pos' dường như chỉ là MÃ ID ({first_pos_text}), không phải văn bản luật!"
            f"\n👉 Hãy kiểm tra lại khâu Mining xem đã bốc trường 'search_text' vào chưa."
        )
        
    # Đọc lại từ đầu file sau khi kiểm tra xong
    f.seek(0)
    
    # ... chạy tiếp vòng for nạp data bình thường ...
    for line in f:
      if not line.strip():
        continue
      data = json.loads(line)
      query = data["query"].strip()
      pos = data["pos"][0].strip()
      negs = data.get("neg", [])

      formatted_query = f"{query_prefix}{query}"
      for neg in negs:
        if neg.strip():
          triples.append({
              "anchor": formatted_query,
              "positive": pos,
              "negative": neg.strip(),
          })

  print(f"-> Đã tạo thành công {len(triples)} mẫu huấn luyện!")
  return Dataset.from_list(triples)


# =====================================================================
# 2. KHỞI TẠO VÀ TRAIN QWEN 8B QUANTIZE (QLoRA)
# =====================================================================
def train_qwen_8b_quantized():
    # Tên mô hình 8B (Đổi thành tên model cụ thể bạn dùng)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"-> Thiết bị huấn luyện: {device.upper()}")

    # ⭐️ CẤU HÌNH LƯỢNG TỬ HÓA 4-BIT (Chống nổ VRAM)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16),
    )

    print(f"-> Đang nạp mô hình 8B dạng 4-bit...")
    model = SentenceTransformer(
        config.BASE_MODEL_ID,
        trust_remote_code=True,
        model_kwargs={
            "quantization_config": bnb_config,
            "device_map": "auto",
        },
    )

    model.max_seq_length = 1024

    # ⭐️ GẮN LORA ADAPTER VÀO MÔ HÌNH NÉN 4-BIT
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )

    # Chuẩn hóa mô hình 4-bit để cho phép nhận gradient
    model[0].auto_model = prepare_model_for_kbit_training(model[0].auto_model)
    peft_model          = get_peft_model(model[0].auto_model, peft_config)
    model[0].auto_model = peft_model  # ⭐️ Nhớ gán lại vào auto_model

    # In thống kê an toàn:
    if hasattr(peft_model, "print_trainable_parameters"):
      print("\n📊 THỐNG KÊ THAM SỐ HUẤN LUYỆN:")
      peft_model.print_trainable_parameters()


    # Nạp dataset mining
    train_dataset = prepare_dataset_from_mined_jsonl()

    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Cấu hình huấn luyện
    training_args = SentenceTransformerTrainingArguments(
        output_dir=config.MODEL_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        warmup_ratio=0.1,
        gradient_checkpointing=False, # False để tránh lỗi TypeError: BaseModel.gradient_checkpointing_enable() got an unexpected keyword argument 'every_n_layers'
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,  # Cứ 10 bước in loss 1 lần
        # =============================================================
        # ⭐️ CẤU HÌNH LƯU LIÊN TỤC (CHỐNG MẤT ĐIỆN / RỚT MẠNG):
        # =============================================================
        save_strategy="steps",  # Đổi từ "epoch" sang "steps"
        save_steps=50,  # ⭐️ Cứ đúng 100 bước là tự động lưu 1 checkpoint
        save_total_limit=2,  # Chỉ giữ lại 2 bản lưu gần nhất (đỡ đầy ổ cứng)
        # =============================================================
        optim="paged_adamw_8bit",
        report_to="none",
    )
    

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=train_loss,
    )         

    print("\n🚀 BẮT ĐẦU HUẤN LUYỆN QWEN 8B QLORA...")
    trainer.train()
    # =================================================================
    # ⭐️ TỰ ĐỘNG PHÁT HIỆN BẢN LƯU (KHÔNG CẦN CHỈNH TAY NỮA)
    # =================================================================
    from transformers.trainer_utils import get_last_checkpoint
    import os

    # Tự động quét xem trong thư mục lưu đã có checkpoint nào chưa
    last_checkpoint = None
    if os.path.isdir(config.MODEL_DIR):
      last_checkpoint = get_last_checkpoint(config.MODEL_DIR)

    if last_checkpoint:
      print("\n" + "=" * 55)
      print(f"🔄 PHÁT HIỆN BẢN LƯU CŨ: {last_checkpoint}")
      print("👉 TỰ ĐỘNG TRAIN NỐI TIẾP TỪ CHỖ BỊ DỪNG!")
      print("=" * 55 + "\n")
      trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
      print("\n🚀 CHƯA CÓ BẢN LƯU CŨ -> BẮT ĐẦU HUẤN LUYỆN TỪ ĐẦU!\n")
      trainer.train()  

    # Lưu mô hình LoRA hoàn chỉnh
    print(f"\n💾 Đang lưu mô hình tại: {config.MODEL_DIR}...")
    model.save_pretrained(config.MODEL_DIR)
    print(f"--- HUẤN LUYỆN QWEN 8B HOÀN TẤT THÀNH CÔNG! ---")


# =====================================================================
# 2. KHỞI TẠO VÀ TRAIN QWEN 8B QUANTIZE (QLoRA) VỚI UNSLOTH
# =====================================================================
def train_qwen_8b_quantized_unsloth():        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"-> Thiết bị huấn luyện: {device.upper()}")

    print(f"-> Đang nạp mô hình 8B dạng 4-bit qua Unsloth...")

    # ⭐️ 1. NẠP MODEL EMBEDDING QUA UNSLOTH (Tự động lượng tử hóa 4-bit chuẩn nhất)
    model = FastSentenceTransformer.from_pretrained(
        model_name=config.BASE_MODEL_ID,
        max_seq_length=1024,
        load_in_4bit=True,  # 4-bit QLoRA
        load_in_16bit=False,  # ⭐️ THÊM DÒNG NÀY VÀO để tắt chế độ 16-bit mặc định
    )

    # ⭐️ 2. GẮN LORA ADAPTER BẰNG CƠ CHẾ TỐI ƯU CỦA UNSLOTH
    model = FastSentenceTransformer.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.0,  # ⭐️ Unsloth tối ưu tốc độ nhanh nhất khi lora_dropout = 0
        bias="none",
        task_type="FEATURE_EXTRACTION",
        # ⭐️ BẬT CHECKPOINTING CỦA UNSLOTH: Tiết kiệm thêm 30% VRAM mà KHÔNG bị lỗi every_n_layers!
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # In thống kê tham số:
    if hasattr(model, "print_trainable_parameters"):
        print("\n📊 THỐNG KÊ THAM SỐ HUẤN LUYỆN:")
        model.print_trainable_parameters()

    # Nạp dataset mining
    train_dataset = prepare_dataset_from_mined_jsonl()

    # Hàm Loss
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # ⭐️ 3. CẤU HÌNH THAM SỐ TRAINER (ĐÃ TỐI ƯU LẠI CHO RTX 3060 12GB)
    training_args = SentenceTransformerTrainingArguments(
        output_dir=config.MODEL_DIR,   # "qwen8b_legal_unsloth_lora"
        num_train_epochs=1.4,  # Bạn có thể cân nhắc để 1-2 epochs nếu muốn chạy nhanh hơn
        # VRAM giờ rất thoáng (chỉ ăn ~7GB), nâng batch_size lên 4 và giảm accumulation xuống 4
        # Effective batch size vẫn là 4 * 4 = 16 (y hệt như 2 * 8 = 16 cũ của bạn)
        # nhưng thanh tiến trình sẽ chạy mượt hơn rất nhiều!
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        warmup_ratio=0.1,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,  # Cứ 10 bước in loss 1 lần
        # =============================================================
        # CẤU HÌNH CHECKPOINT
        # =============================================================
        save_strategy="steps",
        save_steps=50,  # Cứ 100 bước lưu 1 checkpoint
        save_total_limit=2,  # Giữ tối đa 2 checkpoint gần nhất
        optim="paged_adamw_8bit",
        report_to="none",
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=train_loss,
    )

    # =================================================================
    # ⭐️ TỰ ĐỘNG PHÁT HIỆN BẢN LƯU VÀ BẮT ĐẦU HUẤN LUYỆN (ĐÃ FIX LỖI CHẠY 2 LẦN)
    # =================================================================
    last_checkpoint = None
    if os.path.isdir(config.MODEL_DIR):
        last_checkpoint = get_last_checkpoint(config.MODEL_DIR)

    if last_checkpoint:
        print("\n" + "=" * 55)
        print(f"🔄 PHÁT HIỆN BẢN LƯU CŨ: {last_checkpoint}")
        print("👉 TỰ ĐỘNG TRAIN NỐI TIẾP TỪ CHỖ BỊ DỪNG!")
        print("=" * 55 + "\n")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("\n🚀 BẮT ĐẦU HUẤN LUYỆN QWEN 8B VỚI UNSLOTH TỪ ĐẦU!\n")
        trainer.train()

    # ⭐️ Lưu mô hình LoRA hoàn chỉnh
    print(f"\n💾 Đang lưu mô hình tại: {config.MODEL_DIR}...")
    model.save_pretrained(config.MODEL_DIR)
    print(f"--- HUẤN LUYỆN QWEN 8B HOÀN TẤT THÀNH CÔNG! ---")


