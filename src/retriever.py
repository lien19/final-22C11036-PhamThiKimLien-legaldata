import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date
from qdrant_client import models

def search_legal_documents(
    model, 
    qdrant_client, 
    query, 
    collection_name="vietnamese_legal_rag_1", 
    target_date=None, 
    tax_type="TNDN", 
    top_k=3
):
    """
    Hàm nghiệp vụ: Nhận câu hỏi -> Sinh vector -> Lọc hiệu lực -> Trả về kết quả đẹp.
    """
    # 1. Xác định ngày hiệu lực (mặc định hôm nay nếu không truyền)
    if not target_date:
        target_date = date.today().strftime("%Y-%m-%d")

    # 2. Sinh vector cho câu hỏi
    prompt_query = f"Instruct: Truy xuất điều khoản pháp luật trả lời câu hỏi sau\nQuery: {query}"
    query_vec = model.encode([prompt_query], normalize_embeddings=True)[0].tolist()

    # 3. Tạo bộ lọc hiệu lực thời gian & sắc thuế
    # query_filter = models.Filter(
    #     must=[
    #         models.FieldCondition(
    #             key="tax_type",
    #             match=models.MatchValue(value=tax_type)
    #         ),
    #         models.FieldCondition(
    #             key="effective_date",
    #             range=models.DatetimeRange(lte=target_date)
    #         )
    #     ],
    #     should=[
    #         models.FieldCondition(
    #             key="expire_date",
    #             range=models.DatetimeRange(gte=target_date)
    #         ),
    #         models.IsNullCondition(
    #             is_null=models.PayloadField(key="expire_date")
    #         )
    #     ],
    #     min_should_count=1
    # )

    # 3. Trong Qdrant: Bỏ filter ngày, chỉ lọc sắc thuế và lấy Top 10 ứng viên rộng hơn
    results = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vec,
        # query_filter=models.Filter(must=[models.FieldCondition(key="tax_type", match=models.MatchValue(value=tax_type))]  ),
        limit=10  # Lấy 10 kết quả để sàng lọc
    )

    # 4. Áp dụng trừ điểm cho văn bản hết hiệu lực
    candidates = []
    for p in results.points:
        payload = p.payload or {}
        raw_score = float(p.score)
        expire_date = payload.get("expire_date")
        status = payload.get("status", "còn hiệu lực")

        # Kiểm tra xem đã hết hiệu lực tại ngày target_date chưa
        is_expired = False
        if expire_date and expire_date <= target_date:
            is_expired = True
        elif status == "hết hiệu lực":
            is_expired = True

        # ⭐️ CÔNG THỨC PHẠT ĐIỂM:
        if is_expired:
            adjusted_score = raw_score * 0.75  # Giảm 25% số điểm
            status_label = "hết hiệu lực (đã bị trừ điểm)"
        else:
            adjusted_score = raw_score
            status_label = "còn hiệu lực"

        candidates.append({
            "badge": f"Điều {payload.get('article_id')} · {payload.get('law_no')}",
            "content": payload.get("content", ""),
            "score": round(adjusted_score, 3),
            "raw_score": round(raw_score, 3),
            "status": status_label,
            "is_expired": is_expired
        })

    # 3. Sắp xếp lại theo điểm mới sau khi trừ và lấy đúng Top 3
    candidates.sort(key=lambda x: x["score"], reverse=True)
    top_3_results = candidates[:top_k]

    return top_3_results