document.addEventListener("DOMContentLoaded", async () => {
  const select = document.getElementById("question-select");
  const customInput = document.getElementById("custom-query");
  const form = document.getElementById("ask-form");
  const btn = document.getElementById("run-button");
  const btnLabel = btn.querySelector(".run-button__label");
  const btnSpinner = btn.querySelector(".run-button__spinner");
  const resultsSec = document.getElementById("results-section");
  const emptyState = document.getElementById("empty-state");

  // Các container của Tab 1
  const listBase = document.getElementById("list-base");
  const listTunedP = document.getElementById("list-tuned-p");
  const listRerank = document.getElementById("list-rerank");

  // Các container của Tab 2
  const listTunedH = document.getElementById("list-tuned-h");
  const listRRF = document.getElementById("list-rrf");
  const listDBSF = document.getElementById("list-dbsf");

  // 1. Chuyển đổi qua lại giữa 2 Tab
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabPanels = document.querySelectorAll(".tab-panel");

  tabBtns.forEach((tabBtn) => {
    tabBtn.addEventListener("click", () => {
      tabBtns.forEach((b) => b.classList.remove("tab-btn--active"));
      tabPanels.forEach((p) => (p.hidden = true));

      tabBtn.classList.add("tab-btn--active");
      const targetPanel = document.getElementById(tabBtn.dataset.tab);
      if (targetPanel) targetPanel.hidden = false;
    });
  });

  // 2. Tải danh sách câu hỏi mẫu vào Dropdown
  try {
    const res = await fetch("/api/questions");
    const questions = await res.json();
    select.innerHTML = '<option value="" disabled selected>-- Chọn câu hỏi có sẵn --</option>';
    questions.forEach((q) => {
      const opt = document.createElement("option");
      opt.value = q;
      opt.textContent = q;
      select.appendChild(opt);
    });
  } catch (err) {
    select.innerHTML = '<option value="" disabled selected>Lỗi tải danh sách</option>';
  }

  select.addEventListener("change", () => { customInput.value = ""; });
  customInput.addEventListener("input", () => { select.value = ""; });

  // 3. Hàm render Card kết quả
  function renderCard(item, rank, scoreLabel) {
    const statusHtml = item.status 
      ? `<span class="badge ${item.is_expired ? 'badge--expired' : 'badge--active'}">${item.status}</span>` 
      : "";

    return `
      <div class="result-card">
        <div class="result-card__header">
          <div class="result-card__badges">
            <span class="badge badge--law">Top ${rank} · ${item.badge}</span>
            <span class="badge badge--tax">${item.tax_type}</span>
            ${statusHtml}
          </div>
          <span class="result-card__score">${scoreLabel}: ${item.score}</span>
        </div>
        
        <div class="result-card__body">
          <p class="result-card__content">${item.content}</p>
          <div class="result-card__context">
            <strong>Ngữ cảnh:</strong> ${item.hierarchy}
          </div>
        </div>
      </div>
    `;
  }

  // 4. Xử lý tìm kiếm và đổ dữ liệu vào cả 2 Tab
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = customInput.value.trim() || select.value;
    if (!query) return;

    const queryBox = document.getElementById("query-display-box");
    queryBox.innerHTML = `<strong>❓ Câu hỏi tra cứu:</strong> ${query}`;

    btn.disabled = true;
    btnLabel.hidden = true;
    btnSpinner.hidden = false;
    emptyState.hidden = true;

    try {
      const res = await fetch("/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });
      const data = await res.json();

      if (data.error) {
        alert(data.error);
        return;
      }

      // Xóa dữ liệu cũ của tất cả các cột
      listBase.innerHTML = "";
      listTunedP.innerHTML = "";
      listRerank.innerHTML = "";
      listTunedH.innerHTML = "";
      listRRF.innerHTML = "";
      listDBSF.innerHTML = "";

      // -------------------------------------------------------------
      // ĐỔ DỮ LIỆU VÀO TAB 1: TIẾN TRÌNH
      // -------------------------------------------------------------
      if (data.base_error || !data.base || data.base.length === 0) {
        listBase.innerHTML = `
          <div class="error-notice">
            <p class="error-notice__title">⚠️ Không thể hiển thị Mô hình Gốc</p>
            <p class="error-notice__desc">${data.base_error || "Không thể tách lớp LoRA để chạy song song."}</p>
          </div>
        `;
      } else {
        data.base.forEach((item, idx) => {
          listBase.innerHTML += renderCard(item, idx + 1, "Cosine");
        });
      }

      // Cột Fine-tuned (Dense) ở Tab 1
      (data.tuned_dense || []).forEach((item, idx) => {
        listTunedP.innerHTML += renderCard(item, idx + 1, "Cosine");
      });

      // Cột Rerank ở Tab 1
      (data.rerank || []).forEach((item, idx) => {
        listRerank.innerHTML += renderCard(item, idx + 1, "Rerank Score");
      });

      // -------------------------------------------------------------
      // ĐỔ DỮ LIỆU VÀO TAB 2: KỸ THUẬT NÂNG CAO
      // -------------------------------------------------------------
      // Cột Dense thuần ở Tab 2
      (data.tuned_dense || []).forEach((item, idx) => {
        listTunedH.innerHTML += renderCard(item, idx + 1, "Cosine");
      });

      // Cột Hybrid RRF ở Tab 2
      (data.hybrid_rrf || []).forEach((item, idx) => {
        listRRF.innerHTML += renderCard(item, idx + 1, "RRF Score");
      });

      // Cột Hybrid DBSF ở Tab 2
      (data.hybrid_dbsf || []).forEach((item, idx) => {
        listDBSF.innerHTML += renderCard(item, idx + 1, "DBSF Score");
      });

      resultsSec.hidden = false;
    } catch (err) {
      alert("Lỗi kết nối máy chủ: " + err);
    } finally {
      btn.disabled = false;
      btnLabel.hidden = false;
      btnSpinner.hidden = true;
    }
  });
});