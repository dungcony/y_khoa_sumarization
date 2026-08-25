import streamlit as st
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(
    page_title="Hệ Thống Tóm Tắt Văn Bản Y Khoa",
    page_icon="🏥",
    layout="wide"
)


# Dùng chung mô hình giữa các lần Streamlit chạy lại
@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().parent / "outputs" / "best"

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    return tokenizer, model, device


st.title("🏥 Hệ Thống Tóm Tắt Văn Bản Y Khoa (ViT5)")
st.markdown("""
Ứng dụng này sử dụng mô hình **Transformer ViT5** đã được Fine-tune chuyên sâu trên bộ dữ liệu báo chí/bệnh án y khoa. 
Hãy dán một bài báo y khoa dài vào bên dưới để hệ thống trích xuất và tóm tắt những thông tin cốt lõi nhất.
""")

with st.spinner("Đang tải mô hình..."):
    tokenizer, model, device = load_model()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Nhập văn bản cần tóm tắt")
    input_text = st.text_area(
        "Văn bản gốc",
        height=400,
        placeholder="Dán nội dung bài báo, báo cáo y tế vào đây..."
    )

    with st.expander("⚙️ Cài đặt nâng cao"):
        max_length = st.slider("Độ dài tối đa của tóm tắt (tokens)", 50, 300, 160)
        num_beams = st.slider("Beam Search (Độ rẽ nhánh)", 1, 5, 4)

    submit_button = st.button("🚀 Tóm Tắt Ngay", type="primary", width="stretch")

with col2:
    st.subheader("Kết quả tóm tắt")
    if submit_button:
        if not input_text.strip():
            st.warning("Vui lòng nhập văn bản trước khi tóm tắt!")
        else:
            with st.spinner("Mô hình đang phân tích và sinh văn bản..."):
                source_text = "summarize: " + input_text.strip()
                inputs = tokenizer(
                    source_text,
                    return_tensors="pt",
                    max_length=768,
                    truncation=True,
                ).to(device)

                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True,
                    no_repeat_ngram_size=2,
                )

                summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

                st.success("Tóm tắt thành công!")
                st.info(summary)

                original_len = len(input_text.split())
                summary_len = len(summary.split())
                if original_len > 0:
                    compression_ratio = (1 - summary_len / original_len) * 100
                    st.caption(
                        f"📊 Độ dài giảm từ **{original_len} từ** xuống còn **{summary_len} từ** (Tỷ lệ nén: **{compression_ratio:.1f}%**).")
    else:
        st.info("Kết quả sẽ hiển thị ở đây sau khi bạn bấm nút.")

st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>Phát triển bởi Lường Tiến Dũng - B22DCCN128</div>",
            unsafe_allow_html=True)
