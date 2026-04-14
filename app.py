import os
import sys

# ── 1. ADIM: .env dosyasını yükle (yerel geliştirme için)
from dotenv import load_dotenv
load_dotenv()

# ── 2. ADIM: Streamlit Secrets'tan anahtarları al (canlı yayın için)
# Bu blok, diğer tüm importlardan ÖNCE çalışmalıdır.
try:
    import streamlit as st
    for key in ["GOOGLE_API_KEY", "GROQ_API_KEY", "CHROMA_PATH", "CORPUS_PATH"]:
        if key in st.secrets:
            os.environ[key] = st.secrets[key]
except Exception:
    pass

# ── 3. ADIM: Diğer importlar (anahtarlar artık os.environ'da hazır)
import base64
import time
import streamlit as st
from utils import setup_rag_database, generate_pdf_report
from agents import run_full_analysis

# Sayfa Konfigürasyonu
st.set_page_config(
    page_title="Pharma-Guard AI",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: white;
    }
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.5);
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }
    h1, h2, h3 {
        color: #60a5fa !important;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("💊 PHARMA-GUARD AI")
    st.subheader("Yapay Zeka Destekli Akıllı İlaç Denetçisi")

    # Sidebar: Ayarlar ve Veritabanı
    with st.sidebar:
        st.header("⚙️ Sistem Ayarları")
        api_key = st.text_input("Gemini API Key", type="password")
        groq_key = st.text_input("Groq API Key", type="password")
        
        if api_key and groq_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            os.environ["GROQ_API_KEY"] = groq_key
            st.success("API Anahtarları Tanımlandı!")

        st.divider()
        st.header("📚 Bilgi Kaynağı (RAG)")
        uploaded_pdfs = st.file_uploader("Prospektüs PDF'lerini Yükle", type="pdf", accept_multiple_files=True)
        
        if st.button("Veritabanını Güncelle"):
            if uploaded_pdfs:
                with st.spinner("Dosyalar işleniyor..."):
                    corpus_path = os.getenv("CORPUS_PATH", "data/corpus")
                    os.makedirs(corpus_path, exist_ok=True)
                    for pdf in uploaded_pdfs:
                        with open(os.path.join(corpus_path, pdf.name), "wb") as f:
                            f.write(pdf.getbuffer())
                    
                    setup_rag_database(corpus_path, os.getenv("CHROMA_PATH", "data/chroma"))
                    st.success("Hafıza Güncellendi!")
            else:
                st.warning("Lütfen önce PDF yükleyin.")

    # Ana Panel
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📸 İlaç Fotoğrafı veya İsim")
        input_mode = st.radio("Giriş Yöntemi", ["Fotoğraf Yükle", "Manuel İsim Yaz"])
        
        image_data = None
        user_input = ""
        
        if input_mode == "Fotoğraf Yükle":
            uploaded_img = st.file_uploader("İlaç kutusunun net bir fotoğrafını yükleyin", type=["jpg", "jpeg", "png"])
            if uploaded_img:
                st.image(uploaded_img, caption="Yüklenen Görsel", use_container_width=True)
                image_data = base64.b64encode(uploaded_img.read()).decode()
        else:
            user_input = st.text_input("İlaç ismini ve mg değerini girin (Örn: Parol 500mg)")

        if st.button("🔍 ANALİZİ BAŞLAT"):
            if user_input or image_data:
                process_analysis(user_input, image_data)
            else:
                st.error("Lütfen bir girdi sağlayın.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Analiz Raporu")
        if "final_report" in st.session_state:
            st.markdown(st.session_state.final_report)
            
            # PDF İndirme Butonu
            report_data = {
                "Analiz Özeti": st.session_state.final_report
            }
            if st.button("📄 Raporu PDF Olarak İndir"):
                pdf_path = generate_pdf_report(report_data, "pharma_report.pdf")
                with open(pdf_path, "rb") as f:
                    st.download_button("Dosyayı Kaydet", f, file_name="PharmaGuard_Rapor.pdf")
        else:
            st.info("Analiz sonuçları burada görünecektir.")
        st.markdown('</div>', unsafe_allow_html=True)

def process_analysis(user_input, image_data):
    with st.spinner("⏳ Analiz yürütülüyor, lütfen bekleyin..."):
        try:
            report = run_full_analysis(user_input, image_data)
            st.session_state.final_report = report
            st.toast("Analiz Tamamlandı!", icon="✅")
            st.rerun()
        except Exception as e:
            st.error(f"Sistem Hatası: {e}")

if __name__ == "__main__":
    main()
