import os
from pypdf import PdfReader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from fpdf import FPDF
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

def extract_text_from_pdf(pdf_path):
    """PDF dosyasından metin ayıklar."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"PDF okuma hatası: {e}")
        return ""

def setup_rag_database(corpus_path, chroma_path):
    """PDF'leri tarar ve ChromaDB vektör veritabanını oluşturur."""
    api_key = os.getenv("GOOGLE_API_KEY")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    
    documents = []
    for file in os.listdir(corpus_path):
        if file.endswith(".pdf"):
            path = os.path.join(corpus_path, file)
            content = extract_text_from_pdf(path)
            if content:
                documents.append({"content": content, "metadata": {"source": file}})
    
    if not documents:
        return None

    texts = [doc["content"] for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]
    
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=chroma_path
    )
    return vectorstore

def get_retriever():
    """Vektör veritabanından bir retriever döndürür."""
    api_key = os.getenv("GOOGLE_API_KEY")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    chroma_path = os.getenv("CHROMA_PATH", "data/chroma")
    if not os.path.exists(chroma_path):
        return None
    
    vectorstore = Chroma(persist_directory=chroma_path, embedding_function=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

class PharmaReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'PHARMA-GUARD AI ANALIZ RAPORU', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Sayfa {self.page_no()}', 0, 0, 'C')

def generate_pdf_report(data, output_path):
    """Analiz verilerini profesyonel bir PDF raporuna dönüştürür."""
    pdf = PharmaReport()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for section, content in data.items():
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, section, 0, 1, 'L')
        pdf.set_font("Arial", '', 11)
        pdf.multi_cell(0, 10, content)
        pdf.ln(5)

    pdf.output(output_path)
    return output_path
