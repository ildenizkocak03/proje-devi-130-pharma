import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from utils import get_retriever
from dotenv import load_dotenv

load_dotenv()

class PharmaGuardAgents:
    def __init__(self):
        # API anahtarlarını kontrol et
        google_api_key = os.getenv("GOOGLE_API_KEY")
        groq_api_key = os.getenv("GROQ_API_KEY")
        
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY bulunamadı! Lütfen .env dosyasını kontrol edin veya sidebar üzerinden anahtarı girin.")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY bulunamadı! Lütfen .env dosyasını kontrol edin veya sidebar üzerinden anahtarı girin.")

        # Master Orchestrator: Gemini 2.0 Flash
        self.gemini = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            google_api_key=google_api_key,
            temperature=0
        )
        
        # Fast Analysis & Synthesis: Groq Llama-3
        self.groq_llama = ChatGroq(
            model="llama3-70b-8192", 
            groq_api_key=groq_api_key,
            temperature=0
        )
        
        # Vision Scanner: Groq LLaVA
        self.groq_vision = ChatGroq(
            model="llava-v1.5-7b-4096-preview", 
            groq_api_key=groq_api_key,
            temperature=0
        )

    def vision_scanner(self, image_bytes):
        """İlaç kutusunu tarar ve metadata çıkarır."""
        # Note: Groq LLaVA takes base64 images usually via the vision API
        # For simplicity in this structure, we'll use a descriptive prompt
        prompt = """
        Analyze this medicine package image. 
        Extract: Brand Name, Active Ingredient, Dosage (mg/ml), Form (Tablet/Syrup), and Barcode if visible.
        Return ONLY a JSON object.
        """
        # This is a placeholder logic for multimodal input via LangChain
        # If image_bytes is provided, it should be sent as part of message content
        # For this implementation, we will assume the logic will be handled in app.py or here
        return {"Brand": "Unknown", "Dosage": "Unknown"} # Placeholder

    def rag_specialist(self, medicine_name):
        """Prospektüs veritabanında arama yapar."""
        retriever = get_retriever()
        if not retriever:
            return "Prospektüs veritabanı boş veya bulunamadı."
        
        docs = retriever.get_relevant_documents(medicine_name)
        return "\n".join([doc.page_content for doc in docs])

    def master_orchestrator(self, user_input, image_data=None):
        """Tüm süreci yöneten ana beyin. Kota hatası durumunda yedek modele geçer."""
        
        system_prompt = """
        ### ROLE: PHARMA-GUARD MASTER ORCHESTRATOR (PG-MO) ###
        Sen, Gemini tabanlı, multimodal yeteneklere sahip ve çoklu ajan ekosistemini yöneten baş mimarsın.
        Görevin; görsel veya metinsel girişi alınan bir ilacı, sıfır hata toleransı ile analiz etmektir.
        
        OPERASYONEL KURALLAR:
        1. Yazı okunmuyorsa asla tahmin etme!
        2. Bilgi kaynağın %100 tıbbi prospektüsler olmalı.
        3. Bilgiler arasında 1 mg fark olsa bile raporu blokla ve 'VERİ UYUŞMAZLIĞI' alarmı ver.
        
        ÇIKTI FORMATI:
        Tüm alt ajanlardan gelen veriyi birleştir ve Türkçe, profesyonel bir rapor hazırla.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Analiz edilecek girdi: {user_input}")
        ]

        # Retry ve Fallback Mantığı
        models_to_try = [
            self.gemini, # İlk deneme: Gemini 2.0 Flash
            ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0) # Yedek: 1.5 Flash
        ]

        last_error = None
        for model in models_to_try:
            for attempt in range(3): # Her model için 3 deneme
                try:
                    response = model.invoke(messages)
                    return response.content
                except Exception as e:
                    last_error = e
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        import time
                        time.sleep(2 * (attempt + 1)) # Üstel bekleme
                        continue
                    else:
                        break # Diğer hatalarda döngüden çık
            # Eğer buraya geldiyse bu model başarısız olmuştur, bir sonraki modele (yedek) geçilir.
            print(f"Model {model.model} başarısız oldu, yedek modele geçiliyor...")
        
        raise Exception(f"Tüm modeller kota veya sistem hatası verdi. Son hata: {last_error}")

def run_full_analysis(input_text, image_base64=None):
    agents = PharmaGuardAgents()
    
    # 1. Vision Analysis (if image provided)
    vision_results = {}
    if image_base64:
        # Implementation of vision call...
        pass

    # 2. RAG Retrieval
    prospectus_data = agents.rag_specialist(input_text)
    
    # 3. Final Synthesis via Gemini Orchestrator
    # We pass the collected evidence to Gemini
    final_report = agents.master_orchestrator(
        f"Kullanıcı Girdisi: {input_text}\n\nProspektüs Verisi:\n{prospectus_data}"
    )
    
    return final_report
