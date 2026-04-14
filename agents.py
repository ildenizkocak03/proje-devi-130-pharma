import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from utils import get_retriever
from langchain_community.tools import DuckDuckGoSearchRun
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
        
        # Search Tool: DuckDuckGo
        self.search_tool = DuckDuckGoSearchRun()

    def rag_specialist(self, medicine_name):
        """Prospektüs veritabanında arama yapar, yoksa internetten resmi veriyi çeker."""
        # 1. Yerel RAG Araması
        retriever = get_retriever()
        local_content = ""
        if retriever:
            docs = retriever.invoke(medicine_name)
            # Yerel verinin gerçekten ilgili olup olmadığını kontrol et
            # Basit bir anahtar kelime kontrolü: İlaç ismi dökümanda geçiyor mu?
            relevant_docs = []
            for doc in docs:
                content_lower = doc.page_content.lower()
                name_lower = medicine_name.lower().split()[0] # İlk kelimeyi (marka) baz al
                if name_lower in content_lower:
                    relevant_docs.append(doc.page_content)
            
            if relevant_docs:
                local_content = "\n".join(relevant_docs)
        
        if len(local_content.strip()) > 100:
            return f"[KAYNAK: YEREL PROSPEKTÜS]\n{local_content}"
        
        # 2. İnternet Araması (Fallback)
        print(f"Yerel veya uyumlu veri bulunamadı, {medicine_name} için internet taraması başlatılıyor...")
        # Aramayı daha spesifik hale getirelim
        search_query = f"{medicine_name} prospektüs kkt pdf resmi"
        try:
            web_content = self.search_tool.run(search_query)
            return f"[KAYNAK: RESMİ WEB VERİLERİ]\n{web_content}"
        except Exception as e:
            return f"Yerel veri yok ve internet araması başarısız oldu: {e}"

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
            ChatGoogleGenerativeAI(model="gemini-flash-latest", google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0), # Yedek 1
            ChatGoogleGenerativeAI(model="gemini-pro-latest", google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0)   # Yedek 2
        ]

        last_error = None
        for model in models_to_try:
            for attempt in range(3):
                try:
                    response = model.invoke(messages)
                    # response.content bazen list of dicts döndürür — düz metni çıkar
                    content = response.content
                    if isinstance(content, list):
                        parts = []
                        for block in content:
                            if isinstance(block, dict) and 'text' in block:
                                parts.append(block['text'])
                            elif isinstance(block, str):
                                parts.append(block)
                        content = "\n".join(parts)
                    return str(content)
                except Exception as e:
                    last_error = e
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        import time
                        time.sleep(2 * (attempt + 1))
                        continue
                    else:
                        break
            print(f"Sonraki yedek modele geçiliyor...")
        
        raise Exception(f"Tüm modeller başarısız oldu. Son hata: {last_error}")

def run_full_analysis(input_text, image_base64=None):
    agents = PharmaGuardAgents()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    
    if image_base64:
        # Görsel varsa doğrudan Gemini Vision'a gönder (RAG bypass)
        from langchain_core.messages import HumanMessage as HM
        
        vision_prompt = f"""Sen bir ilaç denetim uzmanısın. Bu ilaç kutusunu analiz et ve Türkçe profesyonel bir rapor hazırla:
1. İlaç adı ve etken madde
2. Dozaj (mg/ml)  
3. Kullanım alanları
4. Kritik güvenlik uyarıları ve yan etkiler
5. Genel değerlendirme

Ek kullanıcı notu: {input_text if input_text else 'Yok'}"""

        message = HM(content=[
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
            {"type": "text", "text": vision_prompt}
        ])

        # Yedek model listesi
        vision_models = [
            ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key, temperature=0),
            ChatGoogleGenerativeAI(model="gemini-flash-latest", google_api_key=google_api_key, temperature=0),
            ChatGoogleGenerativeAI(model="gemini-pro-latest", google_api_key=google_api_key, temperature=0),
        ]

        last_error = None
        for vision_model in vision_models:
            for attempt in range(3):
                try:
                    response = vision_model.invoke([message])
                    content = response.content
                    if isinstance(content, list):
                        parts = [block['text'] if isinstance(block, dict) and 'text' in block else str(block) for block in content]
                        content = "\n".join(parts)
                    return str(content)
                except Exception as e:
                    last_error = e
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        import time
                        time.sleep(2 * (attempt + 1))
                        continue
                    else:
                        break
        return f"Görsel analiz başarısız oldu. Son hata: {last_error}"

    else:
        # Metin girdisi: RAG + Orchestrator
        prospectus_data = agents.rag_specialist(input_text)
        final_report = agents.master_orchestrator(
            f"Kullanıcı Girdisi: {input_text}\n\nProspektüs Verisi:\n{prospectus_data}"
        )
        return final_report
