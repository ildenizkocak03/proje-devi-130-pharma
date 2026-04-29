# 💊 Pharma-Guard AI System

**Pharma-Guard AI**, yapay zeka ve RAG (Retrieval-Augmented Generation) teknolojilerini kullanarak ilaç prospektüslerini ve kutularını analiz eden akıllı bir denetim sistemidir. Sistem, kullanıcıların ilaç fotoğraflarını yükleyerek veya isimlerini yazarak detaylı tıbbi bilgi (etken madde, dozaj, kullanım alanları, yan etkiler vb.) almasını sağlar ve bunları profesyonel bir PDF raporu olarak sunar.

## ✨ Özellikler

- 📸 **Görsel Analiz (Vision):** İlaç kutusunun fotoğrafını yükleyerek ilacı otomatik tanıma ve detaylarını analiz etme.
- 💬 **Metin Tabanlı Analiz:** İlaç ismini ve dozajını girerek detaylı bilgi edinebilme.
- 📚 **RAG (Retrieval-Augmented Generation):** Yüklenen prospektüs PDF'leri üzerinden yerel veritabanında arama yapma ve yüksek doğrulukla yanıt verme.
- 🌐 **İnternet Taraması (Fallback):** Yerel veritabanında bulunmayan ilaçlar için internetteki resmi kaynaklardan veriyi çekme.
- 🤖 **Yedekli Yapay Zeka (Fallback Models):** API limitlerine veya hatalarına karşı çoklu model stratejisi (Gemini 2.0 Flash, Gemini Flash Latest, Gemini Pro Latest) ile %100'e yakın çalışma süresi (uptime) sağlama.
- 📄 **PDF Raporlama:** Analiz sonuçlarını özel font desteği ile (Roboto) Türkçe karakter sorunu olmadan PDF formatında indirme.
- 🎨 **Premium Modern Arayüz:** Streamlit tabanlı, kullanıcı dostu ve şık arayüz tasarımı.

## 🛠️ Kurulum ve Çalıştırma

### Gereksinimler

Proje Python 3.9+ gerektirir. Bağımlılıkları yüklemek için:

```bash
pip install -r requirements.txt
```

### Çevresel Değişkenler (.env)

Proje dizininde bir `.env` dosyası oluşturun (veya arayüzden ayarlardan girin) ve aşağıdaki anahtarları doldurun:

```env
GOOGLE_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
```

### Uygulamayı Başlatma

Yerel makinenizde uygulamayı çalıştırmak için:

```bash
streamlit run app.py
```

## 🚀 Canlıya Alma (Streamlit Cloud)

1. Projeyi GitHub'a yükleyin.
2. [Streamlit Cloud](https://streamlit.io/cloud) adresine gidin ve yeni bir uygulama (New App) oluşturun.
3. Repository ve `app.py` yolunu seçin.
4. **Advanced Settings** kısmından `Secrets` paneline aşağıdaki formatta anahtarlarınızı ekleyin:
   ```toml
   GOOGLE_API_KEY = "key"
   GROQ_API_KEY = "key"
   CHROMA_PATH = "data/chroma"
   CORPUS_PATH = "data/corpus"
   ```
5. **Deploy** butonuna basarak uygulamayı canlıya alın.

## 📁 Dosya Yapısı

- `app.py`: Ana Streamlit arayüzü ve uygulama mantığı.
- `agents.py`: LangChain ve Gemini entegrasyonlarını barındıran yapay zeka ajanları.
- `utils.py`: PDF okuma, vektör veritabanı kurulumu (Chroma) ve PDF raporu oluşturma (FPDF2) fonksiyonları.
- `requirements.txt`: Python bağımlılıkları.
- `data/`: RAG veritabanı ve yardımcı dosyalar (Fontlar) için kullanılan klasör.
