# Değişiklik ve Silme Gerekçeleri

| Eski parça / yaklaşım | Yeni karşılığı | Gerekçe |
|---|---|---|
| İki dosyadaki tekrar eden YOLO/OBB/crop/OCR/JSON fonksiyonları | `common.py` | Bakım tekrarı ve farklı sürümlerden kaynaklanan hataları azaltır |
| `generate_ocr_variants` ve tek seçenekli "best variant" döngüsü | `prepare_ocr_image` + `run_ocr_batch` | Tek seçenek (`sharpen`) bulunduğu için gereksiz seçim döngüsü |
| Her OCR işlemi için geçici JPEG yazma-okuma-silme | Doğrudan NumPy + OCR batch | Tekrarlanan disk I/O ve JPEG yeniden kodlama giderilir |
| Bütün OCR çıktısını recursive gezip JSON'a dönüştürme | `rec_texts`/`rec_scores` seçici okuma | Sadece gereken alanlar işlenir |
| Her görüntü karşılaştırmasında 4 kez yeniden `ORB_create`, histogram ve hash çıkarma | `ImageFeatureExtractor` + `features` önbelleği | Referans/karşılaştırma özelliği bir kez hesaplanır |
| Kareye bozan `cv2.resize(256,256)` | Aspect ratio koruyan resize+padding | İnce kitap yazı/geometrisi bozulmaz |
| Kenarı olmayan iki görüntüye yapay 1,0 kenar skoru | Toleranslı kenar örtüşmesi | Yanlış güçlü görsel benzerlikleri azaltır |
| `lower()` sonrasında `İ` kaybı | Unicode `NFKD` + `casefold()` | Türkçe karakter eşleştirmesi düzelir |
| Greedy eşleştirme | Hungarian global matching + dummy eşleşmeme seçeneği | Daha yüksek toplam eşleşme, daha az rastgele erken atama |
| Toplam/fark ile köşe sıralama | Çevresel dört köşe sıralama | 45° döndürülmüş OBB'de aynı köşenin iki kez seçilmesini önler |
| `ölçüm.py` yeniden çalıştığında JSON'u koruyup crop'ları silme | Versioned reference + `--force` | Referans görseli/metni aynı ölçümden gelir |
| Belirsiz eşleşmelere yalnız `GÜÇLÜ/ORTA` etiketi | `ambiguous` + `ambiguity_margin` | Birbirine çok yakın kitap adayları görülebilir |
| PaddleOCR v5 otomatik/fallback seçim | Açık v6 small det + rec, başarısızsa hata | Yanlış modele sessiz geçiş olmaz |
| Önceki YOLOv8 ağırlık adı | YOLO26s-OBB desteği (`--model best.pt`) | İstenen ağırlık ve OBB task doğrulaması |

**Korunanlar:** YOLO OBB tespit, perspektif düzeltme, sharpen OCR, 0/90/180/270 görüntü karşılaştırma, 4 görsel algoritma ve metin benzerliği, JSON referansı, TXT/JSON/annotated JPG raporları, eşleşen/eşleşmeyen/yer değiştiren kitaplar, ayrıntılı tüm çiftlerin skorları.

**Model ağırlığı**, eğitimli `best.pt` dosyası bu pakete dahil edilmemiştir. Performans ve doğruluk gerçek verilerde test edilmeden kesin olarak vaat edilmez.

## 2026-10-15: Seçici 0/1 arka plan gürültüsü düzeltmesi

- `noise.py` yeni modülü eklendi. `0/1` token yoğunluğu konservatif eşikle tespit ediliyor; `1984` gibi gerçek sayısal kitap adlarına dokunulmuyor.
- `common.py`: İlk OCR bütün kitaplarda, ikinci OCR yalnız gürültülü olanlarda; yeni Otsu varyantı başarısızsa ilk metin korunuyor.
- `text_raw`, `text_filtered`, `match_text`, `noise_detected`, `ocr_retry_attempted`, `ocr_retry_used` ve ikinci deneme bilgileri JSON'da tutuluyor.
- `features.py`: `normalized_text` için temizlenmiş `match_text` kullanılıyor. Eski referans JSON'ları geriye dönük olarak okunup eşleştirme sırasında filtrelenebiliyor.
- `ölçüm.py` ve `search.py`: `--ocr-denoise auto|off`, `--ocr-debug-crops` komut satırı seçenekleri eklendi.
- `test_raf.py`: tekrarlama, geri dönüş, eski referans uyumu ve kitap başlığında sayı koruma testleri eklendi.
- **Sınır:** Gerçek PP-OCRv6 ile ikinci OCR'nin doğruluğu ölçülmedi. Eski 100 eşleştirme adayı üzerinden tekrar hesap yapıldı: 10/10 eşleşme, önceki 9/10 yerine.
