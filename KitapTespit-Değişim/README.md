# Kitap Rafı Ölçüm + Arama (Optimizasyonlu)

**Modeller:** YOLO26s-OBB (**kitap verinizle eğitilmiş `best.pt`**) + `PP-OCRv6_small_det` + `PP-OCRv6_small_rec` (Türkçe destekli).

## Dosyalar

| Dosya | Görev |
|---|---|
| `ölçüm.py` | İlk rafın referans JSON ve kitap kırpmalarını oluşturur |
| `search.py` | Yeni rafla eşleştirir; TXT, JSON, çizimli JPG ve crop kaydeder |
| `common.py` | Tek YOLO/OCR yükleme, köşe sırası, crop, toplu OCR, JSON işlemleri |
| `features.py` | Türkçe metin normalizasyonu; HSV, pHash, Canny ve ORB özellik önbelleği |
| `noise.py` | Gürültülü 0/1 OCR tokenlarını seçici silme, ikinci OCR için Otsu görüntüsü |
| `test_raf.py` | Ağ ve model indirme gerektirmeyen birim/regresyon testleri |
| `requirements.txt` | Python bağımlılıkları (PaddlePaddle inference engine hariç) |


## Seçici arka plan gürültüsü temizleme (yeni)

Varsayılan `--ocr-denoise auto` açıkken her kitabın OCR'si **bir kez** çalışır.
Yalnızca yoğun `0` ve `1` token gürültüsü tespit edilen kitapta:

1. İlk OCR metni `text_raw` alanında saklanır.
2. Uzun ikili sayı dizileri yalnızca **eşleştirme metninden** (`match_text`, `text_filtered`) kaldırılır. `1984`, `101 Dalmaçyalı` gibi sayılı kitap isimleri otomatik temizlenmez.
3. Sadece ilgili crop'un gri arka plan deseni Otsu eşikleme ile bastırılır ve ikinci OCR denenir.
4. Yeni sonuç ancak ayırt edici kelimeleri koruyor ve kalite puanı yeterince yükseliyorsa seçilir. Aksi hâlde **ilk OCR aynen korunur**.
5. OCR modelinde ikinci deneme hata verirse ilk başarılı sonuçla devam edilir. Hiçbir başarılı OCR'siz referans oluşturulmaz.

### PowerShell örneği (senin klasörün)

İlk referansı oluştururken:

```powershell
python .\ölçüm.py `
  --model ".\best (1).pt" `
  --image ".\fotoğraflar\orjinal.jpg" `
  --output ".\cevaplar" `
  --yolo-device 0 `
  --ocr-device cpu `
  --ocr-batch 4 `
  --ocr-denoise auto
```

Mevcut referans zaten sağlamsa `ölçüm.py` tekrar çalıştırılmak zorunda değil. Eski JSON'daki metinler de `search.py` tarafından temizlenebilir.
Referansı **yenilemek** istiyorsan yalnızca `ölçüm.py` komutuna `--force` ekle.

Yeni fotoğrafı kontrol etmek için:

```powershell
python .\search.py `
  --model ".\best (1).pt" `
  --image ".\fotoğraflar\karışık.jpg" `
  --reference ".\cevaplar" `
  --output ".\cevaplar\kontrol" `
  --yolo-device 0 `
  --ocr-device cpu `
  --ocr-batch 4 `
  --ocr-denoise auto `
  --ocr-debug-crops
```

`--ocr-debug-crops` yalnızca ikinci OCR için temizlenen görüntüleri `control_slot_crops/debug_cleaned/` altında kaydeder. Normal kullanımda kaldırabilirsin.
Temizleme özelliklerini kapatmak istersen `--ocr-denoise off` kullan.

**Önemli:** `--force` yalnızca `ölçüm.py` parametresidir, `search.py` desteklemez. Ayrıca `noise.py` dosyası `common.py` ve `features.py` ile aynı klasörde olmalıdır.

**Deşifre doğrulaması:** Gönderdiğin JSON'un kayıtlı OCR çıktılarında,
`DESIFRE! ... 01101000 ... MARTI` metni `DESIFRE! DEHA MARTI` olarak filtrelendi.
Önceki görüntü skorları sabit bırakılarak 100 aday yeniden eşleştirildi ve
`SLOT_008 -> SLOT_006` eşleşmesi 0,4086'dan 0,5437'ye yükseldi. Bu
bir *kayıtlı çıktı hesaplamasıdır*; gerçek PaddleOCR yeniden çalıştırma testi değildir.

## Kurulum (Windows PowerShell)

İlk olarak Python ortamını kurun ve etkinleştirin. `PaddlePaddle` CPU/GPU paketini **mevcut CUDA / sürücü sürümüne uygun resmî kurulum talimatlarıyla** yükleyin; GPU paketi için herkese uygun tek bir `pip install` komutu yoktur:

- https://www.paddlepaddle.org.cn/install/quick
- https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/OCR.html

Ardından:

```powershell
python -m pip install -r requirements.txt
```

### Windows + RTX 5070: YOLO GPU, PP-OCRv6 Small CPU

PaddlePaddle 3.3.0/3.3.1'deki oneDNN/PIR CPU hatasını önlemek ve oneDNN hızını korumak için:

```powershell
python -m pip uninstall paddlepaddle paddlepaddle-gpu -y
python -m pip install paddlepaddle==3.2.2 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
python -c "import paddle; print(paddle.__version__); paddle.utils.run_check()"
```

CUDA destekli PyTorch kuruluysa `--yolo-device 0 --ocr-device cpu` kullanın.
Paddle 3.3.x kuruluysa `--ocr-mkldnn auto` (varsayılan) oneDNN'i kapatarak
hata yolunu atlar; ancak bu durumda OCR daha yavaş olabilir.
İsteyen `--ocr-mkldnn off` seçeneğini açıkça verebilir.

### Önceki hatalı referansı yenileme

OCR'nin tamamı başarısız olduğu halde eski sürümün oluşturduğu **boş referansı**
yeniden oluşturmak için `ölçüm.py` çağrısına **`--force`** ekleyin.
Başarılı OCR olmadan artık yeni JSON kaydedilmez. Kısmen metinsiz kitaplara izin verilir,
ama OCR motorunun hata vermesine izin verilmez.


## Çalıştırma

**1) Referans oluştur**

```powershell
python .\ölçüm.py --model "C:\MODELLER\kitap_yolo26s_obb\weights\best.pt" --image "C:\RAF\ilk.jpeg" --output "C:\RAF\crops" --ocr-batch 4
```

`crops/kitap_sirasi_referans.json` ile `crops/slot_crops/REF_xxx/SLOT_*.jpg` üretilir. JSON'daki `slot_crop_file` yolu artık referans klasörüne göre **göreli** tutulur; eski mutlak yol kayıtları okunmaya devam eder.

**2) Referansa göre kontrol et**

```powershell
python .\search.py --model "C:\MODELLER\kitap_yolo26s_obb\weights\best.pt" --image "C:\RAF\son.jpeg" --reference "C:\RAF\crops" --output "C:\RAF\raf_kontrol_sonuc" --ocr-batch 4
```

Çıktılar:

- `raf_kontrol_raporu.txt`: yerinde, taşınan ve eşleşmeyen kitaplar
- `raf_kontrol_eslestirme_sonucu.json`: eşleşmeler ve tüm çift skorları
- `raf_kontrol_annotated.jpg`: kutular ve sıralar
- `control_slot_crops/SLOT_*.jpg`: kontrol fotoğrafındaki kitap kırpmaları

**3) Yeni fotoğrafı yeni referans yapmak için**

```powershell
python .\ölçüm.py --model "C:\MODELLER\kitap_yolo26s_obb\weights\best.pt" --image "C:\RAF\yeni_ilk.jpeg" --output "C:\RAF\crops" --force
```

**Önemli:** Referans varsa normal `ölçüm.py` çalıştırması ona dokunmaz. `--force` yeni crop kümesini hazırlar; yalnızca başarıyla tamamlandığında referans JSON'unu atomik değiştirir. Yeni JSON kaydedildikten sonra eski `REF_*` crop klasörleri silinir. Referans oluşturma giriş görselini saklamak kullanıcının sorumluluğundadır.

## Çalıştırma seçenekleri

| Seçenek | Varsayılan | Açıklama |
|---|---:|---|
| `--yolo-device` | `0` | CUDA 0 / `cpu` |
| `--ocr-mkldnn` | `auto` | Paddle 3.3.x CPU'da oneDNN kapatılır, 3.2.2'de açık |
| `--ocr-device` | `cpu` | Paddle CPU / GPU 0 |
| `--imgsz` | `1280` | YOLO OBB çıkarım boyutu |
| `--conf` | Ölçüm `0.65`, search `0.60` | YOLO tespit eşiği |
| `--ocr-batch` | `8` | OCR giriş listesi ve tanıma batch boyutu; 8 GB GPU için önce `4` deneyin |
| `--ocr-min-side` | `1400` | Crop uzun kenarı küçükse büyüt (eski davranış) |
| `--ocr-max-side` | `2400` | Maksimum uzun kenar |
| `--min-match` | `0.42` | Search için kabul eşiği |
| `--strong-match` | `0.65` | Güçlü eşleşme etiketi |

**Model uyarısı:** `yolo26s-obb.pt` hazır DOTA ağırlıkları, kitap sınıfını tanımaz. Parametre verilmezse bu model otomatik kullanılacaktır, ancak kitap tespiti için **fine-tune edilmiş OBB `best.pt` gereklidir**. Modelin `task="obb"` olması kod içinde kontrol edilir.

## Optimizasyonlar ve gerekçeleri

1. **Görüntü özellik önbelleği:** Referans için 1, güncel kitap için 4 döndürme hesaplanır; tüm NxM eşleştirmeler bu hazır özellikleri kullanır. HSV/pHash/Edge/ORB fonksiyonları korunur.
2. **Şekli koruyan resize:** Uzun dar kitap sırtları 256x256 alana gerilmek yerine aspect ratio korunarak letterbox edilir.
3. **Kenar skorunda boş-kenar düzeltmesi:** Kenarsız iki görüntü birbirine %100 eşit sayılmaz; 3x3 toleranslı kenar örtüşmesi kullanılır. **Eski skorlarla birebir aynı sonuçlar beklenmemeli.**
4. **Unicode Türkçe normalizasyonu:** `İ` / `I` / `ı` / aksan farkları doğru normalleştirilir.
5. **Küresel atama:** Greedy yerine scipy Hungarian; her kitap en fazla bir diğer kitapla eşleşir; eşik altında kalma hakkı korunur.
6. **Bellek içi toplu OCR:** JPEG temp oluşturup tekrar okumaz; PaddleOCR'ye NumPy görüntü listesi gönderir. Listede hata olursa tek tek tekrar dener.
7. **Korumalı referans nesli:** JSON ile crop'ların yanlışlıkla farklı referansa ait olması engellenir.
8. **OBB köşe sırası:** 45° açılı kutularda tekrar eden köşe problemini önleyen çevresel sıralama yapılır.
9. **Ölçümler:** Search sonucu JSON'unda süre alanları tutulur.
10. **Aynı çıktılar:** Eski `matches`, `unmatched_references`, `unmatched_currents`, `current_slots`, `all_pair_scores`, `image_details` gibi anahtarlar korunur. Ek olarak belirsizlik marjı ve model bilgisi kaydedilir.

## Davranış farkları / dikkat edilmesi gerekenler

- **Gerçek model ve raf fotoğraflarıyla hız veya doğruluk ölçülmedi.** Birim testleri model/OCR mock nesneleri ile çalışır.
- Yeni OCR modeli ve görüntü/kenar işlemleri skorları değiştirebilir. `0.42`/`0.65` eşikleri **kalibre edilmemiş eski değerlerdir**. Gerçek referans/sonraki raf fotoğraflarından etiketli eşleşme testleri yaparak ayarlayın.
- Kitap sırası hâlâ soldan sağa OBB minimum X koordinatına göre verilir (orijinal davranış). **Birden çok yatay raf satırı** için raf satırı sınıflandırma/sıralama kuralı ayrıca gerekir.
- OBB kutuları 180° yön bilgisi vermez; PaddleOCR document/textline orientation açık, görsel karşılaştırmada 0/90/180/270° açıları denenir.
- Değişen, eklenen ve kaldırılan kitapları tek tek sınıflandıran gerçek kimlik belirleme hatasız değildir. Aynı baskıdaki tamamen aynı kitaplar bir fotoğraftan kesin olarak ayrılamayabilir. `ambiguous`/`ambiguity_margin` alanları bu gibi durumlarda yardımcıdır.
- Crop JPEG'leri raporlama amacıyla saklanır; **yalnızca OCR ara görüntüsünün temp dosyası kaldırıldı**.
- Kod disk veya ağa fotoğraf göndermez. PaddleOCR/Ultralytics'in **ilk model indirme işlemi** İnternet bağlantısı gerektirebilir; KVKK ortamında modelleri önceden yerelde bulundurun.

## Test

```powershell
python -m pip install pytest
python -m pytest -q
```

Bu testler `paddleocr`/`ultralytics` modellerini indirmeden görüntü hazırlama, Türkçe normalizasyonu, OCR sonuç ayrıştırma, referans koruma, önbellek ve global eşleştirme akışlarını denetler. Gerçek GPU üzerinde model çalışma testi değildir.
