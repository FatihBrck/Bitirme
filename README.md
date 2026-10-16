# Kitap Rafı Kitap Tespiti ve Sıra Kontrolü

**Bitirme projesi | YOLO26s-OBB + PaddleOCR PP-OCRv6 Small | CPU üzerinde çalıştırma**

Bu proje, bir kitap rafındaki kitapları tespit edip kitap sırtlarından metin okuyarak **referans raf** ile **güncel raf** fotoğraflarını karşılaştırır. Kitapların yerinin değişmesini, referansta olup güncel görüntüde bulunmamasını ve eşleşmeyen yeni kitapları raporlar.

> **CPU-only çalışma:** Bu README'deki kurulum, test, tespit, OCR, karşılaştırma ve isteğe bağlı yeniden eğitim komutları **CPU** hedefler. NVIDIA GPU, CUDA ve cuDNN gerekli değildir. Kodda YOLO için varsayılan aygıt `0` (GPU) olduğundan, gerçek çalıştırmalarda **`--yolo-device cpu --ocr-device cpu`** parametreleri mutlaka verilmelidir. Eğitim kayıtlarının geçmişte GPU üzerinde üretilmiş olması, eğitilmiş ağırlıkların CPU'da kullanılmasına engel değildir.

## İçindekiler

1. [Sistemin çalışma mantığı](#sistemin-çalışma-mantığı)
2. [Proje yapısı](#proje-yapısı)
3. [Gereksinimler ve indirme bağlantıları](#gereksinimler-ve-indirme-bağlantıları)
4. [Windows / PowerShell kurulumu](#windows--powershell-kurulumu)
5. [CPU üzerinde çalıştırma](#cpu-üzerinde-çalıştırma)
6. [YOLO modelinin eğitimi](#yolo-modelinin-eğitimi)
7. [Testler ve sorun giderme](#testler-ve-sorun-giderme)

## Sistemin çalışma mantığı

1. **Kitap tespiti:** Eğitilmiş **YOLO26s-OBB** modeli kitap sırtlarını döndürülmüş sınır kutularıyla (Oriented Bounding Box) tespit eder.
2. **Görsel kırpma:** Her tespit için perspektif düzeltilmiş kitap-sırtı görüntüsü (crop) hazırlanır.
3. **Metin okuma:** **PaddleOCR PP-OCRv6 Small** tespit ve tanıma modelleri, kitap adlarını ve yazar adlarını CPU'da okur.
4. **Özellik çıkarma:** Metin, HSV histogramı, pHash, Canny kenarları ve ORB özellikleri değerlendirilir. OCR'da yoğun `0/1` gürültüsü varsa seçici temizleme / yeniden okuma uygulanabilir.
5. **Raf karşılaştırması:** Görsel, metin ve OCR güven skorları birleştirilerek **Hungarian (global birebir) eşleştirme** ile referans ve güncel kitaplar karşılaştırılır.
6. **Çıktılar:** Eşleşme/yer değişikliği bilgileri, eksik ve yeni kitaplar, JSON raporu, metin raporu ve işaretlenmiş raf görseli üretilir.

**İki aşamalı kullanım:** Önce **`ölçüm.py`** ile referans oluşturulur. Daha sonra **`search.py`** ile yeni raf görseli bu referansla karşılaştırılır.

## Proje yapısı

Bu README **`Bitirme/README.md`** yolundadır. Ancak çalıştırılacak Python dosyaları, deponun kökündeki **kardeş klasör** `KitapTespit-Değişim/` altındadır.

~~~text
Bitirme/                         # GitHub deposunun kökü
├── dataset-link.txt             # Veri seti bağlantıları
├── Bitirme/
│   ├── README.md                # Bu belge
│   ├── Train.ipynb              # Orijinal YOLO eğitim not defteri (Colab/GPU geçmişi)
│   └── YOLO26s_OBB_1280/
│       ├── args.yaml            # Orijinal eğitim parametreleri
│       ├── results.csv          # Eğitim metrikleri
│       ├── results.png          # Eğitim grafikleri
│       └── weights/
│           ├── best.pt          # En iyi kaydedilen model
│           └── last.pt          # Son epoch modeli
└── KitapTespit-Değişim/
    ├── best.pt                  # Çalıştırma için hazır eğitilmiş ağırlık
    ├── requirements.txt         # Python bağımlılıkları
    ├── common.py                # YOLO, OCR ve crop işlemleri
    ├── features.py              # Görsel/metin benzerliği özellikleri
    ├── noise.py                 # OCR gürültü analizi
    ├── ölçüm.py                 # Referans raf oluşturma
    ├── search.py                # Güncel raf karşılaştırması
    ├── test_raf.py              # Regresyon/smoke testleri
    ├── fotoğraflar/
    │   ├── orjinal.jpg          # Örnek referans görseli
    │   └── karışık.jpg          # Örnek karıştırılmış raf
    └── cevaplar/                # Repoda bulunan örnek referans ve crop dosyaları
~~~

İlgili dosyalar: [eğitim not defteri](./Train.ipynb), [eğitim ayarları](./YOLO26s_OBB_1280/args.yaml), [sonuçlar](./YOLO26s_OBB_1280/results.csv), [model ağırlığı](./YOLO26s_OBB_1280/weights/best.pt), [çalışan kodlar](../KitapTespit-De%C4%9Fi%C5%9Fim/).

## Gereksinimler ve indirme bağlantıları

- **Windows 10/11, 64 bit** (aşağıdaki komutlar Windows PowerShell içindir).
- **Python 3.10, 64 bit**: [Python 3.10.11 Windows yükleyicileri](https://www.python.org/downloads/release/python-31011/). Kurarken **Add Python to PATH** seçeneğini işaretleyin.
- **Git**: [Git for Windows](https://git-scm.com/downloads/win).
- **PyTorch (CPU)**: [resmî kurulum sayfası](https://pytorch.org/get-started/locally/).
- **PaddlePaddle (CPU)**: [PaddleOCR / PaddlePaddle kurulum kılavuzu](https://www.paddleocr.ai/main/en/version3.x/installation.html).
- **Ultralytics**: [Ultralytics dokümantasyonu](https://docs.ultralytics.com/).

**Model dosyasını ayrıca indirmeniz gerekmiyor:** Depoda [`KitapTespit-Değişim/best.pt`](../KitapTespit-De%C4%9Fi%C5%9Fim/best.pt) ve [eğitim çıktısındaki `best.pt`](./YOLO26s_OBB_1280/weights/best.pt) bulunmaktadır. **Çalıştırma örnekleri ilkini kullanır.** YOLO taban model ağırlığı sadece yeniden eğitime başlarken otomatik indirilebilir. PaddleOCR model ağırlıkları ilk gerçek OCR çalıştırmasında indirilebilir; ilk kurulum/başlatma için internet gerekebilir.

### Veri seti bağlantıları

Veri seti URL'leri deponun kökündeki [`dataset-link.txt`](../dataset-link.txt) dosyasında tutulur:

1. [Roboflow Universe - Book Counter, dataset v1](https://universe.roboflow.com/sakaryauniversite/book-counter-rhspr/dataset/1). Açıklamada, bu veri setinin fork edilip fotoğrafların klonlanarak çözünürlük ayarlarının değiştirildiği belirtilir.
2. [Roboflow Universe - Kitaplaar çalışma alanı](https://universe.roboflow.com/fatihs-workspace-ojhsw/kitaplaar).

**Hazır modelle raf analizi yapmak için veri setini indirmeniz gerekmez.** Veri seti, **yeniden eğitim** veya veri incelemesi içindir. Yeniden eğitim için görüntülerle birlikte **YOLO OBB** etiketleri ve `data.yaml` içeren uygun dışa aktarımı alın. Roboflow indirme/erişim seçenekleri çalışma alanı iznine bağlı olabilir.

## Windows / PowerShell kurulumu

**1. Depoyu klonlayın ve kök klasöre girin:**

~~~powershell
git clone https://github.com/FatihBrck/Bitirme.git
cd .\Bitirme
~~~

Buradaki `cd .\Bitirme`, **GitHub deposunun köküne** girer; daha sonra içinde ayrıca `Bitirme/` adlı bir alt klasör vardır.

**2. Python 3.10 ile sanal ortam oluşturup etkinleştirin:**

~~~powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
~~~

PowerShell betik çalıştırmayı engellerse, yalnızca **o terminal oturumu** için aşağıdaki ayarı uygulayıp etkinleştirmeyi tekrar deneyin:

~~~powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
~~~

**3. CPU sürümlerini ve proje bağımlılıklarını yükleyin:**

~~~powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install paddlepaddle==3.2.2 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
python -m pip install -r ".\KitapTespit-Değişim\requirements.txt"
~~~

`requirements.txt` dosyası **`ultralytics>=8.4.0`**, **`paddleocr==3.7.0`**, OpenCV, NumPy, SciPy ve pytest ister. PaddlePaddle'ın CPU paketi bilerek ayrı kurulmaktadır. Aynı ortamda `paddlepaddle-gpu` kullanmayın.

**4. Kurulumu doğrulayın:**

~~~powershell
python -c "import torch, paddle, ultralytics, paddleocr; print('Torch:', torch.__version__, '| CUDA:', torch.cuda.is_available()); print('Paddle:', paddle.__version__, '| Device:', paddle.device.get_device())"
~~~

CPU paketleriyle beklenen durum **`CUDA: False`** ve **`Device: cpu`** olmasıdır. Bu kontrol model çıkarımı yapmaz.

**5. Çalışan kodların klasörüne geçin:**

~~~powershell
cd ".\KitapTespit-Değişim"
~~~

Bundan sonraki tüm **`ölçüm.py`**, **`search.py`** ve test komutları bu klasörde çalıştırılacaktır.

## CPU üzerinde çalıştırma

### A. Referans kitap rafını oluşturma

Aşağıdaki komut, depodaki **`fotoğraflar/orjinal.jpg`** görselini işler ve yeni bir **`cikti/referans`** klasörü oluşturur:

~~~powershell
python ".\ölçüm.py" --image ".\fotoğraflar\orjinal.jpg" --output ".\cikti\referans" --model ".\best.pt" --yolo-device cpu --ocr-device cpu --ocr-batch 4 --imgsz 1280
~~~

Üretilen referanslar örnek olarak şu yapıda saklanır:

~~~text
cikti/
└── referans/
    ├── kitap_sirasi_referans.json
    └── slot_crops/
        └── REF_<benzersiz_kimlik>/
            ├── SLOT_001.jpg
            ├── SLOT_002.jpg
            └── ...
~~~

- `kitap_sirasi_referans.json`, kitapların sırasını, OCR bilgilerini ve ilgili crop yollarını kaydeder.
- Aynı çıktı klasöründe referans **zaten varsa**, `ölçüm.py` mevcut referansı korur ve yeniden oluşturmaz.
- Yeni bir referansla mevcut kaydı değiştirmek isterseniz **bilerek** `--force` ekleyin. Bu, önceki referans ve ona bağlı oluşturulmuş crop sürümünü yenileyebilir.
- İlk OCR açılışı model indirmelerinden ötürü daha uzun sürebilir.

### B. Değişen rafı referansla karşılaştırma

~~~powershell
python ".\search.py" --image ".\fotoğraflar\karışık.jpg" --reference ".\cikti\referans" --output ".\cikti\kontrol" --model ".\best.pt" --yolo-device cpu --ocr-device cpu --ocr-batch 4 --imgsz 1280
~~~

Bu komut:
- Güncel rafı YOLO ile tespit eder.
- Kitap sırtlarını kırpar ve PaddleOCR ile metinlerini okur.
- Referans ile her kitap için benzerlik hesaplar; en iyi **birebir global eşleştirmeyi** bulur.
- Sonuçları dosyalara kaydeder.

~~~text
cikti/
└── kontrol/
    ├── raf_kontrol_raporu.txt
    ├── raf_kontrol_eslestirme_sonucu.json
    ├── raf_kontrol_annotated.jpg
    └── control_slot_crops/
        ├── SLOT_001.jpg
        └── ...
~~~

**Çıktıların anlamı:**

| Dosya | İçerik |
|---|---|
| `raf_kontrol_raporu.txt` | Okunabilir eşleşme, yer değişimi ve eksik kitap raporu |
| `raf_kontrol_eslestirme_sonucu.json` | Tüm eşleştirmeler, aday skorları, eşleşmeyen kitaplar ve süreler |
| `raf_kontrol_annotated.jpg` | Güncel rafın üzerine çizilmiş sonuçlar |
| `control_slot_crops/` | Güncel raftan kırpılan kitap görüntüleri |

### C. Repoda zaten bulunan örnek referansı kullanma

Önceden oluşturulmuş **`cevaplar/kitap_sirasi_referans.json`** ve crop dosyalarını kullanmak isterseniz referansı yeniden üretmeden de deneyebilirsiniz:

~~~powershell
python ".\search.py" --image ".\fotoğraflar\karışık.jpg" --reference ".\cevaplar" --output ".\cikti\hazir_referans_kontrol" --model ".\best.pt" --yolo-device cpu --ocr-device cpu --ocr-batch 4 --imgsz 1280
~~~

### D. CPU performans / OCR ayarları

Aşağıdaki parametreler **hem referans oluşturmada hem kontrol komutunda** kullanılabilir:

| Parametre | Görev | Not |
|---|---|---|
| `--yolo-device cpu` | YOLO tespitini CPU'da çalıştırır | **Zorunlu CPU tercihi** |
| `--ocr-device cpu` | PaddleOCR'ı CPU'da çalıştırır | **Zorunlu CPU tercihi** |
| `--imgsz 1280` | YOLO görüntü boyutu | Daha düşük değer CPU'da hız kazanabilir; küçük kitap sırtlarını kaçırabilir |
| `--ocr-batch 4` | OCR batch boyutu | Bellek sınırlıysa `1` veya `2` seçin |
| `--ocr-mkldnn auto` | CPU oneDNN ayarı | Paddle 3.3.x'te sorun olursa `off` kullanın |
| `--ocr-denoise auto` | Seçici OCR gürültü temizleme | `off` ile devre dışı bırakılabilir |
| `--conf 0.60` | YOLO güven eşiği | `ölçüm.py` varsayılanı `0.65`; `search.py` varsayılanı `0.60` |
| `--ocr-min-side 1400` / `--ocr-max-side 2400` | OCR yeniden boyutlandırma sınırları | CPU bellek kullanımını etkileyebilir |

Örneğin kaynakları daha sınırlı bir bilgisayarda ilk deneme:

~~~powershell
python ".\search.py" --image ".\fotoğraflar\karışık.jpg" --reference ".\cikti\referans" --output ".\cikti\kontrol_hafif" --model ".\best.pt" --yolo-device cpu --ocr-device cpu --ocr-mkldnn off --ocr-batch 2 --imgsz 960
~~~

**Not:** Bu daha hafif komut, aynı referansla çalışsa da tespit performansı ve sonuçlar farklı olabilir. CPU'daki ilk açılış, disk/görüntü boyutu ve kitap sayısı işlem süresini etkiler; sabit bir süre garantisi yoktur.

## YOLO modelinin eğitimi

### Kullanılan model ve kaydedilmiş eğitim

Bu projede standart eksenlere paralel kutular yerine kitap sırtının açısını da yakalamak için **YOLO26s-OBB** kullanılmıştır. Model, referans raftaki kitapları birbirinden ayrıştırıp döndürülmüş dörtgenlere göre kırpmaya olanak tanır.

Kaynaklar:
- [`Train.ipynb` - özgün eğitim kodu](./Train.ipynb)
- [`args.yaml` - kaydedilen eğitim parametreleri](./YOLO26s_OBB_1280/args.yaml)
- [`results.csv` - epoch metrikleri](./YOLO26s_OBB_1280/results.csv)
- [`results.png` - eğitim eğrileri](./YOLO26s_OBB_1280/results.png)
- [`weights/best.pt` - en iyi model](./YOLO26s_OBB_1280/weights/best.pt)

Kaydedilmiş eğitim ayarlarından öne çıkanlar:

| Ayar | Değer |
|---|---|
| Görev / model | OBB / `yolo26s-obb.pt` |
| Görüntü boyutu | `1280` |
| Epoch | `200` |
| Erken durdurma (patience) | `50` |
| Batch | `48` |
| Optimizer | `MuSGD` |
| Eğitim cache | `ram` |
| Dönüşler | ±5° temel augmentasyon; ayrıca özel ağırlıklı döndürme |
| Ek augmentasyonlar | Yatay/dikey çevirme, ölçek, shear, perspective, HSV, blur/noise, sıkıştırma, gölge, CLAHE |
| Kayıtlı eğitim cihazı | `device=0` (GPU) |

Özgün **`Train.ipynb`**, Colab / A100 GPU ortamına göre yazılmış ve özelleştirilmiş `RotYOLODataset` içerir. Bu yapı yaklaşık olarak %60 görüntüyü olduğu gibi tutan, %20'sine 90°/270° döndürme uygulayan ve kalan %20'sini eğik açılarda döndüren bir veri artırma akışı tanımlar. **Aşağıdaki CPU komutu, bu özel Python monkey-patch akışını birebir yeniden üretmez.** İlgili parametrelerin aslı not defterinde mevcuttur.

Kaydedilmiş eğitimin **200. epoch** satırında precision **0.98444**, recall **0.98302**, mAP@50 **0.98987**, mAP@50:95 **0.87996** görülür. Bunlar mevcut `results.csv` doğrulama metrikleridir; yeni veri kümelerinde veya CPU üzerinde yeniden eğitimde aynı skorlar garanti edilmez.

### Veri setini hazırlama

1. Depo kökündeki [`dataset-link.txt`](../dataset-link.txt) içindeki Roboflow bağlantılarını açın.
2. Yeniden eğitim için görselleri ve **YOLO OBB** koordinat etiketlerini dışa aktarın. OBB anotasyonları nesne sınıfıyla birlikte dört köşenin koordinatlarını içermelidir.
3. Veri setini depo kökünde aşağıdaki örneğe benzer konuma açın:

~~~text
Bitirme/                   # GitHub deposu kökü
└── dataset/
    └── dataset/
        ├── data.yaml
        ├── train/
        │   ├── images/
        │   └── labels/
        ├── valid/          # Bazı veri kümelerinde "val" olabilir
        │   ├── images/
        │   └── labels/
        └── test/           # Varsa
~~~

Özgün Colab eğitimi veri setini `/content/dataset/dataset/data.yaml` yolundan okumuştur. **İndirdiğiniz arşivin klasör düzeni farklıysa komuttaki `data=...` yolunu değiştirin.** `data.yaml` içinde eğitim/doğrulama yollarının ve sınıfların doğru olduğundan emin olun. Veri kümesinin etiket biçimi OBB değilse dönüştürmeden OBB eğitimi başlatmayın.

### İsteğe bağlı: CPU üzerinde yeniden eğitim

Önceki kurulumun yapıldığı sanal ortam açıkken ve terminal **`KitapTespit-Değişim/`** dizinindeyken:

~~~powershell
yolo obb train model=yolo26s-obb.pt data="..\dataset\dataset\data.yaml" device=cpu epochs=200 patience=50 imgsz=1280 batch=1 workers=2 cache=False optimizer=MuSGD plots=True project=".\runs\train" name="yolo26s_obb_cpu"
~~~

- Bu komut **GPU kullanmaz**. Eğitim verisinin disk konumunu kendi indirme klasörünüze göre değiştirin.
- CPU eğitimi, özellikle 1280 çözünürlük ve 200 epoch için **çok yavaş** olabilir. İlk deneme için `epochs=5 imgsz=640` kullanarak veri formatını ve kurulumu kontrol edebilirsiniz. Bu değişiklikler orijinal modelle aynı eğitim koşullarını sağlamaz.
- `batch=1` ve `workers=2` CPU'da daha ulaşılabilir bellek tüketimi hedefler. Cihazınıza göre değiştirebilirsiniz.
- Yeni eğitimin `best.pt` dosyası `.\runs\train\yolo26s_obb_cpu\weights\best.pt` altında oluşur. Çalıştırmada `--model` parametresini bu yeni dosyaya yönlendirebilirsiniz.
- **Orijinal `Train.ipynb` dosyasını CPU üzerinde değiştirmeden çalıştırmayın:** not defteri CUDA/Colab ayarlarına sahiptir. CPU için yukarıdaki ayrı CLI eğitim örneği verilmiştir.

## Testler ve sorun giderme

### Otomatik testler

`KitapTespit-Değişim/` dizininde:

~~~powershell
python -m pytest -q test_raf.py
~~~

Bu testler metin normalizasyonu, OBB crop, OCR sonuç işleme ve eşleştirme gibi davranışları **gerçek model indirmeden** doğrular. Gerçek ağırlıklarla uçtan uca kontrol için yukarıdaki örnek fotoğraf komutlarını çalıştırın.

### Sık karşılaşılan sorunlar

| Sorun | Çözüm |
|---|---|
| `py -3.10` bulunamadı | Python 3.10 (64 bit) yükleyin; `py -0p` ile kurulu sürümleri kontrol edin. |
| PowerShell `Activate.ps1` engeli | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` kullanın. |
| `best.pt` bulunamadı | Komutları `KitapTespit-Değişim/` dizininde çalıştırın ve `--model ".\best.pt"` verin. |
| YOLO GPU/CUDA'ya gitmeye çalışıyor | Her **`ölçüm.py` ve `search.py`** çağrısına `--yolo-device cpu` yazın. |
| PaddleOCR açılmıyor / oneDNN / PIR hatası | CPU PaddlePaddle `3.2.2` kurulumunu doğrulayın, gerekirse `--ocr-mkldnn off` ekleyin. |
| Çok yavaş veya yüksek RAM tüketimi | `--ocr-batch 2` veya `1` deneyin; gerekirse `--imgsz 960` düşürün. |
| Referans JSON bulunamıyor | Önce `ölçüm.py` çalıştırın; `search.py --reference` değeri **JSON'un bulunduğu klasör** olmalıdır. |
| Referans üretilmiyor, eski referans korunuyor | Yeni `--output` klasörü kullanın veya gerçekten üzerine yazmak istiyorsanız `--force` verin. |
| Kitaplar yeterince tespit edilmiyor | Eğitilmiş `best.pt` kullandığınızı ve `--conf` eşiğini kontrol edin; daha düşük eşik yanlış pozitifleri artırabilir. |
| İlk açılışta indirme sorunu | PaddleOCR modellerinin ilk kullanımda indirildiğini unutmayın; bağlantı, proxy ve önbelleği kontrol edin. |

## Kısa CPU komut özeti

~~~powershell
# Depoyu indirme ve ortamı hazırlama (depo kökünde)
git clone https://github.com/FatihBrck/Bitirme.git
cd .\Bitirme
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install paddlepaddle==3.2.2 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
python -m pip install -r ".\KitapTespit-Değişim\requirements.txt"
cd ".\KitapTespit-Değişim"

# Referansı oluşturma
python ".\ölçüm.py" --image ".\fotoğraflar\orjinal.jpg" --output ".\cikti\referans" --model ".\best.pt" --yolo-device cpu --ocr-device cpu --ocr-batch 4 --imgsz 1280

# Güncel rafı karşılaştırma
python ".\search.py" --image ".\fotoğraflar\karışık.jpg" --reference ".\cikti\referans" --output ".\cikti\kontrol" --model ".\best.pt" --yolo-device cpu --ocr-device cpu --ocr-batch 4 --imgsz 1280

# Birim/regresyon testleri
python -m pytest -q test_raf.py
~~~

---

**Önemli:** Eğitim sonuçları ve model ağırlıkları bu depoda saklanır. Günlük **raf kontrolü / inference tamamen CPU üzerinde** çalıştırılabilir. Eğitim bilgisayarı ile çıkarım bilgisayarının aynı donanıma sahip olması gerekmez. Model doğruluğu rafın görüntü kalitesi, ışık, açı, OCR okunabilirliği ve veri dağılımına bağlıdır.
