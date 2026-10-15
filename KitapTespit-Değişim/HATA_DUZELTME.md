# Boş OCR ve Paddle 3.3.x oneDNN hatası

## Sebep

YOLO 10 kitabı buluyor ancak PaddlePaddle CPU 3.3.0 oneDNN/PIR yürütücüsü `ArrayAttribute<DoubleAttribute>` destekleyemediği için OCR başarısız oluyor. Sürenin 0.58 saniye olması başarılı OCR çıkarımı değil.

## Önerilen çözüm (hızlı CPU OCR)

```powershell
python -m pip uninstall paddlepaddle paddlepaddle-gpu -y
python -m pip install paddlepaddle==3.2.2 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
python -c "import paddle; print(paddle.__version__); paddle.utils.run_check()"
python ölçüm.py --model '.\best (1).pt' --image '.\fotoğraflar\orjinal.jpg' --output '.\cevaplar' --yolo-device 0 --ocr-device cpu --ocr-batch 4 --force
```

## Alternatif

3.3.0 CPU sürümünde kalmak zorundaysan `--ocr-mkldnn off` ile oneDNN devre dışı bırakılır. Bu, Paddle OCR'yi yavaşlatabilir. `auto` zaten 3.3.x tespit ettiğinde bunu uygular.

## Kod güvenliği

- Fatal PIR/oneDNN hatası artık her crop için tekrar edilmez, işlem başarısız sonlanır.
- OCR işlem hataları ve 10/10 boş OCR çıktısı geçerli referans olarak yazılmaz.
- Önceden oluşturulan `kitap_sirasi_referans.json` korunur; geçersiz referansı düzeltmek için `--force` gereklidir.
- PP-OCRv6 small det/rec, YOLO26s-OBB, OCR yön düzeltme ve görüntü karşılaştırma korunur.
- `lang` ve `ocr_version` uyarısına yol açan, açık model adları varken yok sayılan parametreler kaldırıldı.

Kaynak: https://github.com/PaddlePaddle/Paddle/issues/77340
