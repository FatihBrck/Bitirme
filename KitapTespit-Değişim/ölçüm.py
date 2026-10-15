# -*- coding: utf-8 -*-
"""Referans kitap rafı oluşturma. YOLO26s-OBB + PP-OCRv6-small.

İlk çalıştırma: referans JSON ve crop'lar üretilir.
Tekrar çalıştırma: referans değiştirilmez (görüntü/JSON tutarlılığı).
Yeni referans: --force ile atomik JSON değişimi yapılır.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

from common import (
    OCR_DET_MODEL, OCR_REC_MODEL, REFERENCE_ORDER_JSON,
    build_paddle_ocr, build_yolo, detect_and_ocr_slots,
    read_json, save_json_atomic, serializable_slots, validate_ocr_slots,
)

IMAGE_PATH = r"C:\Users\Fatih\Documents\KOD\düz.jpeg"
OUTPUT_DIR = r"C:\Users\Fatih\Documents\KOD\crops"
MODEL_PATH = os.environ.get("BOOK_OBB_MODEL", "yolo26s-obb.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kitap rafının referans sırasını kaydet")
    parser.add_argument("--image", default=IMAGE_PATH, help="Referans raf fotoğrafı")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Referans JSON ve crop klasörü")
    parser.add_argument("--model", default=MODEL_PATH, help="EĞİTİLMİŞ YOLO26s-OBB best.pt")
    parser.add_argument("--force", action="store_true", help="Önceki referansı güvenli biçimde yenile")
    parser.add_argument("--yolo-device", default="0", help="0 veya cpu")
    parser.add_argument("--ocr-device", default="cpu", help="cpu veya gpu:0")
    parser.add_argument("--ocr-mkldnn", choices=("auto", "on", "off"), default="auto",
                        help="CPU oneDNN: auto, on veya off; Paddle 3.3.x için auto kapatır")
    parser.add_argument("--conf", type=float, default=0.65)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--ocr-batch", type=int, default=8)
    parser.add_argument("--ocr-denoise", choices=("auto", "off"), default="auto",
                        help="Yoğun 0/1 gürültüsü olan kitapları ikinci OCR ile dene")
    parser.add_argument("--ocr-debug-crops", action="store_true",
                        help="İkinci OCR için temizlenen görüntüleri diske kaydet")
    parser.add_argument("--ocr-min-side", type=int, default=1400)
    parser.add_argument("--ocr-max-side", type=int, default=2400)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (0 <= args.conf <= 1) or args.ocr_batch < 1 or args.ocr_max_side < 1:
        raise ValueError("Geçersiz conf, OCR batch veya çözünürlük")
    if args.ocr_min_side > args.ocr_max_side:
        raise ValueError("OCR minimum uzun kenarı maksimumdan büyük olamaz")
    image_path = Path(args.image).resolve()
    output_dir = Path(args.output).resolve()
    reference_path = output_dir / REFERENCE_ORDER_JSON
    if not image_path.is_file():
        raise FileNotFoundError(f"Referans görseli bulunamadı: {image_path}")

    old_payload = read_json(reference_path)
    if old_payload is not None and not args.force:
        print(f"[KORUMA] Mevcut referansa dokunulmadı: {reference_path}")
        print("[KORUMA] Yeni fotoğrafla yeniden oluşturmak için --force kullan.")
        return

    if args.model == "yolo26s-obb.pt":
        print("[UYARI] Hazır yolo26s-obb.pt DOTA üzerinde eğitilmiş; kitap tespiti için kendi best.pt dosyanı ver.")
    print(f"[MODEL] YOLO26s-OBB: {args.model}")
    model = build_yolo(args.model)
    print(f"[MODEL] {OCR_DET_MODEL} + {OCR_REC_MODEL} (tr)")
    ocr = build_paddle_ocr(args.ocr_device, args.ocr_batch, args.ocr_mkldnn)

    # Referans ve crop tek sürüme aittir. Başarısız işlem eskisini etkilemez.
    generation = "REF_" + uuid.uuid4().hex[:12]
    crop_dir = output_dir / "slot_crops" / generation
    try:
        start = time.perf_counter()
        _, slots, timings = detect_and_ocr_slots(
            model, ocr, image_path, crop_dir,
            yolo_device=args.yolo_device, imgsz=args.imgsz, conf=args.conf,
            ocr_batch_size=args.ocr_batch, ocr_min_side=args.ocr_min_side,
            ocr_max_side=args.ocr_max_side, relative_crop_base=output_dir,
            ocr_denoise=args.ocr_denoise,
            ocr_debug_clean_dir=(crop_dir / "debug_cleaned") if args.ocr_debug_crops else None,
        )
        if not slots:
            raise RuntimeError("Hiç kitap tespit edilemedi; boş referans yazılmadı. best.pt ve conf değerini kontrol et.")
        validate_ocr_slots(slots, origin="Referans")
        payload = {
            "image_path": str(image_path),
            "total_books": len(slots),
            "slots": serializable_slots(slots),
            "reference_generation": generation,
            "yolo_model": args.model,
            "ocr_detection_model": OCR_DET_MODEL,
            "ocr_recognition_model": OCR_REC_MODEL,
        }
        save_json_atomic(reference_path, payload)
        print(f"[DONE] Referans kaydedildi: {reference_path}")
        print(f"[DONE] Kitap: {len(slots)} | YOLO={timings['yolo_seconds']:.2f}s "
              f"OCR={timings['ocr_seconds']:.2f}s toplam={time.perf_counter()-start:.2f}s")
    except Exception:
        if crop_dir.exists():
            shutil.rmtree(crop_dir)
        raise

    # Başarılı JSON değişiminden sonra artık kullanılmayan otomatik nesilleri sil.
    # Legacy slot_crops/SLOT_*.jpg dosyalarına dokunulmaz.
    if args.force:
        for old_generation in (output_dir / "slot_crops").glob("REF_*"):
            if old_generation.is_dir() and old_generation.name != generation:
                try:
                    shutil.rmtree(old_generation)
                except OSError as exc:
                    print(f"[WARN] Eski referans crop klasörü silinemedi: {exc}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[HATA] {exc}", file=sys.stderr)
        raise SystemExit(1)
