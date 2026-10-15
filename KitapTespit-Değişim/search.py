# -*- coding: utf-8 -*-
"""Kontrol rafını referansla karşılaştırır: YOLO26s-OBB + PP-OCRv6-small.

Özellikler: dört yönlü görsel karşılaştırma, HSV/pHash/Canny/ORB,
Türkçe OCR metin benzerliği, birebir global eşleştirme, hareket raporu,
JSON'da tüm çiftlerin skoru, işaretlenmiş fotoğraf ve crop çıktıları.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from common import (
    OCR_DET_MODEL, OCR_REC_MODEL, REFERENCE_ORDER_JSON,
    build_paddle_ocr, build_yolo, detect_and_ocr_slots, imread_unicode,
    imwrite_unicode, read_json, save_json_atomic, serializable_slots,
    clear_old_files, validate_ocr_slots,
)
from features import ImageFeatureExtractor, add_cached_features, text_similarity_normalized
from noise import analyze_binary_noise

MODEL_PATH = os.environ.get("BOOK_OBB_MODEL", "yolo26s-obb.pt")
CONTROL_IMAGE_PATH = r"C:\Users\Fatih\Documents\KOD\farklı.jpeg"
REFERENCE_DIR = r"C:\Users\Fatih\Documents\KOD\crops"
CONTROL_OUTPUT_DIR = r"C:\Users\Fatih\Documents\KOD\raf_kontrol_sonuc"
MIN_MATCH_SCORE = 0.42
STRONG_MATCH_SCORE = 0.65


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Referans raf ile kontrol rafını karşılaştır")
    parser.add_argument("--image", default=CONTROL_IMAGE_PATH, help="Yeni raf fotoğrafı")
    parser.add_argument("--reference", default=REFERENCE_DIR, help="ölçüm.py çıktısının klasörü")
    parser.add_argument("--output", default=CONTROL_OUTPUT_DIR, help="Raporların kaydedileceği klasör")
    parser.add_argument("--model", default=MODEL_PATH, help="EĞİTİLMİŞ YOLO26s-OBB best.pt")
    parser.add_argument("--yolo-device", default="0", help="0 veya cpu")
    parser.add_argument("--ocr-device", default="cpu", help="cpu veya gpu:0")
    parser.add_argument("--ocr-mkldnn", choices=("auto", "on", "off"), default="auto",
                        help="CPU oneDNN: auto, on veya off; Paddle 3.3.x için auto kapatır")
    parser.add_argument("--conf", type=float, default=0.60)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--ocr-batch", type=int, default=8)
    parser.add_argument("--ocr-denoise", choices=("auto", "off"), default="auto",
                        help="Yoğun 0/1 gürültüsünü temizle, gerekirse ikinci OCR yap")
    parser.add_argument("--ocr-debug-crops", action="store_true",
                        help="İkinci OCR için hazırlanan temiz görselleri kaydet")
    parser.add_argument("--ocr-min-side", type=int, default=1400)
    parser.add_argument("--ocr-max-side", type=int, default=2400)
    parser.add_argument("--min-match", type=float, default=MIN_MATCH_SCORE)
    parser.add_argument("--strong-match", type=float, default=STRONG_MATCH_SCORE)
    return parser.parse_args()


def _resolve_crop_file(reference_dir: Path, slot: dict[str, Any]) -> Path:
    """Yeni göreli ve eski mutlak kayıtları yükler; eski slot fallback'i korunur."""
    given = slot.get("slot_crop_file", "")
    path = Path(given) if given else Path("__missing__")
    if not path.is_absolute():
        path = reference_dir / path
    if path.is_file():
        return path
    fallback = reference_dir / "slot_crops" / f"{slot.get('slot_name', '')}.jpg"
    return fallback if fallback.is_file() else path


def load_reference_slots(reference_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reference_dir = Path(reference_dir)
    reference_path = reference_dir / REFERENCE_ORDER_JSON
    data = read_json(reference_path)
    if data is None:
        raise FileNotFoundError(f"Referans JSON yok: {reference_path}. Önce ölçüm.py çalıştır.")
    slots = data.get("slots", [])
    if not slots:
        raise ValueError("Referans dosyasında hiçbir kitap yok")
    validate_ocr_slots(slots, origin="Yüklenen referans")
    loaded = []
    for row in slots:
        path = _resolve_crop_file(reference_dir, row)
        image = None
        if path.is_file():
            image = imread_unicode(path)
        else:
            print(f"[WARN] Referans crop bulunamadı: {row.get('slot_name')}: {path}")
        legacy_clean = analyze_binary_noise(row.get("text", ""))
        loaded.append({
            "slot_name": row.get("slot_name", ""),
            "order_index": int(row.get("order_index", 0)),
            "text": row.get("text", ""),
            "text_raw": row.get("text_raw", row.get("text", "")),
            "match_text": row.get("match_text", legacy_clean.filtered_text),
            "ocr_conf": float(row.get("ocr_conf", 0)),
            "ocr_quality_score": float(row.get("ocr_quality_score", 0)),
            "slot_crop_file": str(path),
            "crop_img": image,
            "raw": row,
        })
    loaded.sort(key=lambda s: s["order_index"])
    return loaded, data


def compare_pair(
    reference: dict[str, Any], current: dict[str, Any], extractor: ImageFeatureExtractor
) -> dict[str, Any]:
    text_score = text_similarity_normalized(
        reference.get("normalized_text", ""), current.get("normalized_text", "")
    )
    best_img, best_details, best_rotation = 0.0, {}, 0
    ref_features = reference["features"].get(0)
    if ref_features is not None:
        for angle, current_features in current["features"].items():
            img_score, details = extractor.score(ref_features, current_features)
            if img_score > best_img:
                best_img, best_details, best_rotation = img_score, details, angle
    best_details = dict(best_details)
    best_details["best_rotation"] = best_rotation
    ref_conf = float(reference.get("ocr_conf", 0.0))
    cur_conf = float(current.get("ocr_conf", 0.0))
    conf_score = float(np.clip((ref_conf + cur_conf) / 2, 0, 1))
    score = float(np.clip(text_score * 0.45 + best_img * 0.50 + conf_score * 0.05, 0, 1))
    return {
        "reference_slot": reference["slot_name"],
        "reference_order": reference["order_index"],
        "current_slot": current["slot_name"],
        "current_order": current["order_index"],
        "reference_text": reference["text"],
        "current_text": current["text"],
        "text_similarity": round(text_score, 4),
        "image_similarity": round(best_img, 4),
        "ocr_conf_score": round(conf_score, 4),
        "final_score": round(score, 4),
        "image_details": best_details,
    }


def global_match(
    reference_slots: list[dict[str, Any]], current_slots: list[dict[str, Any]],
    extractor: ImageFeatureExtractor, minimum: float, strong: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Hungarian global en yüksek toplam skor; her kitap en fazla bir kez eşleşir.

    n_ref adet dummy sütun, referans kitapların eşleşmeden kalmasına izin verir.
    Eşik altı adaylar seçime kapalıdır. Tüm aday skorları raporda tutulur.
    """
    n_ref, n_cur = len(reference_slots), len(current_slots)
    if n_ref == 0:
        return [], [], current_slots, []
    if n_cur == 0:
        return [], reference_slots, [], []

    all_pairs: list[dict[str, Any]] = []
    matrix = np.zeros((n_ref, n_cur), dtype=np.float64)
    for i, ref in enumerate(reference_slots):
        for j, cur in enumerate(current_slots):
            pair = compare_pair(ref, cur, extractor)
            all_pairs.append(pair)
            matrix[i, j] = pair["final_score"]

    # Positive dummy maliyeti yoktur; geçersiz adaylar asla dummy'den iyi olamaz.
    costs = np.zeros((n_ref, n_cur + n_ref), dtype=np.float64)
    costs[:, :n_cur] = np.where(matrix >= minimum, -matrix, 1e6)
    rows, cols = linear_sum_assignment(costs)

    used_refs, used_currents = set(), set()
    matches = []
    pair_lookup = {(p["reference_slot"], p["current_slot"]): p for p in all_pairs}
    for i, j in zip(rows, cols):
        if j >= n_cur or matrix[i, j] < minimum:
            continue
        ref, cur = reference_slots[i], current_slots[j]
        pair = dict(pair_lookup[(ref["slot_name"], cur["slot_name"])])
        used_refs.add(ref["slot_name"])
        used_currents.add(cur["slot_name"])
        original, now = ref["order_index"], cur["order_index"]
        pair["status"] = "YERİNDE" if original == now else "YER DEĞİŞTİRDİ"
        pair["sentence"] = (
            f"{original}. kitap yerinde duruyor." if original == now else
            f"{original}. kitap şu anda {now}. kitap yerinde duruyor."
        )
        pair["confidence_label"] = "GÜÇLÜ" if pair["final_score"] >= strong else "ORTA"
        # Alternatif adaylara yakınlık, yüksek skorla karıştırılmamalı.
        alternatives = [matrix[i, other] for other in range(n_cur) if other != j]
        alternatives += [matrix[other, j] for other in range(n_ref) if other != i]
        margin = float(matrix[i, j] - max(alternatives, default=0.0))
        pair["ambiguity_margin"] = round(margin, 4)
        pair["ambiguous"] = margin < 0.05
        matches.append(pair)

    unmatched_references = [s for s in reference_slots if s["slot_name"] not in used_refs]
    unmatched_currents = [s for s in current_slots if s["slot_name"] not in used_currents]
    matches.sort(key=lambda m: m["reference_order"])
    all_pairs.sort(key=lambda p: p["final_score"], reverse=True)
    return matches, unmatched_references, unmatched_currents, all_pairs


def build_report(
    matches: list[dict[str, Any]], unmatched_references: list[dict[str, Any]],
    unmatched_currents: list[dict[str, Any]],
) -> str:
    lines = ["=" * 70, "RAF KONTROL RAPORU", "=" * 70, "", "[ÖZET]"]
    moved = [m for m in matches if m["status"] == "YER DEĞİŞTİRDİ"]
    same = [m for m in matches if m["status"] == "YERİNDE"]
    lines += [
        f"Eşleşen kitap sayısı: {len(matches)}",
        f"Yerinde duran kitap sayısı: {len(same)}",
        f"Yer değiştiren kitap sayısı: {len(moved)}",
        f"Referansta olup kontrol fotoğrafında net eşleşmeyen kitap sayısı: {len(unmatched_references)}",
        f"Kontrol fotoğrafında olup referansla net eşleşmeyen kitap sayısı: {len(unmatched_currents)}",
        "", "[KİTAP HAREKETLERİ]",
    ]
    if not matches:
        lines.append("Net eşleşme bulunamadı.")
    for match in matches:
        lines += [
            f"- {match['sentence']} Durum: {match['status']} | Güven: {match['confidence_label']} | "
            f"Genel skor: %{match['final_score']*100:.1f} | "
            f"Metin benzerliği: %{match['text_similarity']*100:.1f} | "
            f"Görüntü benzerliği: %{match['image_similarity']*100:.1f}"
            + (" | UYARI: yakın alternatif eşleşme" if match["ambiguous"] else ""),
            f"  Referans OCR: {match['reference_text']}",
            f"  Kontrol OCR : {match['current_text']}", "",
        ]
    if unmatched_references:
        lines.append("[REFERANSTA VAR AMA KONTROLDE NET BULUNAMADI]")
        for ref in unmatched_references:
            lines.append(f"- Referans {ref['order_index']}. kitap ({ref['slot_name']}) "
                         f"net eşleşmedi. OCR: {ref.get('text', '')}")
        lines.append("")
    if unmatched_currents:
        lines.append("[KONTROLDE VAR AMA REFERANSTA NET BULUNAMADI]")
        for cur in unmatched_currents:
            lines.append(f"- Kontrol {cur['order_index']}. sıradaki kitap ({cur['slot_name']}) "
                         f"referansla net eşleşmedi. OCR: {cur.get('text', '')}")
        lines.append("")
    lines.append("[ŞU ANKİ RAFA GÖRE OKUMA]")
    for match in sorted(matches, key=lambda m: m["current_order"]):
        lines.append(f"- Şu an {match['current_order']}. yerde duran kitap, "
                     f"referanstaki {match['reference_order']}. kitap gibi görünüyor. "
                     f"Skor: %{match['final_score'] * 100:.1f}")
    return "\n".join(lines)


def annotate_image(
    image: np.ndarray, current_slots: list[dict[str, Any]],
    matches: list[dict[str, Any]], output_path: Path,
) -> None:
    annotated = image.copy()
    current_to_match = {match["current_slot"]: match for match in matches}
    for cur in current_slots:
        points = np.asarray(cur["quad_points"], dtype=np.int32).reshape(4, 2)
        cv2.polylines(annotated, [points], True, (0, 255, 0), 2)
        now = cur["order_index"]
        match = current_to_match.get(cur["slot_name"])
        label = (f"S{now} <- R{match['reference_order']} %{match['final_score']*100:.0f}"
                 if match else f"S{now} <- ?")
        x = max(0, int(np.min(points[:, 0])))
        y = max(25, int(np.min(points[:, 1])) - 10)
        cv2.putText(annotated, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2, cv2.LINE_AA)
    imwrite_unicode(output_path, annotated)


def main() -> None:
    args = parse_args()
    if not (0 <= args.conf <= 1 and 0 <= args.min_match <= args.strong_match <= 1):
        raise ValueError("conf / min-match / strong-match için geçersiz eşik")
    if args.ocr_batch < 1 or args.ocr_max_side < 1 or args.ocr_min_side > args.ocr_max_side:
        raise ValueError("Geçersiz OCR batch veya çözünürlük")
    image_path = Path(args.image).resolve()
    reference_dir = Path(args.reference).resolve()
    output_dir = Path(args.output).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Kontrol fotoğrafı yok: {image_path}")
    start = time.perf_counter()
    reference_slots, _ = load_reference_slots(reference_dir)
    print(f"[INFO] Referanstaki kitap sayısı: {len(reference_slots)}")
    if args.model == "yolo26s-obb.pt":
        print("[UYARI] Hazır yolo26s-obb.pt kitaplar için eğitilmemiştir; --model best.pt kullan.")
    print(f"[MODEL] YOLO26s-OBB: {args.model}")
    model = build_yolo(args.model)
    print(f"[MODEL] {OCR_DET_MODEL} + {OCR_REC_MODEL} (tr)")
    ocr = build_paddle_ocr(args.ocr_device, args.ocr_batch, args.ocr_mkldnn)

    crop_dir = output_dir / "control_slot_crops"
    clear_old_files(crop_dir)
    image, current_slots, durations = detect_and_ocr_slots(
        model, ocr, image_path, crop_dir,
        yolo_device=args.yolo_device, imgsz=args.imgsz, conf=args.conf,
        ocr_batch_size=args.ocr_batch, ocr_min_side=args.ocr_min_side,
        ocr_max_side=args.ocr_max_side,
        ocr_denoise=args.ocr_denoise,
        ocr_debug_clean_dir=(crop_dir / "debug_cleaned") if args.ocr_debug_crops else None,
    )

    validate_ocr_slots(current_slots, origin="Kontrol fotoğrafı")
    if args.ocr_denoise == "off":
        for row in reference_slots:
            row["match_text"] = row["text"]
    print("[INFO] Özellikler bir defa hesaplanıyor, global eşleştirme hazırlanıyor...")
    compare_start = time.perf_counter()
    extractor = ImageFeatureExtractor()
    add_cached_features(reference_slots, extractor, rotate=False)
    add_cached_features(current_slots, extractor, rotate=True)
    matches, unmatched_refs, unmatched_currents, all_pairs = global_match(
        reference_slots, current_slots, extractor, args.min_match, args.strong_match
    )
    compare_seconds = time.perf_counter() - compare_start
    report_text = build_report(matches, unmatched_refs, unmatched_currents)
    result_payload = {
        "reference_dir": str(reference_dir),
        "control_image_path": str(image_path),
        "reference_total_books": len(reference_slots),
        "current_total_books": len(current_slots),
        "matches": matches,
        "unmatched_references": [
            {"slot_name": r["slot_name"], "order_index": r["order_index"],
             "text": r.get("text", ""), "slot_crop_file": r.get("slot_crop_file", "")}
            for r in unmatched_refs
        ],
        "unmatched_currents": serializable_slots(unmatched_currents),
        "current_slots": serializable_slots(current_slots),
        "all_pair_scores": all_pairs,
        "models": {
            "yolo": args.model, "ocr_detection": OCR_DET_MODEL, "ocr_recognition": OCR_REC_MODEL,
        },
        "timings_seconds": {**durations, "matching": round(compare_seconds, 3),
                            "total": round(time.perf_counter() - start, 3)},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "raf_kontrol_raporu.txt"
    json_path = output_dir / "raf_kontrol_eslestirme_sonucu.json"
    annotated_path = output_dir / "raf_kontrol_annotated.jpg"
    report_path.write_text(report_text, encoding="utf-8")
    save_json_atomic(json_path, result_payload)
    annotate_image(image, current_slots, matches, annotated_path)

    print("\n" + report_text)
    print(f"\n[DONE] {len(matches)} eşleşme, {len(unmatched_refs)} eksik referans, "
          f"{len(unmatched_currents)} eşleşmeyen güncel kitap")
    print(f"[SÜRE] YOLO+crop={durations['detect_and_crop_seconds']:.2f}s "
          f"OCR={durations['ocr_seconds']:.2f}s eşleştirme={compare_seconds:.2f}s "
          f"toplam={time.perf_counter()-start:.2f}s")
    print(f"[DONE] Rapor: {report_path}\n[DONE] JSON: {json_path}\n[DONE] Görsel: {annotated_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[HATA] {exc}", file=sys.stderr)
        raise SystemExit(1)
