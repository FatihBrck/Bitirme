# -*- coding: utf-8 -*-
"""Raf analizi için ortak YOLO26s-OBB / PP-OCRv6-small işlemleri."""
from __future__ import annotations

import json
from importlib import metadata
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from noise import analyze_binary_noise, suppress_light_background, should_use_retry

OCR_DET_MODEL = "PP-OCRv6_small_det"
OCR_REC_MODEL = "PP-OCRv6_small_rec"
OCR_SCORE_THRESHOLD = 0.25
REFERENCE_ORDER_JSON = "kitap_sirasi_referans.json"


def imread_unicode(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Görsel bulunamadı: {path}")
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Görsel okunamadı: {path}")
    return image


def imwrite_unicode(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if image is None or image.size == 0:
        raise ValueError(f"Boş görüntü kaydedilemez: {path}")
    ok, buffer = cv2.imencode(path.suffix or ".jpg", image)
    if not ok:
        raise RuntimeError(f"Görsel kodlanamadı: {path}")
    buffer.tofile(str(path))


def read_json(path: str | Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON nesnesi bekleniyordu: {path}")
    return payload


def save_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    """JSON'u önce aynı klasöre yazar, ardından tek dosya değişimi yapar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".tmp", prefix=".reference_",
            dir=str(path.parent), delete=False
        ) as file:
            temp_name = file.name
            json.dump(payload, file, ensure_ascii=False, indent=4)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_name, path)
    finally:
        if temp_name and Path(temp_name).exists():
            Path(temp_name).unlink()


def clear_old_files(folder: str | Path, patterns: tuple[str, ...] = ("SLOT_*.*",)) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for pattern in patterns:
        for path in folder.glob(pattern):
            if path.is_file():
                path.unlink()


def order_points(points: np.ndarray) -> np.ndarray:
    """Dört OBB köşesini tekrarsız, çevresel sıraya koyar.

    İlk köşe görüntü koordinatlarında en üstteki (eşitlikte en soldaki)
    noktadır. Başlangıç açısı modele göre farklı olsa da OCR yön düzeltme
    ve dört dönüşlü görüntü eşleştirme bunu ele alır.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    if not np.isfinite(pts).all():
        raise ValueError("OBB koordinatlarında geçersiz sayı var")
    if len(np.unique(pts, axis=0)) != 4:
        raise ValueError("OBB köşeleri birbirinden farklı değil")
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    quad = pts[np.argsort(angles)]
    if cv2.contourArea(quad) < 5:
        raise ValueError("OBB alanı çok küçük veya köşeleri geçersiz")
    start = int(np.lexsort((quad[:, 0], quad[:, 1]))[0])
    return np.roll(quad, -start, axis=0).astype(np.float32)


def crop_obb(image: np.ndarray, points: np.ndarray, padding: int = 20) -> np.ndarray | None:
    try:
        tl, tr, br, bl = order_points(points)
    except (ValueError, TypeError):
        return None
    width = int(round(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))))
    height = int(round(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))))
    if width < 5 or height < 5:
        return None
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(np.stack([tl, tr, br, bl]), dst)
    warped = cv2.warpPerspective(image, matrix, (width, height))
    if padding > 0:
        warped = cv2.copyMakeBorder(
            warped, padding, padding, padding, padding, cv2.BORDER_REPLICATE
        )
    return warped


def resize_for_ocr(image: np.ndarray, min_long_side: int = 1400, max_long_side: int = 2400) -> np.ndarray:
    """Eski 1400/2400 ayarlarını varsayılan olarak korur; CLI'dan ayarlanır."""
    h, w = image.shape[:2]
    side = max(h, w)
    if side == 0:
        raise ValueError("Boş OCR görüntüsü")
    if min_long_side > 0 and side < min_long_side:
        factor = min_long_side / side
        image = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
        side = max(image.shape[:2])
    if side > max_long_side:
        factor = max_long_side / side
        image = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    return image


_SHARPEN_KERNEL = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)


def prepare_ocr_image(crop: np.ndarray, min_long_side: int = 1400, max_long_side: int = 2400) -> np.ndarray:
    base = resize_for_ocr(crop, min_long_side, max_long_side)
    return np.ascontiguousarray(cv2.filter2D(base, -1, _SHARPEN_KERNEL))


def clean_text(text: Any) -> str:
    text = re.sub(r"\s+", " ", str(text))
    return text.strip(" -_|.,;:").strip()


def text_quality_score(text: str, conf: float) -> float:
    """Önceki raporla uyumlu kalite metriği; tek varyantta seçim yapılmaz."""
    text = clean_text(text)
    if not text:
        return -999.0
    letters = sum(char.isalpha() for char in text)
    digits = sum(char.isdigit() for char in text)
    alnum = sum(char.isalnum() for char in text)
    tokens = text.split()
    token_count = len(tokens)
    singles = sum(len(token) == 1 for token in tokens)
    weird = sum(not c.isalnum() and c not in " .,-:;()&'’•" for c in text)
    score = float(conf)
    score += min(letters, 80) * 0.01 + min(token_count, 10) * 0.03
    score -= singles / max(token_count, 1) * 0.35
    score -= weird / max(len(text), 1) * 0.50
    if letters < 3:
        score -= 0.70
    if alnum < 5:
        score -= 0.50
    if digits > letters and letters < 6:
        score -= 0.35
    return float(score)


def choose_mkldnn(device: str, mode: str = "auto") -> bool:
    """Paddle 3.3.x CPU/oneDNN-PIR hatasına karşı güvenli varsayılan.

    3.2.2'de oneDNN hızlandırması korunur. 3.3.x'te otomatik
    kapatılır; elle on/off seçilebilir. GPU için CPU oneDNN kullanılmaz.
    """
    if mode not in {"auto", "on", "off"}:
        raise ValueError(f"Geçersiz oneDNN modu: {mode}")
    if mode != "auto":
        return mode == "on"
    if not device.lower().startswith("cpu"):
        return False
    try:
        version_string = metadata.version("paddlepaddle")
    except metadata.PackageNotFoundError:
        try:
            version_string = metadata.version("paddlepaddle-gpu")
        except metadata.PackageNotFoundError:
            return False
    parts = re.match(r"(\d+)\.(\d+)", version_string)
    if not parts:
        return False
    return tuple(map(int, parts.groups())) != (3, 3)


def build_paddle_ocr(
    device: str = "cpu", recognition_batch_size: int = 8, mkldnn_mode: str = "auto"
):
    """Açıkça PP-OCRv6 small modellerini kullan; otomatik model değişimi yok."""
    from paddleocr import PaddleOCR

    mkldnn_enabled = choose_mkldnn(device, mkldnn_mode)
    if device.lower().startswith("cpu"):
        print(f"[OCR] CPU oneDNN={'açık' if mkldnn_enabled else 'kapalı'} "
              f"(mod={mkldnn_mode}); 3.3.x hatası için 3.2.2 CPU önerilir")
    try:
        return PaddleOCR(
            text_detection_model_name=OCR_DET_MODEL,
            text_recognition_model_name=OCR_REC_MODEL,
            device=device,
            enable_mkldnn=mkldnn_enabled,
            use_doc_orientation_classify=True,
            use_doc_unwarping=False,
            use_textline_orientation=True,
            text_rec_score_thresh=OCR_SCORE_THRESHOLD,
            text_recognition_batch_size=recognition_batch_size,
        )
    except Exception as exc:
        raise RuntimeError(
            "PP-OCRv6_small başlatılamadı. PaddleOCR/PaddlePaddle "
            f"{device} kurulumunu kontrol et. Orijinal hata: {exc}"
        ) from exc


def _is_onednn_pir_failure(exc: Exception) -> bool:
    return "ConvertPirAttribute2RuntimeAttribute" in str(exc) or (
        "onednn_instruction.cc" in str(exc)
        and "pir::" in str(exc)
    )


def validate_ocr_slots(slots: list[dict[str, Any]], *, origin: str) -> None:
    """OCR motoru hata verdiğinde bozuk referans/rapor yazılmasını önler.

    Metinsiz bazı kitaplara izin verilir: boş metin tek başına hata değildir.
    """
    failed = [row["slot_name"] for row in slots if row.get("ocr_error", False)]
    if failed:
        raise RuntimeError(
            f"{origin}: {len(failed)}/{len(slots)} kitapta OCR işlem hatası: "
            + ", ".join(failed[:10])
            + ". JSON raporu/referansı yazılmadı."
        )
    if slots and not any(str(row.get("text", "")).strip() for row in slots):
        raise RuntimeError(
            f"{origin}: {len(slots)} kitap crop edildi ancak hiçbirinde OCR metni "
            "bulunamadı. Model çalışıyor olabilir; crop görüntülerini, "
            "çözünürlüğü ve OCR ayarlarını kontrol et. Geçersiz referans/rapor yazılmadı."
        )


def _unwrap_ocr_result(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        data = result
    else:
        raw = getattr(result, "json", None)
        if callable(raw):
            raw = raw()
        if isinstance(raw, str):
            raw = json.loads(raw)
        if not isinstance(raw, dict):
            raw = result.to_dict() if hasattr(result, "to_dict") else {}
        data = raw if isinstance(raw, dict) else {}
    # PaddleOCR Pipeline Result.json genelde {"res": {...}} döndürür.
    while isinstance(data, dict) and isinstance(data.get("res"), dict):
        data = data["res"]
    return data


def extract_text_score_from_result(result: Any) -> tuple[str, float]:
    data = _unwrap_ocr_result(result)
    rec_texts = data.get("rec_texts", [])
    rec_scores = data.get("rec_scores", [])
    if isinstance(rec_texts, np.ndarray):
        rec_texts = rec_texts.tolist()
    if isinstance(rec_scores, np.ndarray):
        rec_scores = rec_scores.tolist()
    texts: list[str] = []
    scores: list[float] = []
    for i, raw in enumerate(rec_texts):
        if raw is None or not str(raw).strip():
            continue
        texts.append(str(raw))
        if i < len(rec_scores):
            try:
                scores.append(float(rec_scores[i]))
            except (TypeError, ValueError):
                pass
    return clean_text(" ".join(texts)), float(np.mean(scores)) if scores else 0.0


def _predict_ocr_results(ocr: Any, images: list[np.ndarray]) -> list[Any]:
    output = ocr.predict(images)
    if isinstance(output, (dict, str)) or hasattr(output, "json"):
        output = [output]
    results = list(output)
    if len(results) != len(images):
        raise RuntimeError(f"OCR {len(images)} görüntüye karşı {len(results)} sonuç döndürdü")
    return results


def run_ocr_batch(
    ocr: Any,
    slots: list[dict[str, Any]],
    batch_size: int = 8,
    min_long_side: int = 1400,
    max_long_side: int = 2400,
    *,
    denoise: str = "auto",
    debug_clean_dir: str | Path | None = None,
) -> None:
    """İlk OCR herkese, temiz görüntü ile tekrar yalnızca gürültülü olanlara.

    `text_raw` ilk OCR'yi korur. `text` seçilen varyantın orijinal metnidir;
    `match_text` yalnızca eşleştirme için gürültüsü filtrelenmiş hâlidir.
    Tekrar OCR başarısız olursa geçerli ilk OCR sonucu aynen korunur.
    """
    if batch_size < 1:
        raise ValueError("OCR batch boyutu >= 1 olmalı")
    if denoise not in ("auto", "off"):
        raise ValueError("denoise yalnızca auto/off olabilir")

    def predict_safe(chunk: list[dict[str, Any]], images: list[np.ndarray]) -> list[Any]:
        try:
            return _predict_ocr_results(ocr, images)
        except Exception as batch_error:
            if _is_onednn_pir_failure(batch_error):
                raise RuntimeError(
                    "Paddle CPU oneDNN/PIR hatası. paddlepaddle==3.2.2 CPU "
                    "veya --ocr-mkldnn off kullan."
                ) from batch_error
            print(f"[WARN] Toplu OCR başarısız: {batch_error}. Tek tek deneniyor.")
            results: list[Any] = []
            for row, image in zip(chunk, images):
                try:
                    single = _predict_ocr_results(ocr, [image])
                    results.append(single[0])
                except Exception as exc:
                    if _is_onednn_pir_failure(exc):
                        raise RuntimeError(
                            "Paddle CPU oneDNN/PIR hatası. paddlepaddle==3.2.2 CPU "
                            "veya --ocr-mkldnn off kullan."
                        ) from exc
                    print(f"[WARN] {row['slot_name']} OCR hatası: {exc}")
                    results.append(None)
            return results

    for start in range(0, len(slots), batch_size):
        chunk = slots[start:start + batch_size]
        images = [prepare_ocr_image(row["crop_img"], min_long_side, max_long_side) for row in chunk]
        first_results = predict_safe(chunk, images)

        retry_rows: list[dict[str, Any]] = []
        retry_images: list[np.ndarray] = []
        for row, result, image in zip(chunk, first_results, images):
            if result is None:
                text, conf = "", 0.0
                row["ocr_error"] = True
            else:
                try:
                    text, conf = extract_text_score_from_result(result)
                    row["ocr_error"] = False
                except Exception as exc:
                    print(f"[WARN] {row['slot_name']} OCR sonuç hatası: {exc}")
                    text, conf = "", 0.0
                    row["ocr_error"] = True
            analysis = analyze_binary_noise(text) if denoise == "auto" else None
            row.update({
                "ocr_mode": "sharpen",
                "ocr_conf": round(conf, 4),
                "ocr_quality_score": round(text_quality_score(text, conf), 4),
                "text": text,
                "text_raw": text,
                "match_text": analysis.filtered_text if analysis else text,
                "text_filtered": analysis.filtered_text if analysis else text,
                "ocr_first_pass_conf": round(conf, 4),
                "noise_detected": bool(analysis.detected) if analysis else False,
                "noise_token_count": analysis.binary_token_count if analysis else 0,
                "noise_removed_tokens": analysis.removed_token_count if analysis else 0,
                "ocr_retry_attempted": False,
                "ocr_retry_used": False,
            })
            # Hata vermemiş ve gerçekten yoğun ikili desen tespit edilmişse yeniden dene.
            if analysis and analysis.detected and not row["ocr_error"]:
                cleaned_image = suppress_light_background(image)
                row["ocr_retry_attempted"] = True
                retry_rows.append(row)
                retry_images.append(cleaned_image)
                if debug_clean_dir is not None:
                    debug_path = Path(debug_clean_dir) / f"{row['slot_name']}_clean.png"
                    imwrite_unicode(debug_path, cleaned_image)
                    row["ocr_clean_crop_file"] = str(debug_path)

        if retry_rows:
            # İkinci OCR tek/çoklu dönebilir; başarısızsa ilk sonuç korunur.
            try:
                retry_results = predict_safe(retry_rows, retry_images)
            except Exception as exc:
                print(f"[WARN] Gürültü temizleme OCR denemesi başarısız: {exc}. İlk OCR korundu.")
                retry_results = [None] * len(retry_rows)
            for row, result in zip(retry_rows, retry_results):
                if result is None:
                    continue
                try:
                    retry_text, retry_conf = extract_text_score_from_result(result)
                except Exception as exc:
                    print(f"[WARN] {row['slot_name']} temiz görüntü OCR sonucu okunamadı: {exc}")
                    continue
                row["ocr_retry_text"] = retry_text
                row["ocr_retry_conf"] = round(retry_conf, 4)
                if should_use_retry(row["text_raw"], row["ocr_first_pass_conf"], retry_text, retry_conf):
                    selection = analyze_binary_noise(retry_text)
                    row.update({
                        "text": retry_text,
                        "match_text": selection.filtered_text,
                        "text_filtered": selection.filtered_text,
                        "ocr_mode": "binary_suppress",
                        "ocr_conf": round(retry_conf, 4),
                        "ocr_quality_score": round(text_quality_score(retry_text, retry_conf), 4),
                        "ocr_retry_used": True,
                    })

        for row in chunk:
            print(f"[OCR] {row['slot_name']} | mod={row['ocr_mode']} | "
                  f"gürültü={row['noise_detected']} | "
                  f"Conf={row['ocr_conf']:.3f} | eşleştirme={row['match_text']}")


def build_yolo(model_path: str):
    from ultralytics import YOLO
    if model_path.lower() != "yolo26s-obb.pt" and not Path(model_path).is_file():
        raise FileNotFoundError(f"YOLO26s-OBB ağırlığı bulunamadı: {model_path}")
    try:
        model = YOLO(model_path)
    except Exception as exc:
        raise RuntimeError(f"YOLO modeli yüklenemedi: {exc}") from exc
    if getattr(model, "task", "obb") != "obb":
        raise ValueError("Seçilen ağırlık OBB modeli değil (task=obb olmalı)")
    return model


def get_obb_detections(result: Any) -> list[dict[str, Any]]:
    if result.obb is None or len(result.obb) == 0:
        return []
    corners = result.obb.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2)
    confs = result.obb.conf.cpu().numpy()
    classes = result.obb.cls.cpu().numpy()
    detections = []
    for points, conf, cls in zip(corners, confs, classes):
        detections.append({
            "points": points.astype(np.float32),
            "conf": float(conf),
            "class_id": int(cls),
            "center_x": float(np.mean(points[:, 0])),
            "center_y": float(np.mean(points[:, 1])),
        })
    detections.sort(key=lambda d: (float(np.min(d["points"][:, 0])), float(np.min(d["points"][:, 1]))))
    return detections


def detect_and_ocr_slots(
    model: Any, ocr: Any, image_path: str | Path, crop_dir: str | Path,
    *, yolo_device: str = "0", imgsz: int = 1280, conf: float = 0.60,
    ocr_batch_size: int = 8, ocr_min_side: int = 1400, ocr_max_side: int = 2400,
    relative_crop_base: str | Path | None = None,
    ocr_denoise: str = "auto", ocr_debug_clean_dir: str | Path | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, float]]:
    start = time.perf_counter()
    crop_dir = Path(crop_dir)
    crop_dir.mkdir(parents=True, exist_ok=True)
    image = imread_unicode(image_path)
    result = model.predict(source=image, imgsz=imgsz, conf=conf, device=yolo_device, verbose=False)[0]
    detection_time = time.perf_counter() - start
    detections = get_obb_detections(result)
    print(f"[INFO] Tespit edilen kitap/nesne sayısı: {len(detections)}")

    slot_rows = []
    for idx, detection in enumerate(detections, start=1):
        name = f"SLOT_{idx:03d}"
        points = detection["points"]
        crop = crop_obb(image, points, padding=20)
        if crop is None:
            print(f"[WARN] {name}: Geçersiz OBB, atlandı")
            continue
        path = crop_dir / f"{name}.jpg"
        imwrite_unicode(path, crop)
        display_path = str(path)
        if relative_crop_base is not None:
            display_path = path.relative_to(Path(relative_crop_base)).as_posix()
        slot_rows.append({
            "slot_name": name,
            "order_index": idx,
            "class_id": detection["class_id"],
            "center_x": round(detection["center_x"], 2),
            "center_y": round(detection["center_y"], 2),
            "yolo_conf": round(detection["conf"], 4),
            "slot_crop_file": display_path,
            "quad_points": points.tolist(),
            "crop_img": crop,
        })

    ocr_start = time.perf_counter()
    if slot_rows:
        run_ocr_batch(ocr, slot_rows, ocr_batch_size, ocr_min_side, ocr_max_side,
                      denoise=ocr_denoise, debug_clean_dir=ocr_debug_clean_dir)
    return image, slot_rows, {
        "detect_and_crop_seconds": round(ocr_start - start, 3),
        "ocr_seconds": round(time.perf_counter() - ocr_start, 3),
        "yolo_seconds": round(detection_time, 3),
    }


def serializable_slots(slots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: value for k, value in slot.items() if k not in ("crop_img", "features")} for slot in slots]
