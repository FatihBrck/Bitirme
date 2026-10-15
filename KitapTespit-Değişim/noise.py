# -*- coding: utf-8 -*-
"""OCR metnindeki dekoratif 0/1 gürültüsünü seçici temizler.

Kelimeler/kitap adları rastgele silinmez. Temizlik yalnızca aynı OCR
sonucunda çok sayıda saf ikili sayı token'ı saptandığında etkinleşir.
Ham metin ayrı saklanır. Görüntü temizliği yalnızca şüpheli slotlarda
ikinci, isteğe bağlı OCR denemesi içindir.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import cv2
import numpy as np

TOKEN_RE = re.compile(r"[^\s]+")
BINARY_RE = re.compile(r"[01]+")
WORD_RE = re.compile(r"[A-Za-zÇĞİÖŞÜçğıöşü]+", re.UNICODE)


@dataclass(frozen=True)
class NoiseResult:
    detected: bool
    filtered_text: str
    binary_token_count: int
    removed_token_count: int
    noise_ratio: float


def _token_parts(token: str) -> list[str]:
    # '01.10' gibi birleşik OCR gürültülerini iki ayrı token gibi ele al.
    return [p for p in re.split(r"[^\w]+", token, flags=re.UNICODE) if p]


def analyze_binary_noise(text: str) -> NoiseResult:
    """Saf ikili token yoğunluğu yoksa hiçbir karakteri kaldırmaz.

    En az 4 adet 0/1 token ve tüm sayısal/alfasayısal karakterlerin
    >=%%30'unun bu token'lardan gelmesi gerekir. Böylece 1984,
    101 Dalmaçyalı, 0 ve 1 gibi gerçek başlıklar korunur.
    """
    text = str(text or "").strip()
    if not text:
        return NoiseResult(False, "", 0, 0, 0.0)
    raw_tokens = TOKEN_RE.findall(text)
    pieces = [p for token in raw_tokens for p in _token_parts(token)]
    binary_pieces = [p for p in pieces if len(p) >= 2 and BINARY_RE.fullmatch(p)]
    binary_chars = sum(len(p) for p in binary_pieces)
    alnum_chars = sum(len(p) for p in pieces if p.isalnum())
    ratio = binary_chars / max(alnum_chars, 1)
    noisy = len(binary_pieces) >= 4 and ratio >= 0.30
    if not noisy:
        return NoiseResult(False, text, len(binary_pieces), 0, round(ratio, 4))

    kept = []
    removed = 0
    for token in raw_tokens:
        parts = _token_parts(token)
        # Tekrarlanan 01.10 gibi kombinasyonlar veya salt ikili rakamlar;
        # sadece gürültü tespiti güvenli olduğunda kaldırılır.
        if parts and all(BINARY_RE.fullmatch(p) for p in parts):
            removed += 1
            continue
        kept.append(token)
    filtered = re.sub(r"\s+", " ", " ".join(kept)).strip()
    # Kitabın ayırt edici metni tamamen kaybolduysa gürültü temizliği uygulama.
    if len("".join(WORD_RE.findall(filtered))) < 3:
        return NoiseResult(True, text, len(binary_pieces), 0, round(ratio, 4))
    return NoiseResult(True, filtered, len(binary_pieces), removed, round(ratio, 4))


def suppress_light_background(image: np.ndarray) -> np.ndarray:
    """Koyu harfleri koruyup açık dekoratif desenleri bastırmayı dener.

    Otsu eşiği görüntüye göre hesaplanır. Yalnızca ikinci OCR varyantı
    olarak kullanılır; orijinal görüntüyü değiştirmez.
    """
    if image is None or image.size == 0:
        raise ValueError("Boş görüntü temizlenemez")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    # Keskinleştirme yerine hafif median; JPEG artifaktlarını hafifletir.
    smooth = cv2.medianBlur(gray, 3)
    _, binary = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return np.ascontiguousarray(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR))


def candidate_score(text: str, confidence: float) -> float:
    """İkinci OCR'yi yalnızca AYIRT EDİCİ metin gerçekten iyileşirse seç.

    OCR güveni ikili desenlerde yanıltıcı biçimde yüksek olabileceği için
    token gürültüsü cezalandırılır, anlamlı harfler ödüllendirilir.
    """
    result = analyze_binary_noise(text)
    letters = len("".join(WORD_RE.findall(result.filtered_text)))
    words = [w for w in WORD_RE.findall(result.filtered_text) if len(w) >= 3]
    return (
        min(max(float(confidence), 0.0), 1.0) * 0.40
        + min(letters, 80) * 0.010
        + min(len(words), 12) * 0.035
        - min(result.binary_token_count, 30) * 0.016
    )


def _word_key(word: str) -> str:
    normalized = unicodedata.normalize("NFKD", word.casefold())
    without_marks = "".join(c for c in normalized if not unicodedata.combining(c))
    return without_marks.replace("ı", "i")


def should_use_retry(original: str, original_conf: float, retry: str, retry_conf: float) -> bool:
    if not retry or not retry.strip():
        return False
    old_clean = analyze_binary_noise(original).filtered_text
    new_clean = analyze_binary_noise(retry).filtered_text
    if old_clean == new_clean:
        return False
    # Orijinalde okunan önemli sözcüklerin tamamını kaybetmemeliyiz.
    old_words = {_word_key(w) for w in WORD_RE.findall(old_clean) if len(w) >= 4}
    new_words = {_word_key(w) for w in WORD_RE.findall(new_clean) if len(w) >= 4}
    if old_words and not (old_words & new_words):
        return False
    return candidate_score(retry, retry_conf) > candidate_score(original, original_conf) + 0.08
