# -*- coding: utf-8 -*-
"""Raf kitap görsellerini bir kez özellik çıkararak karşılaştırır.

İlk sürümün histogram/pHash/Canny/ORB bilgisini korur. Edge benzerliği,
boş kenar haritalarının yanlışlıkla %100 eşleşmesini önlemek için toleranslı
precision/recall ile hesaplanır. Skorlar eski sürümle tam uyumlu değildir.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

import cv2
import numpy as np

from noise import analyze_binary_noise


@dataclass(slots=True)
class ImageFeatures:
    hist: np.ndarray
    phash: np.ndarray
    edge: np.ndarray
    dilated_edge: np.ndarray
    edge_count: int
    descriptors: np.ndarray | None
    keypoint_count: int


TR_TRANSLATE = str.maketrans({"ç": "c", "ğ": "g", "ı": "i", "ö": "o", "ş": "s", "ü": "u"})


def normalize_text_for_match(text: str) -> str:
    # Unicode NFKD: "İ" + lower() -> "i" + combining dot, işareti kaldırılır.
    lowered = str(text).casefold()
    decomposed = unicodedata.normalize("NFKD", lowered)
    without_marks = "".join(c for c in decomposed if not unicodedata.combining(c))
    without_marks = without_marks.translate(TR_TRANSLATE)
    without_marks = re.sub(r"[^a-z0-9 ]+", " ", without_marks)
    return re.sub(r"\s+", " ", without_marks).strip()


def text_similarity_normalized(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    seq_score = SequenceMatcher(None, a, b).ratio()
    tokens_a, tokens_b = set(a.split()), set(b.split())
    overlap = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    small = min(len(tokens_a), len(tokens_b))
    jaccard = overlap / max(union, 1)
    containment = overlap / max(small, 1)
    return float(np.clip(seq_score * 0.60 + jaccard * 0.25 + containment * 0.15, 0, 1))


def rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
    if angle == 0:
        return image
    codes = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    return cv2.rotate(image, codes[angle])


def prepare_for_compare(image: np.ndarray, size: int = 256) -> np.ndarray:
    """Görüntüyü deforme etmeden 256x256 alana, kenar rengini uzatarak sığdır."""
    if image is None or image.size == 0:
        raise ValueError("Karşılaştırma görüntüsü boş")
    h, w = image.shape[:2]
    scale = min(size / w, size / h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    dx, dy = size - new_w, size - new_h
    return cv2.copyMakeBorder(resized, dy // 2, dy - dy // 2, dx // 2, dx - dx // 2,
                              cv2.BORDER_REPLICATE)


class ImageFeatureExtractor:
    def __init__(self, nfeatures: int = 700):
        self.orb = cv2.ORB_create(nfeatures=nfeatures)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def extract(self, image: np.ndarray) -> ImageFeatures:
        image = prepare_for_compare(image)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray32 = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        dct = cv2.dct(np.float32(gray32))
        coeffs = dct[1:9, 1:9]
        phash = (coeffs > np.median(coeffs)).reshape(-1)

        edges = cv2.Canny(cv2.equalizeHist(gray), 80, 180) > 0
        kernel = np.ones((3, 3), dtype=np.uint8)
        dilated = cv2.dilate(edges.astype(np.uint8), kernel, iterations=1) > 0
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        return ImageFeatures(hist, phash, edges, dilated, int(np.count_nonzero(edges)),
                             descriptors, len(keypoints))

    def score(self, reference: ImageFeatures, current: ImageFeatures) -> tuple[float, dict[str, float]]:
        corr = float(cv2.compareHist(reference.hist, current.hist, cv2.HISTCMP_CORREL))
        hist_score = float(np.clip((corr + 1) / 2, 0, 1))
        phash_score = float(1 - np.count_nonzero(reference.phash != current.phash) / len(reference.phash))

        if reference.edge_count == 0 or current.edge_count == 0:
            edge_score = 0.0
        else:
            coverage_ref = np.count_nonzero(reference.edge & current.dilated_edge) / reference.edge_count
            coverage_cur = np.count_nonzero(current.edge & reference.dilated_edge) / current.edge_count
            edge_score = float(np.clip((coverage_ref + coverage_cur) / 2, 0, 1))

        orb_score = 0.0
        if (reference.descriptors is not None and current.descriptors is not None
                and reference.keypoint_count >= 5 and current.keypoint_count >= 5):
            matches = self.matcher.match(reference.descriptors, current.descriptors)
            good = sum(match.distance < 75 for match in matches)
            orb_score = float(np.clip(good / min(reference.keypoint_count, current.keypoint_count), 0, 1))

        score = float(np.clip(hist_score * 0.35 + phash_score * 0.25
                              + edge_score * 0.20 + orb_score * 0.20, 0, 1))
        return score, {
            "hist_score": round(hist_score, 4),
            "phash_score": round(phash_score, 4),
            "edge_score": round(edge_score, 4),
            "orb_score": round(orb_score, 4),
        }


def add_cached_features(slots: list[dict[str, Any]], extractor: ImageFeatureExtractor, *, rotate: bool) -> None:
    """Referans 1 açı, kontrol 4 açı; diğer slotlarla eşleşirken yeniden çıkarılmaz."""
    for slot in slots:
        slot["normalized_text"] = normalize_text_for_match(
            slot.get("match_text", analyze_binary_noise(slot.get("text", "")).filtered_text)
        )
        image = slot.get("crop_img")
        if image is None:
            slot["features"] = {}
            continue
        angles = (0, 90, 180, 270) if rotate else (0,)
        slot["features"] = {
            angle: extractor.extract(rotate_image(image, angle)) for angle in angles
        }
