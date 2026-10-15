# -*- coding: utf-8 -*-
"""Model indirmeden çalışan regresyon / smoke testleri."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

import common
import features
import search
import ölçüm


def test_turkish_normalization():
    assert features.normalize_text_for_match("MARTIN EDEN") == features.normalize_text_for_match("MARTİN EDEN")
    assert features.text_similarity_normalized("martin eden", "martin eden") == 1.0


def test_45_degree_obb_unique_corners():
    quad = np.array([[100, 40], [160, 100], [100, 160], [40, 100]], np.float32)
    sorted_points = common.order_points(quad)
    assert len(np.unique(sorted_points, axis=0)) == 4
    assert cv2.contourArea(sorted_points) > 7000
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    crop = common.crop_obb(img, quad)
    assert crop is not None and crop.size > 0


def test_ocr_json_and_batch_without_files():
    class FakeResult:
        @property
        def json(self):
            return {"res": {"rec_texts": ["KÜÇÜK", "PRENS"], "rec_scores": [0.90, 0.80]}}

    class FakeOCR:
        def predict(self, images):
            assert isinstance(images, list) and all(isinstance(x, np.ndarray) for x in images)
            return [FakeResult() for _ in images]

    arr = np.zeros((50, 180, 3), dtype=np.uint8)
    slots = [{"slot_name": "SLOT_001", "crop_img": arr},
             {"slot_name": "SLOT_002", "crop_img": arr}]
    common.run_ocr_batch(FakeOCR(), slots, batch_size=2, min_long_side=0)
    assert slots[0]["text"] == "KÜÇÜK PRENS" and slots[0]["ocr_conf"] == 0.85
    assert slots[1]["ocr_error"] is False


def test_global_matching_beats_greedy():
    refs = [{"slot_name": "R1", "order_index": 1}, {"slot_name": "R2", "order_index": 2}]
    cur = [{"slot_name": "C1", "order_index": 1}, {"slot_name": "C2", "order_index": 2}]
    scores = {("R1", "C1"): .90, ("R1", "C2"): .89,
              ("R2", "C1"): .88, ("R2", "C2"): .10}

    def fake_compare(ref, now, extractor):
        return {"reference_slot": ref["slot_name"], "reference_order": ref["order_index"],
                "current_slot": now["slot_name"], "current_order": now["order_index"],
                "final_score": scores[ref["slot_name"], now["slot_name"]]}

    with mock.patch.object(search, "compare_pair", fake_compare):
        matched, unref, uncur, pairs = search.global_match(refs, cur, object(), .42, .65)
    assert {(p["reference_slot"], p["current_slot"]) for p in matched} == {
        ("R1", "C2"), ("R2", "C1")}
    assert len(unref) == len(uncur) == 0 and len(pairs) == 4


def test_no_edges_not_perfect_match():
    extractor = features.ImageFeatureExtractor()
    a = extractor.extract(np.zeros((100, 20, 3), np.uint8))
    b = extractor.extract(np.full((100, 20, 3), 255, np.uint8))
    _, details = extractor.score(a, b)
    assert details["edge_score"] == 0.0


def test_reference_version_and_unchanged_rerun(tmp_path: Path):
    img = tmp_path / "test.jpg"
    common.imwrite_unicode(img, np.zeros((200, 200, 3), np.uint8))
    out = tmp_path / "ref"

    def fake_detect(model, ocr, image_path, crop_dir, **kwargs):
        crop_dir.mkdir(parents=True, exist_ok=True)
        crop_path = crop_dir / "SLOT_001.jpg"
        common.imwrite_unicode(crop_path, np.zeros((20, 30, 3), np.uint8))
        return np.zeros((200, 200, 3), np.uint8), [{
            "slot_name": "SLOT_001", "order_index": 1, "text": "Kitap 1", "ocr_conf": 0.9,
            "slot_crop_file": crop_path.relative_to(out).as_posix(),
            "crop_img": np.zeros((20, 30, 3), np.uint8),
        }], {"yolo_seconds": .1, "ocr_seconds": .1}

    argv = ["ölçüm.py", "--image", str(img), "--output", str(out)]
    with mock.patch.object(sys, "argv", argv), \
         mock.patch.object(ölçüm, "build_yolo", return_value=object()), \
         mock.patch.object(ölçüm, "build_paddle_ocr", return_value=object()), \
         mock.patch.object(ölçüm, "detect_and_ocr_slots", side_effect=fake_detect):
        ölçüm.main()
    reference = out / common.REFERENCE_ORDER_JSON
    payload = json.loads(reference.read_text(encoding="utf-8"))
    old_crop = out / payload["slots"][0]["slot_crop_file"]
    assert old_crop.is_file()
    # İkinci komut referansı bozmadan çıkar.
    with mock.patch.object(sys, "argv", argv), \
         mock.patch.object(ölçüm, "build_yolo", side_effect=AssertionError("not called")):
        ölçüm.main()
    assert json.loads(reference.read_text(encoding="utf-8")) == payload
    assert old_crop.is_file()


def test_end_to_end_mock_pipeline(tmp_path: Path):
    image = tmp_path / "control.jpg"
    reference_dir, output_dir = tmp_path / "reference", tmp_path / "output"
    reference_dir.mkdir()
    img = np.zeros((300, 400, 3), np.uint8)
    cv2.putText(img, "BOOK", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    common.imwrite_unicode(image, img)
    ref_crop = reference_dir / "slot_crops" / "REF_test" / "SLOT_001.jpg"
    common.imwrite_unicode(ref_crop, img[30:260, 20:310])
    common.save_json_atomic(reference_dir / common.REFERENCE_ORDER_JSON, {
        "slots": [{"slot_name": "SLOT_001", "order_index": 1,
                   "slot_crop_file": "slot_crops/REF_test/SLOT_001.jpg", "text": "BOOK",
                   "ocr_conf": 0.9, "ocr_quality_score": 1.0}],
        "total_books": 1,
    })

    def fake_detect(model, ocr, image_path, crop_dir, **kwargs):
        crop_dir.mkdir(parents=True, exist_ok=True)
        crop_img = common.imread_unicode(ref_crop)
        path = crop_dir / "SLOT_001.jpg"
        common.imwrite_unicode(path, crop_img)
        return common.imread_unicode(image), [{
            "slot_name": "SLOT_001", "order_index": 1, "text": "BOOK", "ocr_conf": .9,
            "ocr_quality_score": 1.0, "slot_crop_file": str(path), "ocr_mode": "sharpen",
            "crop_img": crop_img, "quad_points": [[20, 30], [310, 30], [310, 260], [20, 260]],
        }], {"yolo_seconds": 0.1, "detect_and_crop_seconds": 0.2, "ocr_seconds": 0.3}

    argv = ["search.py", "--image", str(image), "--reference", str(reference_dir),
            "--output", str(output_dir)]
    with mock.patch.object(sys, "argv", argv), \
         mock.patch.object(search, "build_yolo", return_value=object()), \
         mock.patch.object(search, "build_paddle_ocr", return_value=object()), \
         mock.patch.object(search, "detect_and_ocr_slots", side_effect=fake_detect):
        search.main()
    payload = json.loads((output_dir / "raf_kontrol_eslestirme_sonucu.json").read_text(encoding="utf-8"))
    assert len(payload["matches"]) == 1 and payload["matches"][0]["status"] == "YERİNDE"
    assert len(payload["all_pair_scores"]) == 1
    assert (output_dir / "raf_kontrol_raporu.txt").is_file()
    assert (output_dir / "raf_kontrol_annotated.jpg").is_file()


def test_global_unmatched_and_empty_current():
    refs = [{"slot_name": "R1", "order_index": 1}]
    assert search.global_match(refs, [], object(), 0.42, 0.65) == ([], refs, [], [])
    cur = [{"slot_name": "C1", "order_index": 1}]

    def too_low(r, c, extractor):
        return {"reference_slot": "R1", "reference_order": 1, "current_slot": "C1",
                "current_order": 1, "final_score": .10}

    with mock.patch.object(search, "compare_pair", too_low):
        matches, missing_refs, missing_currents, pairs = search.global_match(refs, cur, object(), .42, .65)
    assert not matches and missing_refs == refs and missing_currents == cur and len(pairs) == 1


def test_feature_cache_no_reextract_on_pair_comparisons():
    picture = np.zeros((200, 90, 3), np.uint8)
    cv2.putText(picture, "ABC", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    refs = [{"slot_name": "R1", "order_index": 1, "text": "ABC", "ocr_conf": .9,
             "crop_img": picture}]
    current = [{"slot_name": f"C{i}", "order_index": i, "text": "ABC", "ocr_conf": .9,
                "crop_img": picture} for i in (1, 2, 3)]
    extractor = features.ImageFeatureExtractor()
    with mock.patch.object(extractor, "extract", wraps=extractor.extract) as call:
        features.add_cached_features(refs, extractor, rotate=False)
        features.add_cached_features(current, extractor, rotate=True)
        assert call.call_count == 1 + 3 * 4
        search.global_match(refs, current, extractor, .42, .65)
        assert call.call_count == 13


def test_binary_background_noise_removed_but_titles_are_preserved():
    from noise import analyze_binary_noise
    original = ("DESIFRE! DEHA 00 01101000 011 01 00100000 0110 0010 "
                "10 01101001 0111 01101000 0110 00100000 01.10 00101 0010 MARTI")
    cleaned = analyze_binary_noise(original)
    assert cleaned.detected
    assert cleaned.filtered_text == "DESIFRE! DEHA MARTI"
    assert cleaned.binary_token_count > 10
    for valid in ("1984 George Orwell", "101 Dalmaçyalı", "0 1 Kitap", "Sayı 001"):  # sayı içeren kitaplar
        result = analyze_binary_noise(valid)
        assert not result.detected
        assert result.filtered_text == valid


def test_real_desifre_pair_scores_above_existing_threshold():
    from noise import analyze_binary_noise
    ref = "DESIFRE! MAIJIA JUU"
    current = ("DESIFRE! DEHA 00 01101000 011 01 00100000 0110 0010 "
               "10 01101001 0111 01101000 0110 00100000 01.10 00101 0010 MARTI")
    old_score = features.text_similarity_normalized(
        features.normalize_text_for_match(ref), features.normalize_text_for_match(current))
    new_score = features.text_similarity_normalized(
        features.normalize_text_for_match(ref),
        features.normalize_text_for_match(analyze_binary_noise(current).filtered_text))
    assert new_score > old_score
    assert new_score * .45 + .5782 * .5 + .8927 * .05 >= .42


def test_selective_ocr_only_noisy_slot_retried_and_raw_preserved():
    class FakeOCR:
        def __init__(self):
            self.calls = []

        def predict(self, images):
            assert isinstance(images, list)
            self.calls.append(images)
            if len(self.calls) == 1:
                return [
                    {"rec_texts": ["DESIFRE!", "DEHA", "01101000 011 01 00100000 0110 0010 10 01101001 0111", "MARTI"],
                     "rec_scores": [0.97, 0.98, 0.99, 0.96]},
                    {"rec_texts": ["1984", "George Orwell"], "rec_scores": [0.93, 0.94]},
                ]
            if len(self.calls) == 2:
                assert len(images) == 1
                # Binary ikinci OCR aynı üç kanallı numpy görüntü alır.
                assert images[0].ndim == 3 and images[0].shape[-1] == 3
                return [{"rec_texts": ["DESIFRE", "DEHA", "MAI JIA", "MARTI"],
                         "rec_scores": [0.99, 0.98, 0.96, 0.95]}]
            raise AssertionError("Beklenmedik OCR çağrısı")

    book = np.full((210, 80, 3), 222, dtype=np.uint8)
    cv2.putText(book, "B", (5, 95), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
    slots = [{"slot_name": "SLOT_001", "crop_img": book},
             {"slot_name": "SLOT_002", "crop_img": book}]
    ocr = FakeOCR()
    common.run_ocr_batch(ocr, slots, batch_size=2, min_long_side=0)
    assert len(ocr.calls) == 2
    assert slots[0]["ocr_retry_attempted"]
    assert slots[0]["ocr_retry_used"]
    assert slots[0]["text_raw"].startswith("DESIFRE!")
    assert slots[0]["text"] == "DESIFRE DEHA MAI JIA MARTI"
    assert slots[0]["match_text"] == slots[0]["text"]
    assert not slots[1]["ocr_retry_attempted"]
    assert slots[1]["text"] == "1984 George Orwell"


def test_denoise_off_disables_second_pass():
    class FakeOCR:
        def __init__(self):
            self.calls = 0

        def predict(self, images):
            self.calls += 1
            return [{"rec_texts": ["DESIFRE 00100000 0110 0010 01101001 0111 01101000 0110 DEHA"],
                     "rec_scores": [0.9]}]

    ocr = FakeOCR()
    slots = [{"slot_name": "SLOT_001", "crop_img": np.zeros((100, 200, 3), np.uint8)}]
    common.run_ocr_batch(ocr, slots, 1, 0, 2400, denoise="off")
    assert ocr.calls == 1
    assert slots[0]["match_text"] == slots[0]["text"]
    assert not slots[0]["ocr_retry_attempted"]


def test_retry_failure_preserves_first_ocr_and_does_not_mark_error():
    class FakeOCR:
        def __init__(self):
            self.calls = 0

        def predict(self, images):
            self.calls += 1
            if self.calls == 1:
                return [{"rec_texts": ["DESIFRE 00100000 0110 0010 01101001 0111 01101000 0110 DEHA"],
                         "rec_scores": [0.9]}]
            raise RuntimeError("Test second pass failure")

    ocr = FakeOCR()
    slots = [{"slot_name": "SLOT_001", "crop_img": np.zeros((100, 200, 3), np.uint8)}]
    common.run_ocr_batch(ocr, slots, 1, 0, 2400)
    assert ocr.calls >= 2
    assert slots[0]["text_raw"] == slots[0]["text"]
    assert slots[0]["match_text"] == "DESIFRE DEHA"
    assert not slots[0]["ocr_error"] and slots[0]["ocr_retry_attempted"]
    assert not slots[0]["ocr_retry_used"]


def test_legacy_reference_noise_processed_without_rebuilding():
    image = np.zeros((200, 50, 3), np.uint8)
    slots = [{"slot_name": "SLOT_001", "crop_img": image, "text":
              "DESIFRE DEHA 00 01101000 011 01 00100000 0110 0010 10 MARTI"}]
    extractor = features.ImageFeatureExtractor()
    features.add_cached_features(slots, extractor, rotate=False)
    assert slots[0]["normalized_text"] == "desifre deha marti"
