"""Đọc nhãn, cắt vùng số và chuẩn bị ảnh công tơ UFPR-AMR cho Task 3."""
import io
import re
import zipfile

import keras
import numpy as np
import pandas as pd
from keras import layers
from PIL import Image

from src.config import AMR_CACHE, AMR_SPLITS, AMR_ZIP, CROP_MARGIN, IMG_H, IMG_W, NUM_DIGITS


def read_annotations(zip_path=AMR_ZIP):
    """Đọc toàn bộ file .txt trong zip thành một bảng: mỗi dòng là một ảnh."""
    rows = []
    with zipfile.ZipFile(zip_path) as z:
        for name in sorted(z.namelist()):
            parts = name.split("/")
            if len(parts) != 3 or not name.endswith(".txt") or parts[1] not in AMR_SPLITS:
                continue
            text = z.read(name).decode("utf-8")
            x, y, w, h = map(int, re.search(r"position: (\d+) (\d+) (\d+) (\d+)", text).groups())
            rows.append({
                "split": AMR_SPLITS[parts[1]],
                "image": name[:-4] + ".jpg",
                "camera": re.search(r"camera: (.*)", text).group(1).strip(),
                "reading": re.search(r"reading: (\d+)", text).group(1),  # giữ dạng chuỗi để không mất số 0 ở đầu
                "x": x, "y": y, "w": w, "h": h,
            })
    return pd.DataFrame(rows)


def crop_counter(z, row, margin=CROP_MARGIN, size=(IMG_W, IMG_H)):
    """Cắt vùng 5 chữ số (nới thêm một chút lề) rồi đưa về kích thước cố định 64 x 256."""
    img = Image.open(io.BytesIO(z.read(row["image"])))
    # ảnh gốc rất lớn (~3000 x 4000): giải mã JPEG ở 1/2 kích thước cho nhanh, đủ nét cho ảnh 64 x 256
    full_w, full_h = img.size
    img.draft("RGB", (full_w // 2, full_h // 2))
    scale = img.size[0] / full_w
    mx, my = int(margin[0] * row["w"]), int(margin[1] * row["h"])
    box = (row["x"] - mx, row["y"] - my, row["x"] + row["w"] + mx, row["y"] + row["h"] + my)
    box = tuple(int(round(v * scale)) for v in box)
    return img.convert("RGB").crop(box).resize(size, Image.BILINEAR)


def reading_to_digits(reading):
    # "07317" -> [0, 7, 3, 1, 7]
    return [int(ch) for ch in reading]


def digits_to_value(digits):
    # [[0, 7, 3, 1, 7], ...] -> [7317, ...] (giá trị kWh dùng cho R², NSE, MAE, RMSE)
    digits = np.asarray(digits)
    return digits @ (10 ** np.arange(NUM_DIGITS - 1, -1, -1))


def build_cache(zip_path=AMR_ZIP, cache_path=AMR_CACHE):
    """Cắt vùng số của cả 2.000 ảnh một lần và lưu thành file .npz (vài phút)."""
    meta = read_annotations(zip_path)
    arrays = {}
    with zipfile.ZipFile(zip_path) as z:
        for split in ["train", "val", "test"]:
            part = meta[meta["split"] == split]
            arrays[f"x_{split}"] = np.stack([np.asarray(crop_counter(z, r)) for _, r in part.iterrows()]).astype(np.uint8)
            arrays[f"y_{split}"] = np.array([reading_to_digits(r) for r in part["reading"]], dtype=np.int64)
            arrays[f"reading_{split}"] = part["reading"].to_numpy(dtype=str)
            arrays[f"image_{split}"] = part["image"].to_numpy(dtype=str)
            arrays[f"camera_{split}"] = part["camera"].to_numpy(dtype=str)
    np.savez_compressed(cache_path, **arrays)
    return meta


def load_counters(cache_path=AMR_CACHE):
    """Đọc ảnh vùng số đã cắt (tạo cache nếu chưa có). Ảnh dạng uint8 (số ảnh, 64, 256, 3), nhãn (số ảnh, 5)."""
    if not cache_path.exists():
        build_cache(cache_path=cache_path)
    with np.load(cache_path) as f:
        return {k: f[k] for k in f.files}


def make_augmenter():
    """Tăng cường dữ liệu cho ảnh vùng số, chỉ có tác dụng khi train.

    Không lật ảnh (chữ số bị lật không còn là chữ số), chỉ xoay rất nhẹ, dịch, phóng to/thu nhỏ,
    đổi độ sáng và độ tương phản (ảnh chụp trong kho với ánh sáng khác nhau).
    """
    return keras.Sequential([
        layers.RandomRotation(0.01, fill_mode="nearest"),               # tối đa ±3.6°
        layers.RandomTranslation(0.08, 0.04, fill_mode="nearest"),      # dịch tối đa 8 % chiều cao, 4 % chiều rộng
        layers.RandomZoom((-0.1, 0.1), fill_mode="nearest"),
        layers.RandomBrightness(0.2, value_range=(0, 255)),
        layers.RandomContrast(0.3),
    ], name="augmentation")
