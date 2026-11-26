"""Tải, chia và tạo cửa sổ dữ liệu thời tiết Hà Nội cho Task 2 (dự báo nhiệt độ 5 giờ tới)."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.config import (END_DATE, FEATURES, HORIZON, LATITUDE, LONGITUDE, N_STEPS, START_DATE, TARGET,
                        TIMEZONE, TRAIN_RATIO, VAL_RATIO, WEATHER_CSV, WEATHER_VARIABLES)


def download_weather(path=WEATHER_CSV):
    # Open-Meteo Historical Weather API (ERA5), trả về CSV, không cần API key
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LATITUDE}&longitude={LONGITUDE}"
        f"&start_date={START_DATE}&end_date={END_DATE}"
        f"&hourly={','.join(WEATHER_VARIABLES)}"
        f"&timezone={TIMEZONE.replace('/', '%2F')}&format=csv"
    )
    # 3 dòng đầu là thông tin vị trí, bảng dữ liệu bắt đầu từ dòng thứ 4
    df = pd.read_csv(url, skiprows=3)
    df.columns = ["time"] + list(WEATHER_VARIABLES.values())
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return url


def load_weather(path=WEATHER_CSV):
    if not path.exists():
        download_weather(path)
    return pd.read_csv(path, parse_dates=["time"])


def add_features(df):
    # hướng gió (độ) -> sin, cos để 359° và 1° nằm gần nhau
    df = df.copy()
    angle = np.deg2rad(df["wind_dir"])
    df["wind_dir_sin"] = np.sin(angle)
    df["wind_dir_cos"] = np.cos(angle)
    return df.drop(columns="wind_dir")


def split_by_time(df):
    # chia theo thứ tự thời gian, không xáo trộn: train = quá khứ, test = giai đoạn mới nhất
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()


def fit_scaler(train_df, columns=FEATURES):
    # MinMaxScaler chỉ học min/max trên tập train để không dùng thông tin của val/test
    return MinMaxScaler().fit(train_df[columns])


def make_windows(values, target_idx, n_steps=N_STEPS, horizon=HORIZON):
    """Tạo các cặp (X, y) bằng cửa sổ trượt, giống hàm split_sequences trong notebook encoder của cô.

    Thời điểm hiện tại t là giờ cuối cùng của cửa sổ input values[i : i + n_steps]:
    X = các giờ t-n_steps+1, ..., t (mọi biến); y = giá trị đích ở t+1, ..., t+horizon.
    """
    X, y = [], []
    for i in range(len(values) - n_steps - horizon + 1):
        X.append(values[i:i + n_steps, :])
        y.append(values[i + n_steps:i + n_steps + horizon, target_idx])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def prepare_forecast_data(n_steps=N_STEPS, horizon=HORIZON, features=FEATURES, target=TARGET):
    """Đọc dữ liệu, chia train/val/test, chuẩn hoá và tạo cửa sổ cho cả ba tập."""
    df = add_features(load_weather())
    splits = dict(zip(["train", "val", "test"], split_by_time(df)))
    scaler = fit_scaler(splits["train"], features)
    target_idx = features.index(target)
    data = {"scaler": scaler, "target_idx": target_idx, "features": features}
    for name, part in splits.items():
        values = scaler.transform(part[features])
        X, y = make_windows(values, target_idx, n_steps, horizon)
        data[f"X_{name}"], data[f"y_{name}"] = X, y
        # thời điểm hiện tại t của từng mẫu (giờ cuối của cửa sổ input), dùng để vẽ biểu đồ
        data[f"t_{name}"] = part["time"].to_numpy()[n_steps - 1:n_steps - 1 + len(X)]
    return data


def inverse_target(y_scaled, scaler, target_idx):
    # đổi giá trị đích từ thang [0, 1] về lại °C
    return np.asarray(y_scaled) * scaler.data_range_[target_idx] + scaler.data_min_[target_idx]
