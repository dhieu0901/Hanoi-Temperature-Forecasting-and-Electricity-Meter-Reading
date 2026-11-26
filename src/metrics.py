"""Các chỉ số đánh giá theo yêu cầu đề: R², NSE, MAE, RMSE.

Theo slide của cô (Chương 2.1, slide 17):
- R² là bình phương hệ số tương quan Pearson giữa giá trị thực và giá trị dự báo,
- NSE = 1 - SSE / SST (đây chính là sklearn.metrics.r2_score, nên không dùng r2_score cho R²).
"""
import numpy as np
import pandas as pd


def r2_pearson(y_true, y_pred):
    y_true, y_pred = np.ravel(y_true), np.ravel(y_pred)
    r = np.corrcoef(y_true, y_pred)[0, 1]
    return float(r ** 2)


def nse(y_true, y_pred):
    y_true, y_pred = np.ravel(y_true), np.ravel(y_pred)
    return float(1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.ravel(y_true) - np.ravel(y_pred))))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.ravel(y_true) - np.ravel(y_pred)) ** 2)))


def regression_metrics(y_true, y_pred):
    return {"R2": r2_pearson(y_true, y_pred), "NSE": nse(y_true, y_pred),
            "MAE": mae(y_true, y_pred), "RMSE": rmse(y_true, y_pred)}


def meter_metrics(digits_true, digits_pred):
    """Chỉ số cho Task 3, digits có dạng (số ảnh, 5).

    - digit accuracy: tỉ lệ chữ số đoán đúng (trên tổng số 5 x số ảnh chữ số),
    - reading accuracy: tỉ lệ ảnh đoán đúng cả 5 chữ số (chỉ số đọc được hoàn toàn chính xác),
    - R², NSE, MAE, RMSE: tính trên giá trị chỉ số công tơ (kWh) ghép từ 5 chữ số, theo yêu cầu đề.
    """
    digits_true, digits_pred = np.asarray(digits_true), np.asarray(digits_pred)
    powers = 10 ** np.arange(digits_true.shape[1] - 1, -1, -1)
    value_true, value_pred = digits_true @ powers, digits_pred @ powers
    return {"digit_accuracy": float((digits_true == digits_pred).mean()),
            "reading_accuracy": float((digits_true == digits_pred).all(axis=1).mean()),
            **regression_metrics(value_true, value_pred)}


def multistep_metrics(y_true, y_pred):
    """Chỉ số cho dự báo nhiều bước, y có dạng (số mẫu, số bước).

    - mỗi bước t+h được đánh giá riêng (sai số thường tăng khi dự báo xa hơn),
    - "all steps": gộp tất cả giá trị của mọi bước rồi tính một lần (cách notebook của cô làm),
    - "mean of steps": trung bình cộng chỉ số của các bước.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rows = {f"t+{h + 1}": regression_metrics(y_true[:, h], y_pred[:, h]) for h in range(y_true.shape[1])}
    table = pd.DataFrame(rows).T
    table.loc["mean of steps"] = table.mean()
    table.loc["all steps"] = pd.Series(regression_metrics(y_true, y_pred))
    return table
