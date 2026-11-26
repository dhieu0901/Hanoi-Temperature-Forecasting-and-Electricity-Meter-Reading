"""Các hàm huấn luyện dùng chung cho Task 2 và Task 3: callbacks, lưu/đọc lịch sử, learning curve, nhật ký thí nghiệm."""
import time

import keras
import matplotlib.pyplot as plt
import pandas as pd

from src.config import LOGS_DIR, MODELS_DIR, REPORTS_DIR


def make_callbacks(run, patience=8, lr_patience=3):
    (LOGS_DIR / run).mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    return [
        # lưu lại epoch có val_loss thấp nhất
        keras.callbacks.ModelCheckpoint(str(MODELS_DIR / f"{run}.keras"), monitor="val_loss", save_best_only=True),
        # dừng khi val_loss không giảm sau `patience` epoch, lấy lại trọng số tốt nhất
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1),
        # giảm learning rate còn một nửa khi val_loss không giảm sau `lr_patience` epoch
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=lr_patience, min_lr=1e-5, verbose=1),
        # ghi loss/metric của từng epoch ra file CSV
        keras.callbacks.CSVLogger(str(LOGS_DIR / run / "history.csv")),
    ]


def train_model(model, run, x_train, y_train, validation_data, epochs, batch_size, patience=8, lr_patience=3):
    """Train model, trả về (lịch sử train dạng DataFrame, số phút đã train)."""
    t0 = time.time()
    model.fit(x_train, y_train, validation_data=validation_data, epochs=epochs, batch_size=batch_size,
              callbacks=make_callbacks(run, patience, lr_patience), verbose=2)
    return load_history(run), (time.time() - t0) / 60


def load_history(run):
    return pd.read_csv(LOGS_DIR / run / "history.csv")


def load_trained(run):
    return keras.models.load_model(MODELS_DIR / f"{run}.keras")


def plot_history(history, title, metric=None):
    keys = ["loss"] + ([metric] if metric else [])
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 3.5), squeeze=False)
    epochs = range(1, len(history) + 1)
    for ax, key in zip(axes[0], keys):
        ax.plot(epochs, history[key], label="train")
        ax.plot(epochs, history["val_" + key], label="validation")
        ax.axvline(history["val_loss"].idxmin() + 1, color="grey", ls="--", lw=0.8)  # epoch tốt nhất
        ax.set_title(key)
        ax.set_xlabel("epoch")
        ax.legend()
    fig.suptitle(title)
    plt.tight_layout()
    return fig


def log_experiment(row, path=REPORTS_DIR / "experiment_log.csv"):
    # mỗi lần train ghi một dòng; chạy lại cùng một run thì thay dòng cũ
    row = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        row = pd.concat([old[old["run"] != row.loc[0, "run"]], row], ignore_index=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    row.convert_dtypes().to_csv(path, index=False)  # convert_dtypes: giữ số nguyên là 64, không thành 64.0
    return row
