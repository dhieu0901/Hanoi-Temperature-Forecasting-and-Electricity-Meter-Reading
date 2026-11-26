from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"  # dữ liệu tải về không đưa lên Git (UFPR-AMR không được chia sẻ lại)
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

SEED = 42

# ---------------- Task 2: dự báo nhiệt độ Hà Nội 5 giờ tới ----------------
WEATHER_CSV = DATA_DIR / "hanoi_weather_hourly_2016_2025.csv"
LATITUDE, LONGITUDE = 21.0285, 105.8542  # trung tâm Hà Nội (Hoàn Kiếm)
START_DATE, END_DATE = "2016-01-01", "2025-12-31"
TIMEZONE = "Asia/Ho_Chi_Minh"

# tên biến trên Open-Meteo -> tên cột trong project
WEATHER_VARIABLES = {
    "temperature_2m": "temp",            # nhiệt độ 2 m (°C)
    "relative_humidity_2m": "humidity",  # độ ẩm tương đối (%)
    "dew_point_2m": "dew_point",         # điểm sương (°C)
    "surface_pressure": "pressure",      # áp suất bề mặt (hPa)
    "precipitation": "precipitation",    # lượng mưa trong giờ (mm)
    "cloud_cover": "cloud_cover",        # độ che phủ mây (%)
    "wind_speed_10m": "wind_speed",      # tốc độ gió 10 m (km/h)
    "wind_direction_10m": "wind_dir",    # hướng gió 10 m (độ)
}

TARGET = "temp"  # pred_column
# hướng gió là góc (0° và 360° là cùng một hướng) nên được đổi thành sin và cos
FEATURES = ["temp", "humidity", "dew_point", "pressure", "precipitation",
            "cloud_cover", "wind_speed", "wind_dir_sin", "wind_dir_cos"]

N_STEPS = 24   # input: 24 giờ gần nhất (t-23, ..., t)
HORIZON = 5    # output: nhiệt độ ở t+1, ..., t+5
TRAIN_RATIO, VAL_RATIO = 0.70, 0.15  # chia theo thời gian, phần còn lại (15 %) là test

# ---------------- Task 3: đọc chỉ số công tơ điện (UFPR-AMR) ----------------
AMR_ZIP = RAW_DIR / "UFPR-AMR.zip"  # file zip gốc nhận qua email, đọc trực tiếp không cần giải nén
AMR_CACHE = RAW_DIR / "ufpr_amr_counters_64x256.npz"  # ảnh vùng số đã cắt sẵn (tạo bởi notebook 04)
AMR_SPLITS = {"training": "train", "validation": "val", "testing": "test"}  # thư mục trong zip -> tên tập
NUM_DIGITS = 5          # mỗi công tơ có đúng 5 chữ số
NUM_CLASSES = 10        # mỗi vị trí là một chữ số 0-9
IMG_H, IMG_W = 64, 256  # kích thước ảnh vùng số đưa vào CNN (rộng gấp 4 lần cao, giống tỉ lệ thật ~4:1)
CROP_MARGIN = (0.08, 0.15)  # nới khung vùng số thêm 8 % mỗi bên theo chiều ngang, 15 % theo chiều dọc
