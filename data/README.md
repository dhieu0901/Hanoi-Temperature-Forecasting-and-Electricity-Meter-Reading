# Data

## Task 2: `hanoi_weather_hourly_2016_2025.csv`

- **Source:** Open-Meteo Historical Weather API (ERA5 reanalysis), https://open-meteo.com/en/docs/historical-weather-api
- **Download link used:**
  https://archive-api.open-meteo.com/v1/archive?latitude=21.0285&longitude=105.8542&start_date=2016-01-01&end_date=2025-12-31&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,surface_pressure,precipitation,cloud_cover,wind_speed_10m,wind_direction_10m&timezone=Asia%2FHo_Chi_Minh&format=csv
- **Recreate:** delete the CSV and run notebook 01 (it calls `src.weather_data.download_weather()`); the committed file was downloaded on 2026-11-16
- **Location:** central Hanoi (21.0285°N, 105.8542°E); Open-Meteo snaps to the nearest grid cell (21.05°N, 105.90°E, elevation 19 m)
- **Period / frequency:** 2016-01-01 00:00 → 2025-12-31 23:00, hourly, local time (GMT+7)
- **Size:** 87,672 rows × 9 columns, no missing values, no duplicate or skipped hours
- **Licence:** CC BY 4.0, cite "Weather data by Open-Meteo.com". The values are ERA5 reanalysis (model-based), not direct station measurements.

| Column | Meaning | Unit |
|---|---|---|
| `time` | timestamp | — |
| `temp` | air temperature at 2 m (**pred_column**) | °C |
| `humidity` | relative humidity at 2 m | % |
| `dew_point` | dew point at 2 m | °C |
| `pressure` | surface pressure | hPa |
| `precipitation` | precipitation in the hour | mm |
| `cloud_cover` | total cloud cover | % |
| `wind_speed` | wind speed at 10 m | km/h |
| `wind_dir` | wind direction at 10 m | degrees |

Split by time (notebook 01): train 61,370 h (2016-01-01 → 2023-01-01), validation 13,151 h (→ 2024-07-02), test 13,151 h (2024-07-02 → 2025-12-31).

**Persistence baseline** (forecast = current temperature), test set:

| Horizon | NSE | R² | MAE (°C) | RMSE (°C) |
|---|---|---|---|---|
| t+1 | 0.971 | 0.972 | 0.64 | 0.89 |
| t+2 | 0.911 | 0.913 | 1.18 | 1.57 |
| t+3 | 0.830 | 0.837 | 1.67 | 2.16 |
| t+4 | 0.736 | 0.753 | 2.13 | 2.70 |
| t+5 | 0.635 | 0.668 | 2.54 | 3.17 |

## Task 3: UFPR-AMR (electricity meter reading)

- **Source:** https://web.inf.ufpr.br/vri/databases/ufpr-amr/ (also https://github.com/raysonlaroca/ufpr-amr-dataset)
- **Citation:** R. Laroca, V. Barroso, M. A. Diniz, G. R. Gonçalves, W. R. Schwartz, D. Menotti, "Convolutional Neural Networks for Automatic Meter Reading," *Journal of Electronic Imaging*, 28(1):013023, 2019.
- **Access:** by e-mail request from a university account (granted on 2026-11-19), for academic, non-commercial use only. The data and the personal download link must not be shared, so neither is in Git. Put the received archive at `data/raw/UFPR-AMR.zip` (5.1 GB); the notebooks read it without unzipping.
- **Content:** 2,000 JPEG photos (2,340 x 4,160 to 3,120 x 4,160) of meters in a Copel warehouse (Paraná, Brazil), three phone cameras (LG G3 D855 947, Samsung Galaxy J7 Prime 584, iPhone 6s 469). Each image has a text file with the camera, the 5-digit reading, the counter box (x, y, w, h) and the box of each digit.
- **Split:** official 800 train / 400 validation / 800 test, kept as is.
- **Prepared input:** notebook 04 crops every counter (box + 8 % / 15 % margin), resizes it to 64 x 256 RGB and caches the result in `data/raw/ufpr_amr_counters_64x256.npz` (58 MB, not in Git).
- Figures named `figures/meter/samples_*` show dataset images; they may be used in the report and slides but are excluded from Git.
