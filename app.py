# 🌦 Varsha — Indore Weather Intelligence

ML-powered weather prediction system for Indore city.
Real-time data from OpenWeatherMap + Random Forest models trained on 123,936 hourly records.

---

## 📁 Project Structure

```
varsha_weather/
├── app.py                  ← Flask backend (main server)
├── requirements.txt        ← Python dependencies
├── models/                 ← Trained ML models (.joblib)
│   ├── temperature_2m_model.joblib
│   ├── relative_humidity_2m_model.joblib
│   ├── wind_speed_10m_model.joblib
│   ├── rain_model.joblib
│   ├── will_rain_model.joblib
│   └── feature_cols.joblib
├── templates/
│   └── index.html          ← Frontend HTML
└── static/
    ├── css/style.css       ← Styles
    └── js/main.js          ← Frontend JavaScript
```

---

## ⚙️ Setup & Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Start the server
```bash
python app.py
```

### Step 3 — Open browser
```
http://localhost:5000
```

---

## 🔌 API Endpoints

| Method | Endpoint              | Description                        |
|--------|-----------------------|------------------------------------|
| GET    | `/api/realtime`       | Current weather from OpenWeatherMap |
| POST   | `/api/predict`        | ML prediction for date + hour       |
| GET    | `/api/forecast`       | 7-day ML forecast                  |
| GET    | `/api/hourly`         | Hourly ML predictions for one day  |

### POST /api/predict — Example
```json
{
  "date": "2026-07-15",
  "hour": 14
}
```

### GET /api/forecast — Query Params
```
/api/forecast?date=2026-07-01&days=7
```

---

## 🧠 ML Models

| Target                | Algorithm             | Performance        |
|-----------------------|-----------------------|--------------------|
| Temperature (°C)      | Random Forest         | R² = 0.997, MAE 0.26°C |
| Humidity (%)          | Random Forest         | R² = 0.998, MAE 0.93%  |
| Wind Speed (km/h)     | Random Forest         | R² = 0.917, MAE 1.07   |
| Rain Amount (mm)      | Gradient Boosting     | R² = 0.316             |
| Will Rain (Yes/No)    | Random Forest         | Accuracy = 92.1%       |

### Features Used (18 total)
- Cyclical time encodings: hour_sin/cos, month_sin/cos, doy_sin/cos
- Pressure (MSL + surface)
- Cloud cover (total, low, mid, high)
- Dew point, apparent temperature
- Wind direction, wind gusts
- Snow depth

---

## 🔑 API Key
OpenWeatherMap API key is embedded in `app.py`:
```python
OWM_API_KEY = "8dd79221ea3d7291f12cef5521b10897"
```
Get a free key at: https://openweathermap.org/api

---

## 📊 Dataset
- **Source**: Indore historical weather (Indorecity.csv)
- **Records**: 123,936 hourly observations
- **Period**: 2010–2024
- **Training sample**: 40,000 rows (stratified random)
