"""
app.py  —  Varsha Weather Backend
==================================
Flask server with two main routes:
  GET  /api/realtime          → Current weather from OpenWeatherMap
  POST /api/predict           → ML model forecast for given date/hour
  GET  /api/forecast/<days>   → Multi-day ML forecast
  GET  /                      → Serve frontend
"""

import os, math, datetime
import numpy as np
import requests
import joblib
from flask import Flask, jsonify, request, render_template, send_from_directory

app = Flask(__name__, template_folder="templates", static_folder="static")

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
OWM_API_KEY = "8dd79221ea3d7291f12cef5521b10897"
INDORE_LAT  = 22.7196
INDORE_LON  = 75.8577
CITY_NAME   = "Indore"

MODEL_DIR   = os.path.join(os.path.dirname(__file__), "models")

# ─────────────────────────────────────────────
# Load ML models once at startup
# ─────────────────────────────────────────────
print("Loading ML models...")
MODELS = {
    "temperature":  joblib.load(os.path.join(MODEL_DIR, "temperature_2m_model.joblib")),
    "humidity":     joblib.load(os.path.join(MODEL_DIR, "relative_humidity_2m_model.joblib")),
    "wind_speed":   joblib.load(os.path.join(MODEL_DIR, "wind_speed_10m_model.joblib")),
    "rain":         joblib.load(os.path.join(MODEL_DIR, "rain_model.joblib")),
    "will_rain":    joblib.load(os.path.join(MODEL_DIR, "will_rain_model.joblib")),
}
FEATURE_COLS = joblib.load(os.path.join(MODEL_DIR, "feature_cols.joblib"))
print(f"Models loaded. Features: {FEATURE_COLS}")

# ─────────────────────────────────────────────
# CORS helper (no flask-cors needed)
# ─────────────────────────────────────────────
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response

@app.after_request
def after_request(response):
    return add_cors(response)

@app.route("/", methods=["OPTIONS"])
@app.route("/api/<path:p>", methods=["OPTIONS"])
def options_handler(p=""):
    return jsonify({}), 200

# ─────────────────────────────────────────────
# Feature engineering helper
# ─────────────────────────────────────────────
def build_features(dt: datetime.datetime, owm_data: dict = None) -> list:
    """
    Build the 18-feature vector expected by the ML models.
    If owm_data is provided, use real pressure/cloud/dew values;
    otherwise use climatological defaults for Indore.
    """
    hour        = dt.hour
    month       = dt.month
    doy         = dt.timetuple().tm_yday
    dow         = dt.weekday()

    hour_sin    = math.sin(2 * math.pi * hour  / 24)
    hour_cos    = math.cos(2 * math.pi * hour  / 24)
    month_sin   = math.sin(2 * math.pi * month / 12)
    month_cos   = math.cos(2 * math.pi * month / 12)
    doy_sin     = math.sin(2 * math.pi * doy   / 365)
    doy_cos     = math.cos(2 * math.pi * doy   / 365)

    if owm_data:
        pressure_msl      = owm_data.get("pressure_msl",      1010.0)
        surface_pressure  = owm_data.get("surface_pressure",   968.0)
        cloud_cover       = owm_data.get("cloud_cover",         50.0)
        cloud_cover_low   = owm_data.get("cloud_cover_low",     30.0)
        cloud_cover_mid   = owm_data.get("cloud_cover_mid",     20.0)
        cloud_cover_high  = owm_data.get("cloud_cover_high",    10.0)
        dew_point_2m      = owm_data.get("dew_point_2m",        18.0)
        apparent_temp     = owm_data.get("apparent_temperature",28.0)
        wind_dir          = owm_data.get("wind_direction_10m",  180.0)
        wind_gusts        = owm_data.get("wind_gusts_10m",      20.0)
        snow_depth        = owm_data.get("snow_depth",           0.0)
    else:
        # Indore climatological defaults per month
        clim = {
            1:  dict(p=1016, sp=974, cc=15, cl=8,  cm=5,  ch=5,  dp=8,  at=16, wd=50,  wg=12, sn=0),
            2:  dict(p=1013, sp=971, cc=20, cl=10, cm=8,  ch=8,  dp=12, at=20, wd=80,  wg=14, sn=0),
            3:  dict(p=1009, sp=967, cc=20, cl=10, cm=8,  ch=8,  dp=15, at=29, wd=160, wg=18, sn=0),
            4:  dict(p=1005, sp=963, cc=15, cl=8,  cm=5,  ch=5,  dp=17, at=37, wd=200, wg=22, sn=0),
            5:  dict(p=1000, sp=958, cc=20, cl=10, cm=8,  ch=8,  dp=20, at=42, wd=220, wg=28, sn=0),
            6:  dict(p=999,  sp=957, cc=65, cl=45, cm=30, ch=20, dp=24, at=36, wd=250, wg=38, sn=0),
            7:  dict(p=1000, sp=958, cc=85, cl=65, cm=45, ch=25, dp=26, at=33, wd=260, wg=42, sn=0),
            8:  dict(p=1001, sp=959, cc=80, cl=60, cm=40, ch=20, dp=25, at=32, wd=255, wg=38, sn=0),
            9:  dict(p=1005, sp=963, cc=60, cl=40, cm=25, ch=15, dp=23, at=32, wd=240, wg=28, sn=0),
            10: dict(p=1010, sp=968, cc=25, cl=12, cm=10, ch=8,  dp=18, at=30, wd=100, wg=16, sn=0),
            11: dict(p=1014, sp=972, cc=15, cl=8,  cm=5,  ch=5,  dp=12, at=24, wd=60,  wg=12, sn=0),
            12: dict(p=1016, sp=974, cc=15, cl=8,  cm=5,  ch=5,  dp=8,  at=18, wd=40,  wg=10, sn=0),
        }
        c = clim.get(month, clim[6])
        pressure_msl, surface_pressure = c["p"], c["sp"]
        cloud_cover, cloud_cover_low   = c["cc"], c["cl"]
        cloud_cover_mid, cloud_cover_high = c["cm"], c["ch"]
        dew_point_2m, apparent_temp    = c["dp"], c["at"]
        wind_dir, wind_gusts, snow_depth = c["wd"], c["wg"], c["sn"]

    feat_map = {
        "hour_sin": hour_sin, "hour_cos": hour_cos,
        "month_sin": month_sin, "month_cos": month_cos,
        "doy_sin": doy_sin, "doy_cos": doy_cos,
        "day_of_week": dow,
        "pressure_msl": pressure_msl,
        "surface_pressure": surface_pressure,
        "cloud_cover": cloud_cover,
        "cloud_cover_low": cloud_cover_low,
        "cloud_cover_mid": cloud_cover_mid,
        "cloud_cover_high": cloud_cover_high,
        "dew_point_2m": dew_point_2m,
        "apparent_temperature": apparent_temp,
        "wind_direction_10m": wind_dir,
        "wind_gusts_10m": wind_gusts,
        "snow_depth": snow_depth,
    }
    return [feat_map.get(f, 0.0) for f in FEATURE_COLS]


def ml_predict(dt: datetime.datetime, owm_data: dict = None) -> dict:
    """Run all 5 ML models and return prediction dict."""
    features = np.array([build_features(dt, owm_data)])

    temp      = float(MODELS["temperature"].predict(features)[0])
    humidity  = float(MODELS["humidity"].predict(features)[0])
    wind      = float(MODELS["wind_speed"].predict(features)[0])
    rain_mm   = float(max(0, MODELS["rain"].predict(features)[0]))
    will_rain = int(MODELS["will_rain"].predict(features)[0])

    # Rain probability from classifier
    rain_proba = float(MODELS["will_rain"].predict_proba(features)[0][1])

    condition  = derive_condition(temp, humidity, rain_proba, rain_mm, dt.hour, dt.month)
    description= derive_description(condition, temp, humidity, wind, rain_proba, dt.hour)

    return {
        "temperature":     round(temp, 1),
        "humidity":        round(max(0, min(100, humidity))),
        "wind_speed":      round(max(0, wind), 1),
        "rain_mm":         round(rain_mm, 2),
        "rain_probability":round(max(0, min(1, rain_proba)), 3),
        "will_rain":       bool(will_rain),
        "condition":       condition,
        "description":     description,
    }


def derive_condition(temp, hum, rain_prob, rain_mm, hour, month):
    if rain_prob > 0.75 or rain_mm > 5:
        return "Heavy Rain"
    if rain_prob > 0.55 or rain_mm > 1:
        return "Light Rain"
    if rain_prob > 0.35:
        return "Drizzle"
    if hum > 85:
        return "Overcast & Muggy"
    if month in (6, 7, 8, 9) and rain_prob > 0.2:
        return "Partly Cloudy"
    if temp > 42:
        return "Extreme Heat"
    if temp > 36:
        return "Hot & Sunny"
    if temp < 12:
        return "Cold & Clear" if hour > 9 else "Foggy Morning"
    if temp < 18:
        return "Cool & Clear"
    return "Partly Cloudy" if hum > 55 else "Clear & Sunny"


def derive_description(cond, temp, hum, wind, rain_prob, hour):
    time_str = "morning" if hour < 12 else ("afternoon" if hour < 18 else "evening")
    base = f"{cond} {time_str} in Indore."
    extras = []
    if temp > 40:
        extras.append(f"Extreme heat at {temp:.0f}°C — stay indoors.")
    elif temp > 35:
        extras.append(f"Hot at {temp:.0f}°C, carry water.")
    elif temp < 15:
        extras.append(f"Cool at {temp:.0f}°C, light jacket recommended.")
    if hum > 80:
        extras.append(f"Very humid ({hum:.0f}%).")
    if wind > 30:
        extras.append(f"Strong winds at {wind:.0f} km/h.")
    if rain_prob > 0.5:
        extras.append("Carry an umbrella.")
    return " ".join([base] + extras)


# ─────────────────────────────────────────────
# Route 1: Real-time weather (OpenWeatherMap)
# ─────────────────────────────────────────────
@app.route("/api/realtime")
def realtime():
    try:
        url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?lat={INDORE_LAT}&lon={INDORE_LON}"
            f"&appid={OWM_API_KEY}&units=metric"
        )
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        d = r.json()

        main   = d.get("main", {})
        wind   = d.get("wind", {})
        clouds = d.get("clouds", {})
        rain   = d.get("rain", {})
        weather= d.get("weather", [{}])[0]

        # Build OWM context for ML model
        temp_owm    = main.get("temp", 28)
        feels_like  = main.get("feels_like", temp_owm)
        pressure    = main.get("pressure", 1010)
        humidity    = main.get("humidity", 60)
        cloud_pct   = clouds.get("all", 30)
        wind_speed  = wind.get("speed", 3) * 3.6   # m/s → km/h
        wind_gust   = wind.get("gust", wind.get("speed", 3)) * 3.6
        wind_deg    = wind.get("deg", 180)
        rain_1h     = rain.get("1h", 0)
        desc        = weather.get("description", "").title()
        icon_code   = weather.get("icon", "01d")

        # Dew point approximation (Magnus formula)
        dew_point = temp_owm - ((100 - humidity) / 5)

        owm_context = {
            "pressure_msl": pressure,
            "surface_pressure": pressure - 42,   # Indore elevation offset
            "cloud_cover": cloud_pct,
            "cloud_cover_low": cloud_pct * 0.6,
            "cloud_cover_mid": cloud_pct * 0.25,
            "cloud_cover_high": cloud_pct * 0.15,
            "dew_point_2m": dew_point,
            "apparent_temperature": feels_like,
            "wind_direction_10m": wind_deg,
            "wind_gusts_10m": wind_gust,
            "snow_depth": 0,
        }

        now = datetime.datetime.utcnow() + datetime.timedelta(hours=5, minutes=30)
        ml  = ml_predict(now, owm_context)

        return jsonify({
            "source": "realtime",
            "city": CITY_NAME,
            "lat": INDORE_LAT,
            "lon": INDORE_LON,
            "timestamp": now.strftime("%d %b %Y, %I:%M %p IST"),
            "owm": {
                "temperature":     round(temp_owm, 1),
                "feels_like":      round(feels_like, 1),
                "humidity":        humidity,
                "pressure":        pressure,
                "wind_speed":      round(wind_speed, 1),
                "wind_direction":  wind_deg,
                "cloud_cover":     cloud_pct,
                "rain_1h":         rain_1h,
                "description":     desc,
                "icon":            icon_code,
            },
            "ml": ml,
        })

    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Cannot reach OpenWeatherMap API. Check internet connection."}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# Route 2: ML Prediction for specific datetime
# ─────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        body = request.get_json(force=True)
        date_str = body.get("date")   # "YYYY-MM-DD"
        hour     = int(body.get("hour", 12))

        if not date_str:
            return jsonify({"error": "date is required (YYYY-MM-DD)"}), 400

        dt = datetime.datetime.strptime(date_str, "%Y-%m-%d").replace(hour=hour)
        result = ml_predict(dt)
        result["date"]      = date_str
        result["hour"]      = hour
        result["timestamp"] = dt.strftime("%d %b %Y, %I:%M %p")

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# Route 3: Multi-day forecast (ML)
# ─────────────────────────────────────────────
@app.route("/api/forecast")
def forecast():
    try:
        days     = int(request.args.get("days", 7))
        date_str = request.args.get("date", datetime.date.today().isoformat())
        days     = max(1, min(days, 14))

        base_date = datetime.date.fromisoformat(date_str)
        result    = []

        for i in range(days):
            d = base_date + datetime.timedelta(days=i)
            day_preds = []
            # Predict at 8 hour intervals
            for h in [6, 9, 12, 15, 18, 21]:
                dt = datetime.datetime.combine(d, datetime.time(h))
                p  = ml_predict(dt)
                p["hour"] = h
                day_preds.append(p)

            temps     = [p["temperature"] for p in day_preds]
            rain_prbs = [p["rain_probability"] for p in day_preds]
            rain_mms  = [p["rain_mm"] for p in day_preds]

            result.append({
                "date":       d.isoformat(),
                "day_name":   d.strftime("%A"),
                "day_short":  d.strftime("%a"),
                "temp_high":  round(max(temps), 1),
                "temp_low":   round(min(temps), 1),
                "rain_prob":  round(max(rain_prbs), 3),
                "rain_mm":    round(sum(rain_mms), 1),
                "condition":  day_preds[2]["condition"],   # noon condition
                "hourly":     day_preds,
            })

        return jsonify({"days": days, "start": date_str, "forecast": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# Route 4: Hourly forecast for one day
# ─────────────────────────────────────────────
@app.route("/api/hourly")
def hourly():
    try:
        date_str = request.args.get("date", datetime.date.today().isoformat())
        d = datetime.date.fromisoformat(date_str)
        hours_data = []
        for h in range(0, 24, 1):
            dt = datetime.datetime.combine(d, datetime.time(h))
            p  = ml_predict(dt)
            p["hour"] = h
            hours_data.append(p)
        return jsonify({"date": date_str, "hourly": hours_data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# Serve frontend
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("templates", "index.html")

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  🌦  VARSHA WEATHER SERVER STARTING")
    print("  Open: http://localhost:5000")
    print("="*55 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
