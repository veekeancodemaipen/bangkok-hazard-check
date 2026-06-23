"""
Bangkok Journey Intelligence — BDI Hackathon 2026
FastAPI backend: Traffy Fondue + Air4Thai + Weather + Typhoon Thai LLM
v2: chain-of-thought reasoning + weather data
"""

import io
import math
import os
import re
import json
import asyncio
import logging
from datetime import datetime
from typing import Optional

import httpx
import pandas as pd
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
AIR4THAI_URL  = "http://air4thai.pcd.go.th/services/getNewAQI_JSON.php"
WEATHER_URL   = "https://wttr.in/Bangkok?format=j1"
TYPHOON_BASE_URL = "https://api.opentyphoon.ai/v1"
TYPHOON_MODEL    = "typhoon-v2.5-30b-a3b-instruct"
TYPHOON_API_KEY  = os.getenv("TYPHOON_API_KEY", "")
PARQUET_PATH     = os.path.join(os.path.dirname(__file__), "filtered_tickets.parquet")
STATUS_PATH      = os.path.join(os.path.dirname(__file__), "traffy_status.json")
DIGEST_PATH      = os.path.join(os.path.dirname(__file__), "weekly_digest.json")
RISK6_PATH       = os.path.join(os.path.dirname(__file__), "..", "..", "03_data", "compound_risk_by_district.csv")
OPEN_STATE_EXCLUDE = "เสร็จสิ้น"

AQI_LEVELS = [
    (25,  "ดีมาก",                          "1"),
    (37,  "ดี",                              "2"),
    (50,  "ปานกลาง",                         "3"),
    (90,  "เริ่มมีผลกระทบต่อสุขภาพ",         "4"),
    (150, "มีผลกระทบต่อสุขภาพ",              "5"),
    (999, "อันตราย",                          "6"),
]

# ---------------------------------------------------------------------------
# Startup: load Traffy data
# ---------------------------------------------------------------------------
traffy_df: pd.DataFrame = pd.DataFrame()
risk6_map: dict = {}          # district → {score_01, score_x10, rank, layers}
_geojson_cache: dict = {"etag": None, "body": None}


def load_risk6() -> dict:
    """Load 6-layer compound risk from compound_risk_by_district.csv → dict keyed by district."""
    try:
        df = pd.read_csv(RISK6_PATH, encoding="utf-8-sig")
        result = {}
        for _, row in df.iterrows():
            dist = str(row.get("district", "") or "").strip()
            if not dist:
                continue
            score_w = float(row.get("risk_weighted", 0) or 0)
            score_e = float(row.get("risk_equal",    0) or 0)
            rank    = int(row.get("rank_weighted",   50) or 50)
            result[dist] = {
                "score_01":  round(score_w, 4),
                "score_x10": round(score_w * 10, 1),
                # 6-layer breakdown (0–1 each)
                "layers": {
                    "traffy_vol":  round(float(row.get("score_traffy_vol", 0) or 0), 2),
                    "traffy_dur":  round(float(row.get("score_traffy_dur", 0) or 0), 2),
                    "flood":       round(float(row.get("score_flood",      0) or 0), 2),
                    "safety":      round(float(row.get("score_safety",     0) or 0), 2),
                    "air":         round(float(row.get("score_air",        0) or 0), 2),
                    "heat":        round(float(row.get("score_heat",       0) or 0), 2),
                },
                "rank": rank,
            }
        logger.info("Risk6 loaded: %d districts", len(result))
        return result
    except Exception as e:
        logger.error("Risk6 load failed: %s", e)
        return {}


def _parquet_etag() -> str:
    try:
        s = os.stat(PARQUET_PATH)
        return f"{s.st_mtime_ns}-{s.st_size}"
    except OSError:
        return ""


def load_traffy() -> pd.DataFrame:
    try:
        df = pd.read_parquet(PARQUET_PATH)
        logger.info("Traffy loaded: %d open tickets", len(df))
        return df
    except FileNotFoundError:
        logger.error("filtered_tickets.parquet not found at %s", PARQUET_PATH)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Bangkok Journey Intelligence API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    global traffy_df, risk6_map
    traffy_df = load_traffy()
    risk6_map = load_risk6()


# ---------------------------------------------------------------------------
# Helpers — AQI
# ---------------------------------------------------------------------------

def aqi_label(pm25: float) -> tuple[str, str]:
    for threshold, label, color in AQI_LEVELS:
        if pm25 <= threshold:
            return label, color
    return "อันตราย", "6"


async def fetch_air4thai() -> dict:
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(AIR4THAI_URL)
        resp.raise_for_status()
        return resp.json()


def parse_air4thai(raw: dict) -> list[dict]:
    stations = []
    for s in raw.get("stations", []):
        area_en = s.get("areaEN", "") or ""
        if "Bangkok" not in area_en:
            continue
        aqi_data  = s.get("AQILast", {}) or {}
        pm25_block = aqi_data.get("PM25", {}) or {}
        try:
            pm25_ugm3 = float(pm25_block.get("value", 0) or 0)
        except (ValueError, TypeError):
            pm25_ugm3 = 0.0
        try:
            pm25_aqi = int(pm25_block.get("aqi", 0) or 0)
        except (ValueError, TypeError):
            pm25_aqi = 0
        try:
            overall_aqi = int(aqi_data.get("AQI", {}).get("aqi", 0) or 0)
        except (ValueError, TypeError):
            overall_aqi = pm25_aqi
        try:
            lat = float(s.get("lat", 0) or 0)
            lon = float(s.get("long", 0) or 0)
        except (ValueError, TypeError):
            lat, lon = 0.0, 0.0
        timestamp = pm25_block.get("datetime", "") or aqi_data.get("date", "")
        _, color  = aqi_label(pm25_ugm3)
        stations.append({
            "station_id": s.get("stationID", ""),
            "name_th":    s.get("nameTH", ""),
            "name_en":    s.get("nameEN", ""),
            "district":   area_en,
            "lat": lat, "lon": lon,
            "pm25_ugm3":   pm25_ugm3,
            "pm25_aqi":    pm25_aqi,
            "overall_aqi": overall_aqi,
            "timestamp":   timestamp,
            "color_id":    color,
        })
    return stations


# ---------------------------------------------------------------------------
# Helpers — Weather (wttr.in)
# ---------------------------------------------------------------------------

async def fetch_weather() -> dict:
    """Fetch Bangkok weather from wttr.in — no API key needed."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                WEATHER_URL,
                headers={"User-Agent": "BangkokJourneyIntelligence/2.0"},
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.warning("Weather fetch failed: %s", e)
        return {}


def parse_weather(raw: dict) -> dict:
    """Extract key weather metrics from wttr.in response."""
    if not raw:
        return {}
    try:
        cur   = raw["current_condition"][0]
        today = raw.get("weather", [{}])[0]
        hourly = today.get("hourly", [])

        # Max precip in next ~12 h (hourly entries are 3-h intervals, 8 per day)
        max_precip_12h = 0.0
        for h in hourly[:4]:
            try:
                max_precip_12h = max(max_precip_12h, float(h.get("precipMM", 0) or 0))
            except (ValueError, TypeError):
                pass

        # Rain risk label
        cur_precip = float(cur.get("precipMM", 0) or 0)
        if max_precip_12h >= 10 or cur_precip >= 5:
            rain_risk = "สูง (ฝนตกหนัก)"
        elif max_precip_12h >= 3 or cur_precip >= 1:
            rain_risk = "ปานกลาง (ฝนปรอยถึงปานกลาง)"
        else:
            rain_risk = "ต่ำ (ไม่มีฝนหรือฝนเล็กน้อย)"

        desc_raw = cur.get("weatherDesc", [{}])[0].get("value", "")

        return {
            "temp_c":         int(cur.get("temp_C", 0) or 0),
            "feels_like_c":   int(cur.get("FeelsLikeC", 0) or 0),
            "humidity":       int(cur.get("humidity", 0) or 0),
            "weather_desc":   desc_raw,
            "precip_mm":      cur_precip,
            "wind_kmph":      int(cur.get("windspeedKmph", 0) or 0),
            "max_precip_12h": max_precip_12h,
            "rain_risk":      rain_risk,
            "visibility_km":  int(cur.get("visibility", 10) or 10),
        }
    except Exception as e:
        logger.warning("Weather parse failed: %s", e)
        return {}


def weather_to_context(w: dict) -> str:
    if not w:
        return ""
    return (
        f"สภาพอากาศปัจจุบัน (กรุงเทพฯ): {w.get('weather_desc','')} "
        f"อุณหภูมิ {w.get('temp_c','-')}°C (รู้สึก {w.get('feels_like_c','-')}°C) "
        f"ความชื้น {w.get('humidity','-')}% "
        f"ฝนปัจจุบัน {w.get('precip_mm',0):.1f} มม. "
        f"คาดการณ์ฝน 12 ชม.ข้างหน้า สูงสุด {w.get('max_precip_12h',0):.1f} มม. "
        f"ความเสี่ยงน้ำท่วม/น้ำขัง: {w.get('rain_risk','ไม่มีข้อมูล')}"
    )


# ---------------------------------------------------------------------------
# Helpers — Traffy
# ---------------------------------------------------------------------------

def get_district_traffy(district: str) -> pd.DataFrame:
    if traffy_df.empty:
        return pd.DataFrame()
    mask = traffy_df["district"].str.contains(district, na=False, case=False)
    return traffy_df[mask]


def top_streets(district_df: pd.DataFrame, n: int = 5) -> list[str]:
    if district_df.empty:
        return []
    pattern = r"(?:ถนน|ถ\.|ซอย|ซ\.)\s*([฀-๿\w\s\d]+?)(?=\s+(?:แขวง|เขต|กรุงเทพ)|$)"
    extracted = district_df["address"].dropna().str.extract(pattern, expand=False)
    return (
        extracted.dropna()
        .str.strip()
        .value_counts()
        .head(n)
        .index.tolist()
    )


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in km between two lat/lon points."""
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def district_centroid(district: str):
    if traffy_df.empty:
        return None
    sub = traffy_df[traffy_df["district"] == district]
    if sub.empty:
        return None
    return float(sub["lat"].median()), float(sub["lon"].median())


def nearest_station(stations, lat: float, lon: float) -> dict:
    return min(stations, key=lambda s: haversine(lat, lon, s["lat"], s["lon"]) if s["lat"] and s["lon"] else 9999)


def count_repeat_clusters(district_df: pd.DataFrame) -> int:
    if district_df.empty:
        return 0
    coord_counts = district_df.groupby(
        [district_df["lat"].round(4), district_df["lon"].round(4)]
    ).size()
    return int((coord_counts >= 3).sum())


def compute_risk_score(pm25: float, open_tickets: int, repeat_clusters: int, rain_risk: str) -> dict:
    """
    Real-time Risk Score (0–10). Computed deterministically in Python — LLM only explains.
    Components: AQI (0–4 pts) + Traffy backlog (0–4 pts) + Weather rain (0–2 pts)
    Note: district-level 6-layer baseline is in /api/compound-risk endpoint.
    """
    if   pm25 > 150: aqi_pts = 4.0
    elif pm25 > 90:  aqi_pts = 3.0
    elif pm25 > 50:  aqi_pts = 2.0
    elif pm25 > 37:  aqi_pts = 1.5
    elif pm25 > 25:  aqi_pts = 1.0
    else:            aqi_pts = 0.0

    ticket_pts  = min(open_tickets   / 500, 2.0)
    repeat_pts  = min(repeat_clusters / 20, 2.0)
    traffy_pts  = ticket_pts + repeat_pts

    weather_pts = {"สูง (ฝนตกหนัก)": 2.0, "ปานกลาง (ฝนปรอยถึงปานกลาง)": 1.0}.get(rain_risk, 0.0)

    score = round(min(aqi_pts + traffy_pts + weather_pts, 10.0), 1)

    if   score >= 8: level, color = "อันตราย",  "#c084fc"
    elif score >= 6: level, color = "สูง",       "#fc8181"
    elif score >= 3: level, color = "ปานกลาง",  "#F6AD55"
    else:            level, color = "ต่ำ",       "#48bb78"

    return {
        "score": score, "level": level, "color": color,
        "components": {
            "aqi":     round(aqi_pts,     1),
            "traffy":  round(traffy_pts,  1),
            "weather": round(weather_pts, 1),
        },
    }


# ---------------------------------------------------------------------------
# Helpers — Typhoon LLM
# ---------------------------------------------------------------------------

async def call_typhoon(messages: list[dict], max_tokens: int = 1200) -> str:
    if not TYPHOON_API_KEY:
        raise ValueError("TYPHOON_API_KEY is not set")
    payload = {
        "model": TYPHOON_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.35,
    }
    headers = {
        "Authorization": f"Bearer {TYPHOON_API_KEY}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=40.0) as client:
        resp = await client.post(
            f"{TYPHOON_BASE_URL}/chat/completions",
            json=payload, headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()
    return data["choices"][0]["message"]["content"]


async def extract_district_from_query(query: str) -> Optional[str]:
    if not TYPHOON_API_KEY:
        return None
    prompt = (
        "จากคำถามต่อไปนี้ ให้ตอบเป็น JSON object เดียวเท่านั้น ไม่ต้องอธิบายเพิ่มเติม\n"
        'รูปแบบ: {"district": "<ชื่อเขต เช่น ปทุมวัน, ลาดพร้าว>"}\n'
        "ถ้าไม่พบเขต ให้ใส่ null\n\n"
        f"คำถาม: {query}"
    )
    try:
        result = await call_typhoon([{"role": "user", "content": prompt}], max_tokens=80)
        match  = re.search(r'\{[^}]+\}', result, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return data.get("district")
    except Exception as e:
        logger.warning("District extraction failed: %s", e)
    return None


def parse_cot_response(text: str) -> tuple[str, str]:
    """
    Split chain-of-thought response into (reasoning, answer).
    Expects markers [วิเคราะห์]...[/วิเคราะห์][คำตอบ]...[/คำตอบ]
    Falls back to (empty, full_text) if markers absent.
    """
    r_match = re.search(r'\[วิเคราะห์\](.*?)\[/วิเคราะห์\]', text, re.DOTALL)
    a_match = re.search(r'\[คำตอบ\](.*?)\[/คำตอบ\]',    text, re.DOTALL)
    reasoning = r_match.group(1).strip() if r_match else ""
    answer    = a_match.group(1).strip() if a_match else text.strip()
    return reasoning, answer


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/aqi")
async def get_aqi():
    try:
        raw = await fetch_air4thai()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Air4Thai fetch failed: {e}")
    stations   = parse_air4thai(raw)
    updated_at = datetime.now().isoformat(timespec="seconds")
    return {"stations": stations, "updated_at": updated_at}


@app.get("/api/weather")
async def get_weather():
    raw = await fetch_weather()
    parsed = parse_weather(raw)
    return {"weather": parsed, "updated_at": datetime.now().isoformat(timespec="seconds")}


# ---------------------------------------------------------------------------

class JourneyQueryRequest(BaseModel):
    query:    str
    district: Optional[str] = None
    lat:      Optional[float] = None
    lon:      Optional[float] = None


@app.post("/api/journey-query")
async def journey_query(req: JourneyQueryRequest):
    """
    Agentic endpoint v2:
      1. Intent parse (district extraction)
      2. Parallel data fetch (AQI + Weather + Traffy)
      3. Chain-of-thought reasoning via Typhoon
      4. Return answer + reasoning separately
    """
    if not TYPHOON_API_KEY:
        raise HTTPException(status_code=503, detail="TYPHOON_API_KEY is not configured.")

    # ── 1. Resolve district ──────────────────────────────────────────────────
    district = req.district
    if not district:
        district = await extract_district_from_query(req.query)
    district_detected = district or "ไม่ระบุเขต"

    # ── 2. Parallel data fetch ───────────────────────────────────────────────
    aqi_task     = asyncio.create_task(fetch_air4thai())
    weather_task = asyncio.create_task(fetch_weather())

    # Traffy is sync but fast (in-memory DataFrame)
    district_df     = get_district_traffy(district) if district else pd.DataFrame()
    open_count      = len(district_df)
    streets         = top_streets(district_df)
    repeat_clusters = count_repeat_clusters(district_df)

    # Await async tasks
    raw_aqi, raw_weather = await asyncio.gather(
        aqi_task, weather_task, return_exceptions=True
    )

    # Parse AQI
    stations  = parse_air4thai(raw_aqi) if isinstance(raw_aqi, dict) else []
    pm25_value = None
    aqi_level  = "ไม่มีข้อมูล"
    aqi_context = ""
    if stations:
        centroid = district_centroid(district) if district else None
        if centroid:
            matched = nearest_station(stations, *centroid)
        else:
            matched = next((s for s in stations if district and district in s["district"]), stations[0])
        pm25_value = matched["pm25_ugm3"]
        aqi_level, _ = aqi_label(pm25_value)
        aqi_context = (
            f"คุณภาพอากาศ PM2.5 จากสถานี {matched['name_th']} ({matched['district']}): "
            f"{pm25_value:.1f} µg/m³ ระดับ{aqi_level}"
        )

    # Parse Weather
    weather      = parse_weather(raw_weather) if isinstance(raw_weather, dict) else {}
    weather_ctx  = weather_to_context(weather)

    # Traffy context
    traffy_context = ""
    if district:
        traffy_context = (
            f"รายงานปัญหาถนน/ทางเท้าที่ยังค้างแก้ในเขต{district}: {open_count} รายการ"
            f" (ซ้ำซาก {repeat_clusters} จุด)"
        )
        if streets:
            traffy_context += f" ถนนที่มีปัญหาบ่อย: {', '.join(streets)}"

    # Compound risk score (computed in Python — not by LLM)
    risk = compute_risk_score(
        pm25        = pm25_value or 0.0,
        open_tickets = open_count,
        repeat_clusters = repeat_clusters,
        rain_risk   = weather.get("rain_risk", ""),
    )
    risk_context = (
        f"Compound Risk Score (คำนวณโดย Python ไม่ใช่ LLM): {risk['score']}/10 ระดับ{risk['level']}"
        f" [PM2.5:{risk['components']['aqi']}/4 | ถนน/ซ้ำซาก:{risk['components']['traffy']}/4 | ฝน:{risk['components']['weather']}/2]"
    )

    # ── 3. Build chain-of-thought prompt ────────────────────────────────────
    data_block = "\n".join(filter(None, [aqi_context, weather_ctx, traffy_context, risk_context]))
    if not data_block:
        data_block = "ไม่มีข้อมูลเพิ่มเติมในขณะนี้"

    system_prompt = """คุณเป็น Bangkok Journey Intelligence — ระบบ AI วิเคราะห์ความปลอดภัยในการเดินทางกรุงเทพฯ
คุณมีความเชี่ยวชาญด้าน: คุณภาพอากาศ, สภาพถนน, สภาพอากาศ, และความเสี่ยงในการเดินทาง

ข้อมูลที่ได้รับมีค่า Compound Risk Score ที่คำนวณแล้วโดย Python (ไม่ใช่ LLM) — ห้ามสร้างตัวเลขใหม่ ให้อ้างอิงตัวเลขที่ให้มาเท่านั้น

กรุณาตอบในรูปแบบต่อไปนี้เท่านั้น (ต้องมีทั้ง 2 ส่วน):

[วิเคราะห์]
1. ตีความคำถาม: ผู้ใช้ต้องการทำอะไร? เส้นทางหรือพื้นที่ไหน? กลุ่มเสี่ยงอะไร?
2. ข้อมูลที่ได้รับ: สรุปข้อมูลจากแต่ละแหล่ง (AQI / สภาพอากาศ / Traffy)
3. ความเสี่ยงที่ระบุ: ระบุความเสี่ยงเฉพาะที่พบ พร้อมเหตุผล
4. Compound Risk Score: อ้างอิงคะแนนที่คำนวณมาแล้ว อธิบายว่าแต่ละ component ส่งผลอย่างไร
5. ข้อสรุป: ปลอดภัย / ควรระวัง / ไม่แนะนำ — เพราะอะไร?
[/วิเคราะห์]

[คำตอบ]
ตอบเป็นภาษาไทยกระชับ เป็นธรรมชาติ ไม่เกิน 5 ประโยค
ระบุความเสี่ยงหลัก + คำแนะนำปฏิบัติได้จริง + อ้างอิงแหล่งข้อมูล
[/คำตอบ]"""

    user_prompt = (
        f"คำถาม: {req.query}\n\n"
        f"ข้อมูลล่าสุด ณ เวลานี้:\n{data_block}\n\n"
        "วิเคราะห์และตอบตามรูปแบบที่กำหนด"
    )

    # ── 4. Call Typhoon with CoT prompt ─────────────────────────────────────
    try:
        raw_response = await call_typhoon([
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ], max_tokens=1200)
    except Exception as e:
        logger.error("Typhoon call failed: %s", e)
        raise HTTPException(status_code=502, detail=f"Typhoon LLM error: {e}")

    reasoning, answer = parse_cot_response(raw_response)

    return {
        "answer":            answer,
        "reasoning":         reasoning,
        "district_detected": district_detected,
        "open_tickets_count": open_count,
        "repeat_clusters":   repeat_clusters,
        "compound_risk":     risk,
        "aqi_level":         aqi_level,
        "pm25_ugm3":         pm25_value,
        "top_streets":       streets,
        "weather":           weather,
        "sources":           ["Traffy Fondue", "Air4Thai/PCD", "wttr.in Weather", "Typhoon ThaiLLM"],
    }


# ---------------------------------------------------------------------------

@app.get("/api/traffy/status")
async def traffy_status():
    """Return last refresh time and ticket count."""
    status = {"open_tickets": len(traffy_df) if not traffy_df.empty else 0}
    if os.path.exists(STATUS_PATH):
        with open(STATUS_PATH, encoding="utf-8") as f:
            status.update(json.load(f))
    else:
        status["last_updated"] = "manual snapshot (no auto-refresh yet)"
    return status


@app.get("/api/traffy/district/{district_name}")
async def traffy_district(district_name: str):
    if traffy_df.empty:
        raise HTTPException(status_code=503, detail="Traffy data not loaded")
    district_df   = get_district_traffy(district_name)
    if district_df.empty:
        return {"district": district_name, "open_tickets_count": 0, "top_streets": [], "type_breakdown": {}}
    type_breakdown = district_df["type"].value_counts().to_dict()
    streets        = top_streets(district_df, n=5)
    return {
        "district":           district_name,
        "open_tickets_count": len(district_df),
        "top_streets":        streets,
        "type_breakdown":     type_breakdown,
    }


@app.get("/api/traffy/geojson")
async def traffy_geojson():
    if traffy_df.empty:
        return Response(
            content=b'{"type":"FeatureCollection","features":[]}',
            media_type="application/json",
        )

    etag = _parquet_etag()
    if etag and _geojson_cache["etag"] == etag and _geojson_cache["body"] is not None:
        return Response(content=_geojson_cache["body"], media_type="application/json")

    df = traffy_df.copy()
    df["_lat4"] = df["lat"].round(4)
    df["_lon4"] = df["lon"].round(4)
    coord_counts = df.groupby(["_lat4", "_lon4"]).size()

    features = []
    for row in df.itertuples(index=False):
        lat = row.lat; lon = row.lon
        if not lat or not lon or lat == 0.0 or lon == 0.0:
            continue
        is_repeat = coord_counts.get((round(lat, 4), round(lon, 4)), 0) >= 3
        days_val  = int(row.days) if not pd.isna(row.days) else 0
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {
                "cg":          str(row.cg or ""),
                "district":    str(row.district or ""),
                "subdistrict": str(row.subdistrict or ""),
                "address":     str(row.address or "")[:120],
                "days":        days_val,
                "is_repeat":   bool(is_repeat),
            },
        })

    import json as _json
    body = _json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False).encode()
    _geojson_cache["etag"] = etag
    _geojson_cache["body"] = body
    return Response(content=body, media_type="application/json")


# ---------------------------------------------------------------------------

@app.get("/api/digest")
async def get_digest():
    """Latest weekly BMA district digest generated by weekly_digest.py."""
    if not os.path.exists(DIGEST_PATH):
        return {"available": False, "message": "ยังไม่มี digest — รันทุกวันจันทร์ 02:00 BKK"}
    with open(DIGEST_PATH, encoding="utf-8") as f:
        return {**json.load(f), "available": True}


@app.get("/api/traffy/export.csv")
async def traffy_export_csv():
    """Download full backlog as CSV (UTF-8 BOM for Excel compatibility)."""
    if traffy_df.empty:
        raise HTTPException(status_code=503, detail="Traffy data not loaded")
    csv_bytes = traffy_df.to_csv(index=False).encode("utf-8-sig")
    return Response(
        content=csv_bytes,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=bangkok-road-backlog.csv"},
    )


@app.get("/api/clusters")
async def get_clusters():
    """District Debt Screener — ranked by debt_score = repeat_clusters × log1p(median_days)."""
    if traffy_df.empty:
        return {"districts": [], "updated_at": datetime.now().isoformat(timespec="seconds")}

    # Load previous district counts for Δ
    dist_prev: dict = {}
    if os.path.exists(STATUS_PATH):
        with open(STATUS_PATH, encoding="utf-8") as f:
            st = json.load(f)
        dist_prev = st.get("district_counts_prev", {})

    df = traffy_df.copy()
    results = []

    for district, group in df.groupby("district"):
        if not district or str(district) in ("nan", "None", ""):
            continue
        total = int(len(group))
        days_valid = group["days"][group["days"] > 0]
        median_days = int(days_valid.median()) if len(days_valid) > 0 else 0

        coord_counts = group.groupby(
            [group["lat"].round(4), group["lon"].round(4)]
        ).size()
        repeat_clusters = int((coord_counts >= 3).sum())

        debt_score = round(repeat_clusters * math.log1p(median_days), 1)
        delta = total - dist_prev.get(district, total)

        r6 = risk6_map.get(district, {})
        results.append({
            "district":          district,
            "total_open":        total,
            "median_days":       median_days,
            "repeat_clusters":   repeat_clusters,
            "debt_score":        debt_score,
            "delta":             delta,
            "compound_risk_6l":  r6.get("score_x10"),   # 0–10 (None if no data)
            "compound_risk_rank": r6.get("rank"),
            "compound_risk_layers": r6.get("layers"),
        })

    results.sort(key=lambda x: x["debt_score"], reverse=True)
    return {
        "districts":   results,
        "updated_at":  datetime.now().isoformat(timespec="seconds"),
        "has_delta":   bool(dist_prev),
    }


# Seasonal ticket volume factors derived from 5-year historical CSV (361,438 tickets)
_SEASONAL = {1:0.85,2:0.80,3:0.90,4:1.05,5:1.20,6:1.50,7:1.80,8:2.10,9:2.30,10:1.70,11:1.10,12:0.90}
_MONTH_TH  = {1:"ม.ค.",2:"ก.พ.",3:"มี.ค.",4:"เม.ย.",5:"พ.ค.",6:"มิ.ย.",
               7:"ก.ค.",8:"ส.ค.",9:"ก.ย.",10:"ต.ค.",11:"พ.ย.",12:"ธ.ค."}


@app.get("/api/district/trend")
async def district_trend(district: str):
    """Monthly breakdown of WHEN currently-open tickets were reported."""
    if traffy_df.empty:
        raise HTTPException(status_code=503, detail="Traffy data not loaded")
    df = get_district_traffy(district)
    if df.empty:
        return {"district": district, "months": [], "older_count": 0, "total": 0}

    now = datetime.now()
    monthly: dict = {}
    older_count = 0

    cutoff_year  = now.year
    cutoff_month = now.month - 11  # 12 months including current
    if cutoff_month <= 0:
        cutoff_month += 12
        cutoff_year -= 1

    for ts_str in df["timestamp"]:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(str(ts_str)[:19], fmt)
                key = f"{dt.year}-{dt.month:02d}"
                # Is this ticket older than 12 months?
                if (dt.year, dt.month) < (cutoff_year, cutoff_month):
                    older_count += 1
                else:
                    monthly[key] = monthly.get(key, 0) + 1
                break
            except (ValueError, TypeError):
                pass

    # Build last 12 months list
    months = []
    for i in range(11, -1, -1):
        m = now.month - i
        y = now.year
        while m <= 0:
            m += 12
            y -= 1
        key = f"{y}-{m:02d}"
        months.append({"key": key, "label": _MONTH_TH[m], "count": monthly.get(key, 0)})

    return {
        "district":    district,
        "months":      months,
        "older_count": older_count,
        "total":       len(df),
    }


@app.get("/api/route-risk")
async def route_risk_api(from_district: str, to_district: str):
    """Compound Risk Score for a route between two district centroids."""
    if traffy_df.empty:
        raise HTTPException(status_code=503, detail="Traffy data not loaded")

    from_c = district_centroid(from_district)
    to_c   = district_centroid(to_district)
    if not from_c or not to_c:
        raise HTTPException(status_code=404, detail="ไม่พบพิกัดสำหรับเขตที่ระบุ")

    # Sample 10 waypoints along straight-line route
    N = 9
    waypoints = [
        (from_c[0] + (i / N) * (to_c[0] - from_c[0]),
         from_c[1] + (i / N) * (to_c[1] - from_c[1]))
        for i in range(N + 1)
    ]

    # Bounding box around entire route + 600m buffer
    RADIUS_KM = 0.6
    PAD = RADIUS_KM / 111.0 + 0.003
    min_lat = min(w[0] for w in waypoints) - PAD
    max_lat = max(w[0] for w in waypoints) + PAD
    min_lon = min(w[1] for w in waypoints) - PAD
    max_lon = max(w[1] for w in waypoints) + PAD

    df_route = traffy_df[
        (traffy_df["lat"] >= min_lat) & (traffy_df["lat"] <= max_lat) &
        (traffy_df["lon"] >= min_lon) & (traffy_df["lon"] <= max_lon)
    ].copy()

    # For each waypoint collect nearby tickets (bounding-box approximation)
    nearby_mask = pd.Series(False, index=df_route.index)
    wp_counts = []
    for lat, lon in waypoints:
        dlat = RADIUS_KM / 111.0
        dlon = RADIUS_KM / (111.0 * math.cos(math.radians(lat)))
        box = (
            (df_route["lat"] >= lat - dlat) & (df_route["lat"] <= lat + dlat) &
            (df_route["lon"] >= lon - dlon) & (df_route["lon"] <= lon + dlon)
        )
        wp_counts.append({"lat": round(lat, 5), "lon": round(lon, 5), "nearby": int(box.sum())})
        nearby_mask |= box

    df_near = df_route[nearby_mask]
    total_nearby   = len(df_near)
    repeat_along   = count_repeat_clusters(df_near)

    # AQI + weather at route midpoint
    mid_lat = (from_c[0] + to_c[0]) / 2
    mid_lon = (from_c[1] + to_c[1]) / 2

    raw_aqi, raw_weather = await asyncio.gather(
        fetch_air4thai(), fetch_weather(), return_exceptions=True
    )

    pm25 = 0.0
    station_name = None
    if isinstance(raw_aqi, dict):
        stations = parse_air4thai(raw_aqi)
        if stations:
            st = nearest_station(stations, mid_lat, mid_lon)
            pm25 = st["pm25_ugm3"]
            station_name = st["name_th"]

    weather = parse_weather(raw_weather) if isinstance(raw_weather, dict) else {}
    risk = compute_risk_score(
        pm25=pm25,
        open_tickets=total_nearby,
        repeat_clusters=repeat_along,
        rain_risk=weather.get("rain_risk", ""),
    )

    return {
        "from_district":   from_district,
        "to_district":     to_district,
        "route_km":        round(haversine(from_c[0], from_c[1], to_c[0], to_c[1]), 1),
        "compound_risk":   risk,
        "pm25_ugm3":       pm25,
        "aqi_station":     station_name,
        "total_tickets":   total_nearby,
        "repeat_clusters": repeat_along,
        "weather":         weather,
        "waypoints":       wp_counts,
    }

@app.get("/api/compound-risk")
async def compound_risk_all():
    """
    6-layer compound risk scores for all 50 districts.
    Computed from: Traffy 5Y + BMA flood data + Win motorcycle proxy + Air4Thai + CoolingCenter + Greenpark.
    Source of truth: compound_risk_by_district.csv (run 2026-06-23).
    """
    districts = []
    for district, data in sorted(risk6_map.items(), key=lambda x: x[1]["score_x10"], reverse=True):
        districts.append({
            "district":    district,
            "score_x10":   data["score_x10"],
            "rank":        data["rank"],
            "layers":      data["layers"],
        })
    return {
        "districts": districts,
        "n":         len(districts),
        "note":      "6-layer district baseline จาก 361,438 รายงาน 5 ปี + 5 แหล่งข้อมูล — ไม่ใช่ real-time",
        "layers_desc": {
            "traffy_vol":  "ปริมาณรายงานถนน/ทางเท้า Traffy Fondue (5 ปี)",
            "traffy_dur":  "ระยะเวลาซ่อม median (Traffy เสร็จสิ้น)",
            "flood":       "จุดเสี่ยงน้ำท่วม BMA (737 จุด)",
            "safety":      "ความปลอดภัย (สถานีวินมอไซค์ + จุดเสี่ยง BMA)",
            "air":         "คุณภาพอากาศ PM2.5 Air4Thai (79 สถานี)",
            "heat":        "ความร้อน/สีเขียว (CoolingCenter + Greenpark inverse)",
        },
        "demo_districts": {
            "ลาดกระบัง": risk6_map.get("ลาดกระบัง", {}).get("score_x10"),
            "วัฒนา":      risk6_map.get("วัฒนา",      {}).get("score_x10"),
            "ทวีวัฒนา":  risk6_map.get("ทวีวัฒนา",  {}).get("score_x10"),
        },
    }


@app.get("/api/compound-risk/{district_name}")
async def compound_risk_district(district_name: str):
    """6-layer compound risk for a specific district."""
    data = risk6_map.get(district_name)
    if not data:
        raise HTTPException(status_code=404, detail=f"ไม่พบเขต: {district_name}")
    return {"district": district_name, **data}


@app.get("/api/seasonal")
async def seasonal_forecast():
    """Seasonal backlog forecast based on 5-year historical patterns."""
    current_month = datetime.now().month
    forecast = [
        {
            "month":      m,
            "month_th":   _MONTH_TH[m],
            "factor":     _SEASONAL[m],
            "is_current": m == current_month,
            "is_rainy":   m in (5,6,7,8,9,10),
        }
        for m in range(1, 13)
    ]
    return {
        "current_month":   current_month,
        "current_month_th": _MONTH_TH[current_month],
        "current_factor":  _SEASONAL[current_month],
        "peak_factor":     2.30,
        "peak_month_th":   "ก.ย.",
        "forecast":        forecast,
        "note":            "คาดการณ์เชิงสถิติจากข้อมูล 5 ปีย้อนหลัง (361,438 รายการ) · ไม่ใช่การพยากรณ์จาก LLM",
    }


# ---------------------------------------------------------------------------
# Static — must be last
# ---------------------------------------------------------------------------
STATIC_DIR = os.path.dirname(os.path.abspath(__file__))

@app.get("/")
async def root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return {
        "service": "Bangkok Journey Intelligence API",
        "version": "2.0.0",
        "traffy_open_tickets": len(traffy_df) if not traffy_df.empty else 0,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
