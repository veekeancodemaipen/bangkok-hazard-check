"""
Weekly BMA District Digest
Generates a Thai-language summary of road/pavement backlog by district.

Cron (Mon 2AM Bangkok = Sun 19:00 UTC):
  0 19 * * 0 /usr/bin/python3 /home/ec2-user/weekly_digest.py >> /home/ec2-user/digest.log 2>&1
"""

import os
import json
import math
import logging
import urllib.request
import urllib.error
from datetime import datetime, timezone

import pandas as pd

PARQUET_PATH = os.path.join(os.path.dirname(__file__), "filtered_tickets.parquet")
DIGEST_PATH  = os.path.join(os.path.dirname(__file__), "weekly_digest.json")
TYPHOON_URL  = "https://api.opentyphoon.ai/v1/chat/completions"
TYPHOON_MODEL = "typhoon-v2.5-30b-a3b-instruct"
def _load_env(path="/home/ec2-user/.env"):
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())

_load_env()
TYPHOON_API_KEY = os.getenv("TYPHOON_API_KEY", "")
TOP_N = 8

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("weekly_digest")

_MONTH_TH = {1:"ม.ค.",2:"ก.พ.",3:"มี.ค.",4:"เม.ย.",5:"พ.ค.",6:"มิ.ย.",
              7:"ก.ค.",8:"ส.ค.",9:"ก.ย.",10:"ต.ค.",11:"พ.ย.",12:"ธ.ค."}
_SEASONAL  = {1:0.85,2:0.80,3:0.90,4:1.05,5:1.20,6:1.50,7:1.80,8:2.10,9:2.30,10:1.70,11:1.10,12:0.90}


def compute_top_districts(df: pd.DataFrame, n: int = TOP_N) -> list[dict]:
    results = []
    for district, group in df.groupby("district"):
        if not district or str(district) in ("nan", "None", ""):
            continue
        total = int(len(group))
        days_valid = group["days"][group["days"] > 0]
        median_days = int(days_valid.median()) if len(days_valid) > 0 else 0
        coord_counts = group.groupby([group["lat"].round(4), group["lon"].round(4)]).size()
        repeat_clusters = int((coord_counts >= 3).sum())
        debt_score = round(repeat_clusters * math.log1p(median_days), 1)
        results.append({
            "district": district,
            "total_open": total,
            "median_days": median_days,
            "repeat_clusters": repeat_clusters,
            "debt_score": debt_score,
        })
    results.sort(key=lambda x: x["debt_score"], reverse=True)
    return results[:n]


def call_typhoon(prompt: str) -> str:
    if not TYPHOON_API_KEY:
        return "(ไม่มี TYPHOON_API_KEY)"
    payload = json.dumps({
        "model": TYPHOON_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 800,
        "temperature": 0.4,
    }).encode()
    req = urllib.request.Request(
        TYPHOON_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {TYPHOON_API_KEY}",
        },
    )
    with urllib.request.urlopen(req, timeout=40) as resp:
        return json.loads(resp.read())["choices"][0]["message"]["content"]


def build_digest_prompt(top: list[dict], now: datetime) -> str:
    month_th = _MONTH_TH[now.month]
    factor   = _SEASONAL[now.month]
    lines = "\n".join(
        f"  {i+1}. {d['district']}: ค้างแก้ {d['total_open']:,} รายการ · "
        f"ซ้ำซาก {d['repeat_clusters']} จุด · median {d['median_days']} วัน · debt {d['debt_score']}"
        for i, d in enumerate(top)
    )
    return (
        f"คุณเป็นนักวิเคราะห์ข้อมูลเมือง กรุงเทพมหานคร\n"
        f"วันที่ {now.strftime('%d')} {month_th} {now.year + 543} · ช่วงนี้คาดว่าปริมาณร้องเรียนอยู่ที่ {factor}× baseline (ฤดูฝน)\n\n"
        f"ข้อมูล Traffy Fondue top {len(top)} เขตที่มี 'รายงานถนน/ทางเท้าค้างแก้' สูงสุด (วัดด้วย debt score):\n"
        f"{lines}\n\n"
        f"กรุณาเขียนรายงานสรุปรายสัปดาห์เป็นภาษาไทย ความยาวไม่เกิน 200 คำ:\n"
        f"- ระบุ 3 เขตที่น่ากังวลสูงสุด พร้อมเหตุผล\n"
        f"- ให้บริบทฤดูกาล (ฤดูฝน/แล้ง)\n"
        f"- ห้ามสร้างตัวเลขใหม่นอกจากที่ให้มา\n"
        f"- ห้ามอ้างว่า 'ลดอุบัติเหตุ' หรือ 'อันตราย' — ใช้ 'รายงานค้างแก้' เท่านั้น"
    )


def main():
    logger.info("=== Weekly digest started ===")
    now = datetime.now(timezone.utc)

    if not os.path.exists(PARQUET_PATH):
        logger.error("Parquet not found — abort")
        return

    df = pd.read_parquet(PARQUET_PATH)
    logger.info("Loaded %d rows", len(df))

    top = compute_top_districts(df)
    logger.info("Top %d districts computed", len(top))

    prompt = build_digest_prompt(top, now)
    logger.info("Calling Typhoon…")
    try:
        digest_text = call_typhoon(prompt)
    except Exception as e:
        logger.error("Typhoon call failed: %s", e)
        digest_text = "(Typhoon unavailable — ดูตารางด้านล่าง)"

    output = {
        "generated_at":  now.isoformat(timespec="seconds"),
        "week_of":       now.strftime("%Y-%m-%d"),
        "total_open":    int(len(df)),
        "top_districts": top,
        "digest":        digest_text,
    }
    with open(DIGEST_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info("Digest saved → %s", DIGEST_PATH)
    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
