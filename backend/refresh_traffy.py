"""
Nightly Traffy Fondue refresh
Fetches last 14 days of tickets from public API → merges with existing parquet.

Cron (2 AM Bangkok = 19:00 UTC):
  0 19 * * * /usr/bin/python3 /home/ec2-user/refresh_traffy.py >> /home/ec2-user/refresh.log 2>&1

Endpoint discovered from bkkchangelog (creatorsgarten/bkkchangelog):
  https://publicapi.traffy.in.th/share/teamchadchart/search
  params: limit, offset, last_activity_start, last_activity_end
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta, timezone
import urllib.request
import urllib.error
import urllib.parse

import pandas as pd

# ---------------------------------------------------------------------------
PARQUET_PATH  = os.path.join(os.path.dirname(__file__), "filtered_tickets.parquet")
PARQUET_TMP   = PARQUET_PATH + ".tmp"
STATUS_PATH   = os.path.join(os.path.dirname(__file__), "traffy_status.json")
TRAFFY_SEARCH = "https://publicapi.traffy.in.th/share/teamchadchart/search"
ROAD_KEYWORDS = {"ถนน", "ทางเท้า", "สะพาน", "ซอย", "ถ.", "ทางลาด"}
OPEN_TYPES    = {"ถนน", "ถนน/สะพาน", "ทางเท้า"}
CLOSED_STATE  = "เสร็จสิ้น"
FETCH_DAYS    = 14
PAGE_SIZE     = 1000
MAX_PAGES     = 200
SLEEP_BETWEEN = 2.0   # match bkkchangelog's 2s delay to be polite
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("refresh_traffy")


def parse_coords(coords) -> tuple[float, float]:
    """coords is either a list [lon_str, lat_str] or a WKT string."""
    try:
        if isinstance(coords, (list, tuple)) and len(coords) >= 2:
            return float(coords[1]), float(coords[0])  # lat, lon
        s = str(coords)
        # WKT: POINT(lon lat)
        if "POINT" in s:
            import re
            m = re.search(r"POINT\s*\(([0-9.]+)\s+([0-9.]+)\)", s)
            if m:
                return float(m.group(2)), float(m.group(1))
        # comma-separated
        parts = s.split(",")
        if len(parts) == 2:
            return float(parts[0].strip()), float(parts[1].strip())
    except (ValueError, TypeError, IndexError):
        pass
    return 0.0, 0.0


def fetch_page(offset: int, start: str, end: str) -> dict:
    params = urllib.parse.urlencode({
        "limit": PAGE_SIZE,
        "offset": offset,
        "last_activity_start": start,
        "last_activity_end": end,
    })
    url = f"{TRAFFY_SEARCH}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "BangkokJourneyIntelligence/2.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_recent_tickets(days: int = FETCH_DAYS) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    start = (now - timedelta(days=days)).strftime("%Y-%m-%d")
    end   = (now + timedelta(days=1)).strftime("%Y-%m-%d")
    logger.info("Fetching tickets: %s → %s", start, end)

    rows = []
    offset = 0

    for page in range(MAX_PAGES):
        try:
            data = fetch_page(offset, start, end)
        except urllib.error.HTTPError as e:
            logger.warning("HTTP %s at offset %d — stopping", e.code, offset)
            break
        except Exception as e:
            logger.warning("Error at offset %d: %s — stopping", offset, e)
            break

        results = data.get("results", [])
        if not results:
            logger.info("No more results at offset %d (page %d)", offset, page + 1)
            break

        for r in results:
            lat, lon = parse_coords(r.get("coords"))
            # New API uses "type" for problem category (ถนน, ทางเท้า, etc.)
            problem_type = str(r.get("type") or r.get("problem_type_fondue") or "")
            rows.append({
                "ticket_id":   str(r.get("ticket_id", "")),
                "type":        problem_type,
                "cg":          problem_type,
                "address":     str(r.get("address", "")),
                "subdistrict": str(r.get("subdistrict", "")),
                "district":    str(r.get("district", "")),
                "timestamp":   str(r.get("timestamp", "")),
                "last_activity": str(r.get("last_activity", "")),
                "state":       str(r.get("state", "")),
                "lat":         lat,
                "lon":         lon,
            })

        offset += len(results)
        logger.info("  page %d: total fetched so far %d", page + 1, offset)

        if len(results) < PAGE_SIZE:
            break

        time.sleep(SLEEP_BETWEEN)

    logger.info("Fetched %d raw records", len(rows))
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def compute_days(df: pd.DataFrame) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    def _days(ts_str):
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(str(ts_str)[:19], fmt)
                dt = dt.replace(tzinfo=timezone.utc)
                return (now - dt).days
            except (ValueError, TypeError):
                pass
        return 0
    df = df.copy()
    df["days"] = df["timestamp"].apply(_days)
    return df


def merge_with_existing(new_df: pd.DataFrame) -> pd.DataFrame:
    if not os.path.exists(PARQUET_PATH):
        logger.warning("No existing parquet — using fetched data only")
        road = new_df[new_df["cg"].isin(OPEN_TYPES)].copy()
        return compute_days(road)

    existing = pd.read_parquet(PARQUET_PATH)
    logger.info("Existing: %d rows", len(existing))

    # New API has type=null; classify by description keywords if type missing
    def is_road(row):
        if row.get("cg") and row["cg"] not in ("None", "nan", ""):
            return row["cg"] in {"ถนน", "ถนน/สะพาน", "ทางเท้า"}
        desc = str(row.get("address", ""))
        return any(kw in desc for kw in ROAD_KEYWORDS)

    road_new = new_df[new_df.apply(is_road, axis=1)].copy()
    logger.info("New road/pavement tickets: %d", len(road_new))

    if road_new.empty:
        logger.info("No new road/pavement tickets — keeping existing parquet")
        # Still remove newly-closed tickets from existing set
        closed_ids = new_df[new_df["state"] == CLOSED_STATE]["ticket_id"]
        if len(closed_ids):
            existing = existing[~existing["ticket_id"].isin(closed_ids)].copy()
            logger.info("Removed %d closed tickets", len(closed_ids))
        return compute_days(existing)

    if "ticket_id" in existing.columns and "ticket_id" in road_new.columns:
        existing_clean = existing[~existing["ticket_id"].isin(road_new["ticket_id"])].copy()
        closed_ids = road_new[road_new["state"] == CLOSED_STATE]["ticket_id"]
        existing_clean = existing_clean[~existing_clean["ticket_id"].isin(closed_ids)]
        open_new = road_new[road_new["state"] != CLOSED_STATE]
        merged = pd.concat([existing_clean, open_new], ignore_index=True)
    else:
        merged = pd.concat([existing, road_new], ignore_index=True)

    merged = compute_days(merged)
    logger.info("Merged: %d rows", len(merged))
    return merged


def save_status(rows_before: int, rows_after: int, fetched: int,
                prev_df: "pd.DataFrame | None" = None,
                curr_df: "pd.DataFrame | None" = None):
    def _dist_counts(df):
        if df is None or df.empty or "district" not in df.columns:
            return {}
        return {k: int(v) for k, v in df["district"].value_counts().items()}

    status = {
        "last_updated":          datetime.now().isoformat(timespec="seconds"),
        "rows_before":           rows_before,
        "rows_after":            rows_after,
        "new_fetched":           fetched,
        "district_counts_prev":  _dist_counts(prev_df),
        "district_counts_curr":  _dist_counts(curr_df),
    }
    with open(STATUS_PATH, "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)
    logger.info("Status saved → %s", STATUS_PATH)


def main():
    logger.info("=== Traffy refresh started ===")
    t0 = time.time()

    prev_df = pd.read_parquet(PARQUET_PATH) if os.path.exists(PARQUET_PATH) else pd.DataFrame()
    rows_before = len(prev_df)

    try:
        new_df = fetch_recent_tickets(days=FETCH_DAYS)
    except Exception as e:
        logger.error("Refresh aborted: %s", e)
        save_status(rows_before, rows_before, 0, prev_df, prev_df)
        return

    if new_df.empty:
        logger.warning("No data fetched — keeping existing parquet")
        save_status(rows_before, rows_before, 0, prev_df, prev_df)
        return

    merged = merge_with_existing(new_df)
    rows_after = len(merged)
    merged.to_parquet(PARQUET_TMP, index=False)
    os.replace(PARQUET_TMP, PARQUET_PATH)
    logger.info("Saved: %d rows", rows_after)
    save_status(rows_before, rows_after, len(new_df), prev_df, merged)

    logger.info("=== Done in %.1f s | before=%d after=%d new=%d ===",
                time.time() - t0, rows_before, rows_after, len(new_df))


if __name__ == "__main__":
    main()
