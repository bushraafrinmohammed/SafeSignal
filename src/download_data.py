"""
Download Alcon medical-device adverse event reports from the FDA MAUDE database
via the public openFDA API (no API key needed for small pulls).

openFDA requires a free API key: https://open.fda.gov/apis/authentication/
Pass it with --api-key or set the environment variable OPENFDA_API_KEY.

Usage:
  python src/download_data.py --api-key YOUR_KEY
  python src/download_data.py --start 20220101 --end 20251231 --max 6000 --api-key YOUR_KEY
Output:
  data/alcon_events.csv
"""
import argparse
import os
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://api.fda.gov/device/event.json"
PAGE = 1000  # openFDA max per request


def first(lst, key, default=""):
    return (lst[0].get(key, default) if lst else default) or default


def parse(rec: dict) -> dict:
    dev = rec.get("device", [])
    texts = rec.get("mdr_text", [])
    event_desc = " ".join(t.get("text", "") for t in texts
                          if "Description of Event" in t.get("text_type_code", ""))
    openfda = (dev[0].get("openfda", {}) if dev else {}) or {}
    return {
        "report_id": rec.get("mdr_report_key"),
        "date_received": rec.get("date_received"),
        "event_type": rec.get("event_type"),
        "brand_name": first(dev, "brand_name"),
        "generic_name": first(dev, "generic_name"),
        "product_code": first(dev, "device_report_product_code"),
        "device_class_name": openfda.get("device_name", ""),
        "product_problems": "; ".join(rec.get("product_problems", []) or []),
        "event_text": event_desc.strip(),
    }


def main(start: str, end: str, max_records: int, api_key: str):
    if not api_key:
        raise SystemExit("openFDA needs a free API key.\n  1. Get one at https://open.fda.gov/apis/authentication/\n"
                         "  2. Run: python src/download_data.py --api-key YOUR_KEY")
    search = f"device.manufacturer_d_name:ALCON+AND+date_received:[{start}+TO+{end}]"
    rows, skip = [], 0
    while skip < max_records and skip <= 25000:  # openFDA skip limit
        url = f"{BASE}?api_key={api_key}&search={search}&sort=date_received:desc&limit={PAGE}&skip={skip}"
        r = requests.get(url, timeout=120, headers={"User-Agent": "SafeSignal-portfolio-project"})
        if r.status_code == 404:  # no more results
            break
        if r.status_code in (401, 403):
            raise SystemExit(f"openFDA refused the request ({r.status_code}). Check that your API key is correct.\n"
                             f"Server said: {r.text[:300]}")
        r.raise_for_status()
        batch = r.json().get("results", [])
        if not batch:
            break
        rows += [parse(x) for x in batch]
        skip += PAGE
        print(f"  fetched {len(rows):,}")
        time.sleep(0.5)  # stay well under the rate limit
    df = pd.DataFrame(rows)
    out = ROOT / "data" / "alcon_events.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df):,} reports to {out}")
    print(df["event_type"].value_counts())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20240101")
    ap.add_argument("--end", default=time.strftime("%Y%m%d"))
    ap.add_argument("--max", type=int, default=6000)
    ap.add_argument("--api-key", default=os.environ.get("OPENFDA_API_KEY", ""))
    a = ap.parse_args()
    main(a.start, a.end, a.max, a.api_key)
