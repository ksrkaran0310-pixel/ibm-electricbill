"""
data_generator.py
=================
Generates data/utility_billing_records.csv with:
  - ~300 consumers × 12 months (Domestic, Commercial, Industrial)
  - Realistic seasonal variation
  - Labeled anomaly types:
      1. meter_bypass   – sudden >=55% drop vs peer average
      2. bill_mismatch  – billed_units != (current_reading - prev_reading)
      3. flatline       – >=2 consecutive 0-kWh months
      4. spike          – >=250% usage spike
  - Normal winter/monsoon zone-wide dips (must NOT trigger false positives)
"""

import os
import random
import numpy as np
import pandas as pd

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_CONSUMERS = 300
MONTHS = list(range(1, 13))          # Jan–Dec
ZONES = ["Domestic", "Commercial", "Industrial"]
TARIFF_MAP = {
    "Domestic":    {"rate": 5.5,  "base_kwh": 250,  "std": 60},
    "Commercial":  {"rate": 8.0,  "base_kwh": 800,  "std": 180},
    "Industrial":  {"rate": 6.5,  "base_kwh": 4000, "std": 800},
}

# Seasonal multipliers per month (index 0 = Jan … 11 = Dec)
# Slight dip in winter (Jan, Feb) and monsoon (Jul, Aug) – zone-wide, NOT fraud
SEASONAL = [0.88, 0.85, 0.95, 1.00, 1.05, 1.08, 0.90, 0.88, 1.02, 1.05, 0.98, 0.92]

# Anomaly injection budget (as fraction of consumers)
FRAC_BYPASS   = 0.07   # ~21 consumers
FRAC_MISMATCH = 0.06   # ~18 consumers
FRAC_FLATLINE = 0.04   # ~12 consumers
FRAC_SPIKE    = 0.05   # ~15 consumers

OUTPUT_DIR  = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "utility_billing_records.csv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assign_consumers():
    """Return a list of consumer dicts with id, name, zone, tariff."""
    consumers = []
    zone_cycle = ZONES * (NUM_CONSUMERS // len(ZONES) + 1)
    for i in range(NUM_CONSUMERS):
        zone = zone_cycle[i]
        consumers.append({
            "consumer_id":   f"CON{i+1:04d}",
            "consumer_name": f"Consumer_{i+1}",
            "zone":          zone,
            "tariff_type":   zone,
        })
    return consumers


def _base_usage(zone: str, month: int) -> float:
    """Return realistic monthly kWh usage with seasonal adjustment."""
    cfg = TARIFF_MAP[zone]
    seasonal_factor = SEASONAL[month - 1]
    usage = max(0.0, np.random.normal(cfg["base_kwh"] * seasonal_factor, cfg["std"]))
    return round(usage, 2)


def _amount(zone: str, units: float) -> float:
    rate = TARIFF_MAP[zone]["rate"]
    return round(units * rate, 2)


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    consumers = _assign_consumers()

    # Pick anomaly sets (non-overlapping consumer IDs)
    all_ids = [c["consumer_id"] for c in consumers]
    random.shuffle(all_ids)
    idx = 0

    n_bypass   = int(NUM_CONSUMERS * FRAC_BYPASS)
    n_mismatch = int(NUM_CONSUMERS * FRAC_MISMATCH)
    n_flatline = int(NUM_CONSUMERS * FRAC_FLATLINE)
    n_spike    = int(NUM_CONSUMERS * FRAC_SPIKE)

    bypass_ids   = set(all_ids[idx: idx + n_bypass]);   idx += n_bypass
    mismatch_ids = set(all_ids[idx: idx + n_mismatch]); idx += n_mismatch
    flatline_ids = set(all_ids[idx: idx + n_flatline]); idx += n_flatline
    spike_ids    = set(all_ids[idx: idx + n_spike]);    idx += n_spike

    # Decide anomaly month per consumer (months 4–10 to avoid seasonal overlap)
    anomaly_month = {cid: random.randint(4, 10) for cid in
                     bypass_ids | mismatch_ids | flatline_ids | spike_ids}

    rows = []

    for consumer in consumers:
        cid  = consumer["consumer_id"]
        zone = consumer["zone"]
        prev_reading = round(random.uniform(100, 2000), 1)
        prev_usage   = None   # tracks last month's billed units for flatline check

        for month in MONTHS:
            # Normal seasonal usage
            usage = _base_usage(zone, month)

            anomaly_label = "normal"

            # --- Anomaly injection ---

            if cid in bypass_ids and month == anomaly_month[cid]:
                # Sudden >=55% drop (meter bypass / tampering)
                usage = round(usage * random.uniform(0.10, 0.44), 2)
                anomaly_label = "meter_bypass"

            elif cid in flatline_ids and month in (
                anomaly_month[cid], anomaly_month[cid] + 1
            ):
                # Two consecutive 0-kWh months
                usage = 0.0
                anomaly_label = "flatline"

            elif cid in spike_ids and month == anomaly_month[cid]:
                # >=250% spike
                usage = round(usage * random.uniform(2.5, 4.0), 2)
                anomaly_label = "spike"

            current_reading = round(prev_reading + usage, 1)
            billed_units    = round(usage, 2)

            if cid in mismatch_ids and month == anomaly_month[cid]:
                # Bill shows fewer units than actually consumed
                billed_units  = round(usage * random.uniform(0.40, 0.75), 2)
                anomaly_label = "bill_mismatch"

            amount_paid = _amount(zone, billed_units)

            rows.append({
                "consumer_id":      cid,
                "consumer_name":    consumer["consumer_name"],
                "zone":             zone,
                "tariff_type":      consumer["tariff_type"],
                "billing_month":    month,
                "prev_reading":     round(prev_reading, 1),
                "current_reading":  round(current_reading, 1),
                "billed_units":     billed_units,
                "amount_paid":      amount_paid,
                "anomaly_label":    anomaly_label,
            })

            prev_reading = current_reading
            prev_usage   = usage

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"[data_generator] Saved {len(df)} rows -> {OUTPUT_FILE}")

    # Quick sanity summary
    label_counts = df["anomaly_label"].value_counts()
    print("[data_generator] Anomaly label distribution:")
    for label, count in label_counts.items():
        print(f"  {label:20s}: {count}")

    # Confirm winter dip exists but is zone-wide (not flagged as bypass)
    winter = df[(df["billing_month"].isin([1, 2])) & (df["anomaly_label"] == "normal")]
    zone_avgs = winter.groupby("zone")["billed_units"].mean()
    print("[data_generator] Zone avg kWh in winter (Jan-Feb) – should show natural dip:")
    print(zone_avgs.to_string())

    return df


if __name__ == "__main__":
    generate()
