"""
seed_dev_data.py
================
Seeds the Bill Vault with a sample homeowner history for quick demoing.

SAFETY GATE: Only runs when APP_ENV=development.
Never run this in production — it inserts synthetic records.

Injected scenario:
  - HOME001 / Rahul Sharma – 6 months of normal bills
  - Month 2024-04 has a cross-month continuity error:
      March current_reading (1950) ≠ April prev_reading (1980)
      This is detected by fraud_engine.py Rule 2.
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

APP_ENV = os.getenv("APP_ENV", "production").strip().lower()

if APP_ENV != "development":
    print(
        "[seed_dev_data] Skipped – APP_ENV is not 'development'. "
        "Set APP_ENV=development in your .env to run seeding."
    )
    sys.exit(0)

# Only import after the gate
from bill_vault import init_db, insert_bill, DB_PATH  # noqa: E402

SAMPLE_BILLS = [
    # (billing_month, prev_reading, current_reading, billed_units, amount_paid, profile)
    ("2024-01", 1200.0, 1450.0, 250.0, 1375.0, "2 BHK / 1 AC"),
    ("2024-02", 1450.0, 1680.0, 230.0, 1265.0, "2 BHK / 1 AC"),
    ("2024-03", 1680.0, 1950.0, 270.0, 1485.0, "2 BHK / 1 AC"),
    # ↑ March current = 1950
    # ↓ April prev = 1980  ← CONTINUITY ERROR (should be 1950)
    ("2024-04", 1980.0, 2250.0, 270.0, 1485.0, "2 BHK / 1 AC"),
    ("2024-05", 2250.0, 2510.0, 260.0, 1430.0, "2 BHK / 1 AC"),
    ("2024-06", 2510.0, 2780.0, 270.0, 1485.0, "2 BHK / 1 AC"),
]

def seed():
    init_db(DB_PATH)
    inserted = 0
    for month, prev_r, curr_r, units, amt, profile in SAMPLE_BILLS:
        insert_bill(
            consumer_id="HOME001",
            consumer_name="Rahul Sharma",
            billing_month=month,
            prev_reading=prev_r,
            current_reading=curr_r,
            billed_units=units,
            amount_paid=amt,
            tariff_type="Domestic",
            household_profile=profile,
        )
        inserted += 1
        print(f"  [seed] Inserted {month}")

    print(f"[seed_dev_data] Done – {inserted} bills seeded for HOME001 (Rahul Sharma).")
    print("[seed_dev_data] NOTE: Month 2024-04 has a deliberate continuity error "
          "(March current=1950 vs April prev=1980).")


if __name__ == "__main__":
    seed()
