"""
bill_vault.py
=============
SQLite-backed personal Bill Vault for homeowners / shopkeepers.

Security:
  - All queries use parameterized placeholders (never f-string SQL).
  - PDF extraction includes strict numeric plausibility guards.
  - No secrets stored here; DB path comes from caller or defaults to data/.
"""

import hashlib
import os
import re
import sqlite3
from pathlib import Path
from typing import Optional

DB_PATH = os.path.join("data", "bill_vault.db")

# ---------------------------------------------------------------------------
# Plausibility guards for PDF-extracted numbers
# ---------------------------------------------------------------------------
# Max sane kWh reading or units on a residential/commercial bill
MAX_KWH_READING = 999_999
MAX_KWH_UNITS   = 50_000     # monthly units; flag anything above as suspect
MAX_AMOUNT_INR  = 500_000    # ₹5 lakh upper bound per bill

# Patterns that look like phone numbers, dates, or consumer IDs — must be rejected
_PHONE_PATTERN    = re.compile(r"^\d{10,15}$")
_DATE_PATTERN     = re.compile(r"^\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}$")
_CONSUMER_ID_PAT  = re.compile(r"^[A-Za-z]+\d+$")   # e.g. CON0012


def _is_plausible_kwh(value: float) -> bool:
    """Return True if value could realistically be a kWh reading or units."""
    if value < 0:
        return False
    if value > MAX_KWH_READING:
        return False
    return True


def _is_plausible_units(value: float) -> bool:
    return 0.0 <= value <= MAX_KWH_UNITS


def _is_plausible_amount(value: float) -> bool:
    return 0.0 <= value <= MAX_AMOUNT_INR


def _reject_non_numeric_disguised(raw: str) -> bool:
    """Return True (= reject) if the string looks like a phone, date, or consumer ID."""
    s = raw.strip()
    if _PHONE_PATTERN.match(s):
        return True
    if _DATE_PATTERN.match(s):
        return True
    if _CONSUMER_ID_PAT.match(s):
        return True
    return False


# ---------------------------------------------------------------------------
# Database initialisation
# ---------------------------------------------------------------------------

def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: str = DB_PATH) -> None:
    """Create tables if they don't exist."""
    with get_connection(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS bill_records (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                consumer_id       TEXT    NOT NULL,
                consumer_name     TEXT    NOT NULL,
                billing_month     TEXT    NOT NULL,   -- 'YYYY-MM' format
                prev_reading      REAL    NOT NULL,
                current_reading   REAL    NOT NULL,
                billed_units      REAL    NOT NULL,
                amount_paid       REAL    NOT NULL,
                tariff_type       TEXT    NOT NULL,
                household_profile TEXT,               -- e.g. '2 BHK / 1 AC'
                source            TEXT    DEFAULT 'manual',  -- 'manual' | 'pdf'
                created_at        TEXT    DEFAULT (datetime('now')),
                UNIQUE(consumer_id, billing_month)
            )
        """)
        conn.commit()
    print(f"[bill_vault] DB ready -> {db_path}")


# ---------------------------------------------------------------------------
# CRUD operations (all parameterized)
# ---------------------------------------------------------------------------

def insert_bill(
    consumer_id: str,
    consumer_name: str,
    billing_month: str,
    prev_reading: float,
    current_reading: float,
    billed_units: float,
    amount_paid: float,
    tariff_type: str,
    household_profile: str = "",
    source: str = "manual",
    db_path: str = DB_PATH,
) -> int:
    """Insert or replace a bill record. Returns the row id."""
    # Basic plausibility guards
    if not _is_plausible_kwh(prev_reading):
        raise ValueError(f"prev_reading={prev_reading} outside plausible kWh range.")
    if not _is_plausible_kwh(current_reading):
        raise ValueError(f"current_reading={current_reading} outside plausible kWh range.")
    if not _is_plausible_units(billed_units):
        raise ValueError(f"billed_units={billed_units} outside plausible range.")
    if not _is_plausible_amount(amount_paid):
        raise ValueError(f"amount_paid={amount_paid} outside plausible ₹ range.")
    if current_reading < prev_reading:
        raise ValueError("current_reading cannot be less than prev_reading.")

    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO bill_records
                (consumer_id, consumer_name, billing_month, prev_reading,
                 current_reading, billed_units, amount_paid, tariff_type,
                 household_profile, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(consumer_id, billing_month)
            DO UPDATE SET
                consumer_name     = excluded.consumer_name,
                prev_reading      = excluded.prev_reading,
                current_reading   = excluded.current_reading,
                billed_units      = excluded.billed_units,
                amount_paid       = excluded.amount_paid,
                tariff_type       = excluded.tariff_type,
                household_profile = excluded.household_profile,
                source            = excluded.source
            """,
            (
                consumer_id, consumer_name, billing_month,
                prev_reading, current_reading, billed_units,
                amount_paid, tariff_type, household_profile, source,
            ),
        )
        conn.commit()
        return cursor.lastrowid


def get_bills(consumer_id: str, db_path: str = DB_PATH) -> list[dict]:
    """Return all bills for a consumer sorted by billing_month ascending."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT * FROM bill_records
            WHERE consumer_id = ?
            ORDER BY billing_month ASC
            """,
            (consumer_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_all_bills(db_path: str = DB_PATH) -> list[dict]:
    """Return all bill records."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM bill_records ORDER BY consumer_id, billing_month ASC"
        ).fetchall()
    return [dict(r) for r in rows]


def delete_bill(record_id: int, db_path: str = DB_PATH) -> None:
    """Delete a bill record by primary key."""
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM bill_records WHERE id = ?", (record_id,))
        conn.commit()


def get_bill_count(db_path: str = DB_PATH) -> int:
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT COUNT(*) FROM bill_records").fetchone()
    return row[0]


# ---------------------------------------------------------------------------
# Optional PDF text extractor
# ---------------------------------------------------------------------------

def extract_bill_from_pdf(pdf_path: str) -> dict:
    """
    Extract numeric fields from a PDF electricity bill.

    Returns a dict with keys:
        prev_reading, current_reading, billed_units, amount_paid
        (all validated; unrecognised / implausible fields are None)

    Raises ImportError if pypdf is not installed.
    Raises ValueError for invalid file path / size.
    """
    # Size guard: reject files > 10 MB
    MAX_SIZE_BYTES = 10 * 1024 * 1024
    path = Path(pdf_path)
    if not path.exists():
        raise ValueError(f"PDF not found: {pdf_path}")
    if path.stat().st_size > MAX_SIZE_BYTES:
        raise ValueError("PDF exceeds 10 MB size limit.")
    if path.suffix.lower() != ".pdf":
        raise ValueError("Only .pdf files are accepted.")

    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError("pypdf is required for PDF extraction. Install it via requirements.txt.") from exc

    reader  = PdfReader(str(path))
    full_text = "\n".join(page.extract_text() or "" for page in reader.pages)

    # Extract all standalone numeric tokens (int or float)
    raw_numbers = re.findall(r"\b(\d+(?:\.\d+)?)\b", full_text)

    # Filter out phone/date/consumer-ID patterns from raw text tokens
    raw_tokens = re.findall(r"\b[\dA-Za-z/\-]+\b", full_text)
    blocked = {t for t in raw_tokens if _reject_non_numeric_disguised(t)}

    candidates = []
    for n in raw_numbers:
        # Skip if the token appeared in a blocked context
        if n in blocked:
            continue
        v = float(n)
        candidates.append(v)

    # Heuristic extraction: pick plausible numeric fields by magnitude
    kwh_candidates    = [v for v in candidates if _is_plausible_kwh(v) and v > 0]
    units_candidates  = [v for v in candidates if _is_plausible_units(v) and v > 0]
    amount_candidates = [v for v in candidates if _is_plausible_amount(v) and v > 0]

    result = {
        "prev_reading":    None,
        "current_reading": None,
        "billed_units":    None,
        "amount_paid":     None,
        "raw_text_length": len(full_text),
    }

    # Try to populate fields conservatively
    if len(kwh_candidates) >= 2:
        sorted_kwh = sorted(kwh_candidates)
        result["prev_reading"]    = sorted_kwh[-2]
        result["current_reading"] = sorted_kwh[-1]
    if units_candidates:
        result["billed_units"] = min(units_candidates)
    if amount_candidates:
        result["amount_paid"] = max(amount_candidates)

    return result


# ---------------------------------------------------------------------------
# Self-test (import safety)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    tmpdir = tempfile.mkdtemp()
    try:
        db = os.path.join(tmpdir, "test_vault.db")
        init_db(db)

        # Insert
        insert_bill(
            consumer_id="HOME001",
            consumer_name="Rahul Sharma",
            billing_month="2024-01",
            prev_reading=1200.0,
            current_reading=1450.0,
            billed_units=250.0,
            amount_paid=1375.0,
            tariff_type="Domestic",
            household_profile="2 BHK / 1 AC",
            db_path=db,
        )

        bills = get_bills("HOME001", db_path=db)
        assert len(bills) == 1, "Expected 1 bill"
        assert bills[0]["billed_units"] == 250.0

        # Delete
        delete_bill(bills[0]["id"], db_path=db)
        assert get_bill_count(db_path=db) == 0

        print("[bill_vault] Self-test PASSED")
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
