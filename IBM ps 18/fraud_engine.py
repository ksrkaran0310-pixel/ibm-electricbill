"""
fraud_engine.py
===============
Stage-1 Anomaly Detection Engine — zero LLM dependency.

Rules:
  Rule 1 — Meter-to-Bill Mismatch
  Rule 2 — Cross-Month Continuity Breach (Bill Vault)
  Rule 3 — Peer-Normalized Sudden Drop (avoids seasonal false positives)
  Rule 4 — Flatline, Spike & Cold-Start Checks

ML Layer:
  IsolationForest + Z-Score -> risk_score (0-100), risk_tier, triggered_rules,
  estimated_revenue_impact (INR)

All inputs are validated; negative/nonsense values are rejected before scoring.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MISMATCH_TOLERANCE     = 0.02    # ±2% rounding tolerance for Rule 1
SUDDEN_DROP_THRESHOLD  = 0.55    # 55% drop vs rolling baseline — Rule 3
PEER_DROP_THRESHOLD    = 0.20    # peer zone must drop <20% for Rule 3 to fire
BASELINE_WINDOW        = 6       # months for rolling baseline
FLATLINE_CONSECUTIVE   = 2       # consecutive zero-kWh months — Rule 4
SPIKE_MULTIPLIER       = 2.50    # 250% spike threshold — Rule 4
DOMESTIC_BASELINE_KWH  = 250.0   # single-bill cold-start proxy (Domestic)
COMMERCIAL_BASELINE_KWH = 800.0
INDUSTRIAL_BASELINE_KWH = 4000.0

ZONE_BASELINES = {
    "Domestic":   DOMESTIC_BASELINE_KWH,
    "Commercial": COMMERCIAL_BASELINE_KWH,
    "Industrial": INDUSTRIAL_BASELINE_KWH,
}

TARIFF_RATES = {
    "Domestic":   5.5,
    "Commercial": 8.0,
    "Industrial": 6.5,
}

# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class FraudResult:
    consumer_id:               str
    billing_month:             Any          # int or str
    risk_score:                float        # 0–100
    risk_tier:                 str          # Critical / Suspicious / Normal
    triggered_rules:           list[str]    = field(default_factory=list)
    estimated_revenue_impact:  float        = 0.0   # INR
    anomaly_score_raw:         float        = 0.0   # raw IsolationForest score
    z_score:                   float        = 0.0
    details:                   dict         = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_record(record: dict) -> None:
    """Raise ValueError for nonsense / negative inputs."""
    required = ["consumer_id", "billed_units", "current_reading", "prev_reading"]
    for key in required:
        if key not in record:
            raise ValueError(f"Missing required field: {key}")
    if record["billed_units"] < 0:
        raise ValueError(f"billed_units cannot be negative: {record['billed_units']}")
    if record["current_reading"] < 0 or record["prev_reading"] < 0:
        raise ValueError("Meter readings cannot be negative.")
    if record["current_reading"] < record["prev_reading"]:
        raise ValueError(
            f"current_reading ({record['current_reading']}) < "
            f"prev_reading ({record['prev_reading']}) — meter rollback not supported."
        )


# ---------------------------------------------------------------------------
# Rule 1 — Meter-to-Bill Mismatch
# ---------------------------------------------------------------------------

def rule_meter_mismatch(record: dict) -> tuple[bool, float, str]:
    """
    Flag when billed_units deviates from (current_reading - prev_reading)
    beyond ±2% tolerance.

    Returns (flagged, gap_units, description).
    """
    metered   = record["current_reading"] - record["prev_reading"]
    billed    = record["billed_units"]
    if metered <= 0:
        return False, 0.0, ""
    gap       = abs(metered - billed)
    tolerance = MISMATCH_TOLERANCE * metered
    if gap > tolerance:
        desc = (
            f"Metered {metered:.2f} kWh vs billed {billed:.2f} kWh "
            f"(gap {gap:.2f} kWh, {gap/metered*100:.1f}%)"
        )
        return True, gap, desc
    return False, 0.0, ""


# ---------------------------------------------------------------------------
# Rule 2 — Cross-Month Continuity Breach (Bill Vault)
# ---------------------------------------------------------------------------

def rule_continuity_breach(bills: list[dict]) -> list[dict]:
    """
    Given a consumer's bills sorted by billing_month ascending, flag any
    month where prev_reading != prior month's current_reading.

    Returns a list of breach dicts {billing_month, expected_prev, actual_prev, gap}.
    """
    breaches = []
    for i in range(1, len(bills)):
        expected_prev = bills[i - 1]["current_reading"]
        actual_prev   = bills[i]["prev_reading"]
        gap           = abs(actual_prev - expected_prev)
        if gap > 0.1:   # 0.1 kWh floating-point tolerance
            breaches.append({
                "billing_month":  bills[i]["billing_month"],
                "expected_prev":  expected_prev,
                "actual_prev":    actual_prev,
                "gap":            round(gap, 2),
            })
    return breaches


# ---------------------------------------------------------------------------
# Rule 3 — Peer-Normalized Sudden Drop
# ---------------------------------------------------------------------------

def rule_peer_normalized_drop(
    consumer_history: pd.DataFrame,
    zone_df: pd.DataFrame,
    consumer_id: str,
    month: int,
) -> tuple[bool, float, float, str]:
    """
    Flag a >=55% drop vs the consumer's 6-month rolling baseline ONLY IF
    the zone/category peer average did NOT similarly drop (>=20% peer drop
    is treated as a natural seasonal event and suppressed).

    Parameters
    ----------
    consumer_history : DataFrame with columns [billing_month, billed_units]
                       for this consumer, sorted ascending.
    zone_df          : DataFrame with columns [consumer_id, billing_month,
                       billed_units] for all peers in the same zone.
    consumer_id      : ID of the consumer being evaluated.
    month            : billing_month being evaluated (int 1-12).

    Returns (flagged, consumer_drop_pct, peer_drop_pct, description).
    """
    ch = consumer_history.sort_values("billing_month")
    prior = ch[ch["billing_month"] < month].tail(BASELINE_WINDOW)

    if len(prior) < 3:
        return False, 0.0, 0.0, "Insufficient history for baseline"

    baseline     = prior["billed_units"].mean()
    current_row  = ch[ch["billing_month"] == month]
    if current_row.empty or baseline <= 0:
        return False, 0.0, 0.0, ""

    current_usage    = float(current_row["billed_units"].iloc[0])
    consumer_drop    = (baseline - current_usage) / baseline  # positive = drop

    if consumer_drop < SUDDEN_DROP_THRESHOLD:
        return False, consumer_drop, 0.0, ""

    # Compute peer zone average for the same month vs prior month
    peers_now   = zone_df[
        (zone_df["billing_month"] == month) &
        (zone_df["consumer_id"] != consumer_id)
    ]["billed_units"]
    peers_prior = zone_df[
        (zone_df["billing_month"] == month - 1) &
        (zone_df["consumer_id"] != consumer_id)
    ]["billed_units"] if month > 1 else pd.Series(dtype=float)

    if peers_now.empty or peers_prior.empty:
        # No peer data — flag conservatively
        peer_drop = 0.0
    else:
        peer_avg_now   = peers_now.mean()
        peer_avg_prior = peers_prior.mean()
        peer_drop      = (
            (peer_avg_prior - peer_avg_now) / peer_avg_prior
            if peer_avg_prior > 0 else 0.0
        )

    # Suppress if peer zone also dropped significantly (seasonal event)
    if peer_drop >= PEER_DROP_THRESHOLD:
        desc = (
            f"Drop {consumer_drop*100:.1f}% suppressed — zone peers also "
            f"dropped {peer_drop*100:.1f}% (seasonal event)"
        )
        return False, consumer_drop, peer_drop, desc

    desc = (
        f"Sudden drop {consumer_drop*100:.1f}% vs {BASELINE_WINDOW}-month "
        f"baseline ({baseline:.1f} kWh). Peer zone avg dropped only "
        f"{peer_drop*100:.1f}% — not seasonal."
    )
    return True, consumer_drop, peer_drop, desc


# ---------------------------------------------------------------------------
# Rule 4 — Flatline, Spike & Cold-Start
# ---------------------------------------------------------------------------

def rule_flatline_spike(
    consumer_history: pd.DataFrame,
    month: int,
    zone: str,
    household_profile: str = "",
) -> tuple[bool, str, str]:
    """
    Flag:
      - >= FLATLINE_CONSECUTIVE consecutive 0-kWh months ending at `month`
      - >= SPIKE_MULTIPLIER x rolling baseline spike
      - Cold-start: single-bill consumers — validate reading math and tariff
        against household profile baseline.

    Returns (flagged, rule_name, description).
    """
    ch = consumer_history.sort_values("billing_month")
    history_up_to = ch[ch["billing_month"] <= month]

    # Cold-start: only 1 bill
    if len(history_up_to) == 1:
        row     = history_up_to.iloc[0]
        metered = row["current_reading"] - row["prev_reading"]
        billed  = row["billed_units"]
        gap     = abs(metered - billed)
        tol     = MISMATCH_TOLERANCE * max(metered, 1)
        zone_base = ZONE_BASELINES.get(zone, DOMESTIC_BASELINE_KWH)

        issues = []
        if gap > tol:
            issues.append(
                f"Reading math error: metered {metered:.1f} vs billed {billed:.1f}"
            )
        if billed > zone_base * SPIKE_MULTIPLIER:
            issues.append(
                f"First bill {billed:.1f} kWh is {billed/zone_base:.1f}x zone baseline"
            )
        if issues:
            return True, "cold_start_anomaly", "; ".join(issues)
        return False, "", ""

    # Flatline check — last FLATLINE_CONSECUTIVE months
    recent = history_up_to.tail(FLATLINE_CONSECUTIVE)
    if len(recent) >= FLATLINE_CONSECUTIVE:
        if (recent["billed_units"] == 0).all():
            months_str = recent["billing_month"].tolist()
            return (
                True,
                "flatline",
                f"Zero usage for {FLATLINE_CONSECUTIVE} consecutive months: {months_str}",
            )

    # Spike check
    prior = history_up_to[history_up_to["billing_month"] < month].tail(BASELINE_WINDOW)
    if len(prior) >= 2:
        baseline = prior["billed_units"].mean()
        current_row = history_up_to[history_up_to["billing_month"] == month]
        if not current_row.empty and baseline > 0:
            current_usage = float(current_row["billed_units"].iloc[0])
            if current_usage >= baseline * SPIKE_MULTIPLIER:
                return (
                    True,
                    "spike",
                    f"Usage {current_usage:.1f} kWh = {current_usage/baseline:.1f}x "
                    f"baseline ({baseline:.1f} kWh)",
                )

    return False, "", ""


# ---------------------------------------------------------------------------
# ML Layer — IsolationForest + Z-Score
# ---------------------------------------------------------------------------

def _build_features(df: pd.DataFrame) -> np.ndarray:
    """Extract numeric features for IsolationForest."""
    features = df[["billed_units", "prev_reading", "current_reading"]].copy()
    features["metered"] = features["current_reading"] - features["prev_reading"]
    features["mismatch"] = (features["metered"] - features["billed_units"]).abs()
    return features[["billed_units", "metered", "mismatch"]].values


def train_isolation_forest(df: pd.DataFrame) -> tuple[IsolationForest, StandardScaler]:
    """Train IsolationForest on the provided DataFrame. Returns (model, scaler)."""
    X = _build_features(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = IsolationForest(
        n_estimators=200,
        contamination=0.08,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_scaled)
    return clf, scaler


def score_record(
    record: dict,
    clf: IsolationForest,
    scaler: StandardScaler,
    df: pd.DataFrame,
) -> tuple[float, float]:
    """
    Score a single record.
    Returns (risk_score 0-100, z_score).
    """
    features = np.array([[
        record["billed_units"],
        record["current_reading"] - record["prev_reading"],
        abs((record["current_reading"] - record["prev_reading"]) - record["billed_units"]),
    ]])
    X_scaled     = scaler.transform(features)
    raw_score    = clf.score_samples(X_scaled)[0]   # more negative = more anomalous

    # IsolationForest score_samples returns values roughly in [-0.5, 0.5]
    # Normalise to 0-100: invert so higher = more anomalous
    risk_score = float(np.clip((0.5 - raw_score) * 100, 0, 100))

    # Z-score of billed_units within the full dataset
    mu    = df["billed_units"].mean()
    sigma = df["billed_units"].std()
    z     = (record["billed_units"] - mu) / sigma if sigma > 0 else 0.0

    return round(risk_score, 2), round(float(z), 3)


# ---------------------------------------------------------------------------
# Risk tier mapping
# ---------------------------------------------------------------------------

def _tier(risk_score: float, rule_count: int) -> str:
    if risk_score >= 70 or rule_count >= 2:
        return "Critical"
    if risk_score >= 40 or rule_count >= 1:
        return "Suspicious"
    return "Normal"


# ---------------------------------------------------------------------------
# Revenue impact estimate
# ---------------------------------------------------------------------------

def _revenue_impact(record: dict, triggered_rules: list[str]) -> float:
    """
    Estimate lost INR revenue based on triggered rules.
    Uses tariff rate × estimated unrecorded units.
    """
    rate  = TARIFF_RATES.get(record.get("tariff_type", "Domestic"), 5.5)
    total = 0.0

    metered  = record["current_reading"] - record["prev_reading"]
    billed   = record["billed_units"]

    if "meter_to_bill_mismatch" in triggered_rules:
        gap   = abs(metered - billed)
        total += gap * rate

    if "peer_normalized_drop" in triggered_rules:
        # Conservative: treat 50% of the drop as unrecorded units
        baseline_est = metered / 0.40 if metered > 0 else 0
        total += (baseline_est - metered) * 0.5 * rate

    if "flatline" in triggered_rules:
        # Estimate based on zone baseline
        zone_base = ZONE_BASELINES.get(record.get("tariff_type", "Domestic"), 250.0)
        total += zone_base * 2 * rate   # 2 flatline months

    if "spike" in triggered_rules:
        # Spike may indicate unauthorized sub-metering; flag half the excess
        baseline_est = billed / SPIKE_MULTIPLIER
        total += (billed - baseline_est) * 0.5 * rate

    return round(total, 2)


# ---------------------------------------------------------------------------
# Single-record analysis
# ---------------------------------------------------------------------------

def analyze_record(
    record: dict,
    consumer_history: pd.DataFrame,
    zone_df: pd.DataFrame,
    clf: IsolationForest,
    scaler: StandardScaler,
    full_df: pd.DataFrame,
    bills_vault: list[dict] | None = None,
) -> FraudResult:
    """
    Run all 4 rules + ML scoring on a single billing record.

    Parameters
    ----------
    record           : dict with keys: consumer_id, billing_month, prev_reading,
                       current_reading, billed_units, tariff_type, zone
    consumer_history : DataFrame of this consumer's historical rows
    zone_df          : DataFrame of all records in the same zone
    clf / scaler     : trained IsolationForest model
    full_df          : entire dataset (for Z-score computation)
    bills_vault      : optional list of Bill Vault dicts for Rule 2
    """
    _validate_record(record)

    triggered_rules = []
    details         = {}

    # Rule 1 — Meter-to-Bill Mismatch
    r1_flag, r1_gap, r1_desc = rule_meter_mismatch(record)
    if r1_flag:
        triggered_rules.append("meter_to_bill_mismatch")
        details["rule1"] = r1_desc

    # Rule 2 — Cross-Month Continuity (Bill Vault)
    if bills_vault and len(bills_vault) >= 2:
        breaches = rule_continuity_breach(bills_vault)
        if breaches:
            triggered_rules.append("cross_month_continuity_breach")
            details["rule2"] = breaches

    # Rule 3 — Peer-Normalized Sudden Drop
    r3_flag, r3_drop, r3_peer, r3_desc = rule_peer_normalized_drop(
        consumer_history, zone_df, record["consumer_id"], record["billing_month"]
    )
    if r3_flag:
        triggered_rules.append("peer_normalized_drop")
        details["rule3"] = r3_desc

    # Rule 4 — Flatline / Spike / Cold-Start
    r4_flag, r4_name, r4_desc = rule_flatline_spike(
        consumer_history,
        record["billing_month"],
        record.get("zone", record.get("tariff_type", "Domestic")),
        record.get("household_profile", ""),
    )
    if r4_flag:
        triggered_rules.append(r4_name)
        details["rule4"] = r4_desc

    # ML scoring
    risk_score, z_score = score_record(record, clf, scaler, full_df)
    raw_score           = clf.score_samples(
        scaler.transform([[
            record["billed_units"],
            record["current_reading"] - record["prev_reading"],
            abs((record["current_reading"] - record["prev_reading"]) - record["billed_units"]),
        ]])
    )[0]

    # Boost risk_score by rule hits
    boost = len(triggered_rules) * 12
    risk_score = min(100.0, risk_score + boost)

    tier   = _tier(risk_score, len(triggered_rules))
    impact = _revenue_impact(record, triggered_rules)

    return FraudResult(
        consumer_id=record["consumer_id"],
        billing_month=record["billing_month"],
        risk_score=risk_score,
        risk_tier=tier,
        triggered_rules=triggered_rules,
        estimated_revenue_impact=impact,
        anomaly_score_raw=round(float(raw_score), 4),
        z_score=z_score,
        details=details,
    )


# ---------------------------------------------------------------------------
# Bulk analysis — entire DataFrame
# ---------------------------------------------------------------------------

def analyze_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full pipeline on a billing CSV DataFrame.

    Returns the original DataFrame with added columns:
        risk_score, risk_tier, triggered_rules, estimated_revenue_impact
    """
    required_cols = {
        "consumer_id", "billing_month", "prev_reading",
        "current_reading", "billed_units", "zone",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    if (df["billed_units"] < 0).any():
        raise ValueError("DataFrame contains negative billed_units values.")

    # Train IsolationForest once on the full dataset
    clf, scaler = train_isolation_forest(df)

    results = []
    for consumer_id, group in df.groupby("consumer_id"):
        group = group.sort_values("billing_month").reset_index(drop=True)
        zone  = group["zone"].iloc[0]
        zone_df = df[df["zone"] == zone].copy()

        for _, row in group.iterrows():
            record = row.to_dict()
            result = analyze_record(
                record=record,
                consumer_history=group,
                zone_df=zone_df,
                clf=clf,
                scaler=scaler,
                full_df=df,
            )
            results.append({
                "consumer_id":              result.consumer_id,
                "billing_month":            result.billing_month,
                "risk_score":               result.risk_score,
                "risk_tier":                result.risk_tier,
                "triggered_rules":          "|".join(result.triggered_rules),
                "estimated_revenue_impact": result.estimated_revenue_impact,
                "z_score":                  result.z_score,
            })

    results_df = pd.DataFrame(results)
    merged     = df.merge(results_df, on=["consumer_id", "billing_month"], how="left")
    return merged


# ---------------------------------------------------------------------------
# CLI self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    CSV_PATH = os.path.join("data", "utility_billing_records.csv")
    if not os.path.exists(CSV_PATH):
        print("[fraud_engine] CSV not found — run data_generator.py first.")
    else:
        df = pd.read_csv(CSV_PATH)
        print(f"[fraud_engine] Analysing {len(df)} records ...")
        out = analyze_dataframe(df)
        flagged = out[out["risk_tier"] != "Normal"]
        print(f"[fraud_engine] Flagged records: {len(flagged)}")
        tier_counts = out["risk_tier"].value_counts()
        print(f"[fraud_engine] Tier distribution:\n{tier_counts.to_string()}")
        print("[fraud_engine] Done.")
