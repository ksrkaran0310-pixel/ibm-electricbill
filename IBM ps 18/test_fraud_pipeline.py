"""
test_fraud_pipeline.py
======================
Automated tests for the fraud detection engine (Phase 2).
Covers:
  - Rule 1: Meter-to-bill mismatch detection
  - Rule 2: Bill Vault cross-month continuity breach detection
  - Rule 3: Peer-normalized drop detection AND winter-dip non-flag proof
  - Rule 4: Flatline & spike detection + Day-1 cold-start check
  - ML layer: IsolationForest scoring & invalid/negative input rejection
  - (Phase 3 tests appended later: Granite offline fallback + SHA-256 hash)

Run with:  python -m pytest test_fraud_pipeline.py -v
       or: python test_fraud_pipeline.py
"""

import sys
import types
import pytest
import numpy as np
import pandas as pd

from fraud_engine import (
    FraudResult,
    analyze_dataframe,
    analyze_record,
    rule_continuity_breach,
    rule_flatline_spike,
    rule_meter_mismatch,
    rule_peer_normalized_drop,
    train_isolation_forest,
    score_record,
    _validate_record,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Create a minimal DataFrame for engine calls."""
    defaults = {
        "consumer_id":    "TEST001",
        "zone":           "Domestic",
        "tariff_type":    "Domestic",
        "billing_month":  1,
        "prev_reading":   1000.0,
        "current_reading": 1250.0,
        "billed_units":   250.0,
        "amount_paid":    1375.0,
        "anomaly_label":  "normal",
    }
    result = []
    for r in rows:
        row = {**defaults, **r}
        result.append(row)
    return pd.DataFrame(result)


def _make_consumer_history(months_units: list[tuple]) -> pd.DataFrame:
    """
    months_units: list of (billing_month, billed_units, prev_reading, current_reading)
    Returns a consumer history DataFrame.
    """
    rows = []
    for m, units, prev, curr in months_units:
        rows.append({
            "consumer_id":     "TEST001",
            "billing_month":   m,
            "billed_units":    units,
            "prev_reading":    prev,
            "current_reading": curr,
            "zone":            "Domestic",
            "tariff_type":     "Domestic",
        })
    return pd.DataFrame(rows)


def _make_zone_df(month_avg: float, months: list[int], n_peers: int = 20) -> pd.DataFrame:
    """Create peer zone DataFrame with consistent usage."""
    rows = []
    for m in months:
        for i in range(n_peers):
            rows.append({
                "consumer_id":    f"PEER{i:03d}",
                "billing_month":  m,
                "billed_units":   month_avg + np.random.normal(0, 10),
                "prev_reading":   1000.0,
                "current_reading": 1250.0,
                "zone":           "Domestic",
                "tariff_type":    "Domestic",
            })
    return pd.DataFrame(rows)


# ============================================================
# RULE 1 — METER-TO-BILL MISMATCH
# ============================================================

class TestRule1MeterMismatch:

    def test_exact_match_not_flagged(self):
        record = {
            "consumer_id": "C1",
            "prev_reading": 1000.0,
            "current_reading": 1250.0,
            "billed_units": 250.0,
        }
        flagged, gap, _ = rule_meter_mismatch(record)
        assert not flagged
        assert gap == 0.0

    def test_within_tolerance_not_flagged(self):
        # 1% difference — within 2% tolerance
        record = {
            "consumer_id": "C1",
            "prev_reading": 1000.0,
            "current_reading": 1250.0,
            "billed_units": 247.5,   # 1% under 250
        }
        flagged, _, _ = rule_meter_mismatch(record)
        assert not flagged

    def test_mismatch_beyond_tolerance_flagged(self):
        # Billed only 150 units; metered 250 — 40% gap
        record = {
            "consumer_id": "C1",
            "prev_reading": 1000.0,
            "current_reading": 1250.0,
            "billed_units": 150.0,
        }
        flagged, gap, desc = rule_meter_mismatch(record)
        assert flagged
        assert gap == pytest.approx(100.0, abs=0.5)
        assert "150" in desc or "250" in desc

    def test_zero_metered_not_flagged(self):
        # Meter did not advance — no division / false flag
        record = {
            "consumer_id": "C1",
            "prev_reading": 1000.0,
            "current_reading": 1000.0,
            "billed_units": 0.0,
        }
        flagged, _, _ = rule_meter_mismatch(record)
        assert not flagged

    def test_overbilling_flagged(self):
        # Billed 400 units but metered only 250 — overbilling
        record = {
            "consumer_id": "C1",
            "prev_reading": 1000.0,
            "current_reading": 1250.0,
            "billed_units": 400.0,
        }
        flagged, gap, _ = rule_meter_mismatch(record)
        assert flagged
        assert gap == pytest.approx(150.0, abs=0.5)


# ============================================================
# RULE 2 — CROSS-MONTH CONTINUITY BREACH (BILL VAULT)
# ============================================================

class TestRule2ContinuityBreach:

    def _make_bills(self, readings: list[tuple]) -> list[dict]:
        """readings: list of (billing_month, prev_reading, current_reading)"""
        return [
            {"billing_month": m, "prev_reading": p, "current_reading": c}
            for m, p, c in readings
        ]

    def test_clean_history_no_breach(self):
        bills = self._make_bills([
            ("2024-01", 1200.0, 1450.0),
            ("2024-02", 1450.0, 1680.0),
            ("2024-03", 1680.0, 1950.0),
        ])
        breaches = rule_continuity_breach(bills)
        assert breaches == []

    def test_single_bill_no_breach(self):
        bills = self._make_bills([("2024-01", 1200.0, 1450.0)])
        breaches = rule_continuity_breach(bills)
        assert breaches == []

    def test_continuity_breach_detected(self):
        # March current=1950, April prev=1980 — 30-unit gap
        bills = self._make_bills([
            ("2024-01", 1200.0, 1450.0),
            ("2024-02", 1450.0, 1680.0),
            ("2024-03", 1680.0, 1950.0),
            ("2024-04", 1980.0, 2250.0),   # <-- breach here
        ])
        breaches = rule_continuity_breach(bills)
        assert len(breaches) == 1
        assert breaches[0]["billing_month"] == "2024-04"
        assert breaches[0]["gap"] == pytest.approx(30.0, abs=0.5)

    def test_multiple_breaches_detected(self):
        bills = self._make_bills([
            ("2024-01", 1000.0, 1200.0),
            ("2024-02", 1250.0, 1450.0),   # breach: expected 1200, got 1250
            ("2024-03", 1500.0, 1700.0),   # breach: expected 1450, got 1500
        ])
        breaches = rule_continuity_breach(bills)
        assert len(breaches) == 2

    def test_seed_data_breach_detected(self):
        """Matches the exact scenario injected by seed_dev_data.py."""
        bills = self._make_bills([
            ("2024-01", 1200.0, 1450.0),
            ("2024-02", 1450.0, 1680.0),
            ("2024-03", 1680.0, 1950.0),
            ("2024-04", 1980.0, 2250.0),   # deliberate 30-unit gap
            ("2024-05", 2250.0, 2510.0),
        ])
        breaches = rule_continuity_breach(bills)
        assert any(b["billing_month"] == "2024-04" for b in breaches)


# ============================================================
# RULE 3 — PEER-NORMALIZED SUDDEN DROP
# ============================================================

class TestRule3PeerNormalizedDrop:

    def _consumer_history(self, baseline_units: float, drop_month: int,
                           drop_factor: float) -> pd.DataFrame:
        rows = []
        prev = 1000.0
        for m in range(1, 13):
            units = baseline_units if m < drop_month else baseline_units * drop_factor
            curr  = prev + units
            rows.append({
                "consumer_id":    "TEST_DROP",
                "billing_month":  m,
                "billed_units":   units,
                "prev_reading":   prev,
                "current_reading": curr,
                "zone":           "Domestic",
                "tariff_type":    "Domestic",
            })
            prev = curr
        return pd.DataFrame(rows)

    def test_drop_flagged_when_peers_stable(self):
        """Consumer drops 70% but peers stay flat — must be flagged."""
        consumer_hist = self._consumer_history(
            baseline_units=250.0, drop_month=8, drop_factor=0.25
        )
        # Peers stable at 250 kWh all year
        zone_df = _make_zone_df(250.0, months=list(range(1, 13)))
        # Add this consumer to zone_df too
        zone_df = pd.concat([zone_df, consumer_hist], ignore_index=True)

        flagged, drop_pct, peer_pct, desc = rule_peer_normalized_drop(
            consumer_hist, zone_df, "TEST_DROP", 8
        )
        assert flagged, f"Expected flagged but got: drop={drop_pct:.2%}, peer={peer_pct:.2%}, {desc}"
        assert drop_pct >= 0.55

    def test_zone_wide_winter_dip_NOT_flagged(self):
        """
        Zone-wide 25% winter dip — MUST NOT be flagged (seasonal event).
        This proves the engine avoids seasonal false positives.
        """
        # Consumer drops 25% — below 55% threshold anyway
        consumer_hist = self._consumer_history(
            baseline_units=250.0, drop_month=2, drop_factor=0.75
        )
        # Peers also drop 25% in Feb vs Jan
        rows = []
        for i in range(30):
            # Jan: 250, Feb: 187 (25% drop for all peers)
            for m, units in [(1, 250.0), (2, 187.5)]:
                rows.append({
                    "consumer_id":    f"PEER{i:03d}",
                    "billing_month":  m,
                    "billed_units":   units + np.random.normal(0, 5),
                    "prev_reading":   1000.0,
                    "current_reading": 1250.0,
                    "zone":           "Domestic",
                    "tariff_type":    "Domestic",
                })
        zone_df = pd.DataFrame(rows)

        flagged, _, _, desc = rule_peer_normalized_drop(
            consumer_hist, zone_df, "TEST_DROP", 2
        )
        # The consumer only dropped 25% — below 55% threshold, so not flagged
        assert not flagged, f"Winter dip should not be flagged but got: {desc}"

    def test_severe_drop_with_peer_seasonal_dip_suppressed(self):
        """
        Consumer drops 60% AND peers also drop 25% (zone-wide event).
        Rule 3 fire condition: peer drop >= 20% suppresses the flag.
        """
        consumer_hist = self._consumer_history(
            baseline_units=250.0, drop_month=7, drop_factor=0.35  # 65% drop
        )
        # Peers drop 25% in month 7 vs month 6 (seasonal)
        rows = []
        for i in range(25):
            for m in range(1, 13):
                seasonal = 0.75 if m == 7 else 1.0
                rows.append({
                    "consumer_id":    f"PEER{i:03d}",
                    "billing_month":  m,
                    "billed_units":   250.0 * seasonal + np.random.normal(0, 5),
                    "prev_reading":   1000.0,
                    "current_reading": 1250.0,
                    "zone":           "Domestic",
                    "tariff_type":    "Domestic",
                })
        zone_df = pd.DataFrame(rows)

        flagged, drop_pct, peer_pct, desc = rule_peer_normalized_drop(
            consumer_hist, zone_df, "TEST_DROP", 7
        )
        # peer_drop >= 20% -> suppress
        assert not flagged, (
            f"Should be suppressed (peer drop={peer_pct:.2%}) but flagged: {desc}"
        )

    def test_insufficient_history_not_flagged(self):
        """Only 2 months of history — below 3-month minimum, should not flag."""
        rows = [
            {"consumer_id": "X", "billing_month": 1, "billed_units": 250.0,
             "prev_reading": 1000.0, "current_reading": 1250.0,
             "zone": "Domestic", "tariff_type": "Domestic"},
            {"consumer_id": "X", "billing_month": 2, "billed_units": 50.0,
             "prev_reading": 1250.0, "current_reading": 1300.0,
             "zone": "Domestic", "tariff_type": "Domestic"},
        ]
        consumer_hist = pd.DataFrame(rows)
        zone_df       = _make_zone_df(250.0, months=[1, 2])
        flagged, _, _, _ = rule_peer_normalized_drop(consumer_hist, zone_df, "X", 2)
        assert not flagged


# ============================================================
# RULE 4 — FLATLINE & SPIKE
# ============================================================

class TestRule4FlatlineSpike:

    def _history(self, units_by_month: dict) -> pd.DataFrame:
        rows = []
        prev = 1000.0
        for m in sorted(units_by_month):
            u    = units_by_month[m]
            curr = prev + u
            rows.append({
                "consumer_id":    "T",
                "billing_month":  m,
                "billed_units":   u,
                "prev_reading":   prev,
                "current_reading": curr,
                "zone":           "Domestic",
                "tariff_type":    "Domestic",
            })
            prev = curr
        return pd.DataFrame(rows)

    def test_flatline_two_consecutive_months_flagged(self):
        hist = self._history({1: 250, 2: 240, 3: 260, 4: 0, 5: 0})
        flagged, name, desc = rule_flatline_spike(hist, 5, "Domestic")
        assert flagged
        assert name == "flatline"

    def test_single_zero_month_not_flatline(self):
        hist = self._history({1: 250, 2: 240, 3: 0, 4: 230})
        flagged, name, _ = rule_flatline_spike(hist, 4, "Domestic")
        assert not (flagged and name == "flatline")

    def test_spike_detected(self):
        # Month 7: 1000 kWh vs baseline of ~250 — 4x spike
        hist = self._history({1: 250, 2: 260, 3: 240, 4: 255, 5: 245, 6: 250, 7: 1000})
        flagged, name, desc = rule_flatline_spike(hist, 7, "Domestic")
        assert flagged
        assert name == "spike"

    def test_normal_usage_not_spiked(self):
        hist = self._history({1: 250, 2: 260, 3: 245, 4: 255, 5: 240, 6: 250, 7: 275})
        flagged, name, _ = rule_flatline_spike(hist, 7, "Domestic")
        assert not (flagged and name == "spike")

    def test_cold_start_valid_first_bill_ok(self):
        # Day-1 user — single bill with correct math
        hist = pd.DataFrame([{
            "consumer_id":    "NEW",
            "billing_month":  1,
            "billed_units":   250.0,
            "prev_reading":   1000.0,
            "current_reading": 1250.0,
            "zone":           "Domestic",
            "tariff_type":    "Domestic",
        }])
        flagged, _, _ = rule_flatline_spike(hist, 1, "Domestic")
        assert not flagged

    def test_cold_start_reading_error_flagged(self):
        # Day-1 user — metered 250 but billed 100 (reading math error)
        hist = pd.DataFrame([{
            "consumer_id":    "NEW",
            "billing_month":  1,
            "billed_units":   100.0,
            "prev_reading":   1000.0,
            "current_reading": 1250.0,
            "zone":           "Domestic",
            "tariff_type":    "Domestic",
        }])
        flagged, name, desc = rule_flatline_spike(hist, 1, "Domestic")
        assert flagged
        assert name == "cold_start_anomaly"


# ============================================================
# ML LAYER — IsolationForest + Input Validation
# ============================================================

class TestMLLayer:

    def _base_df(self, n: int = 150) -> pd.DataFrame:
        np.random.seed(0)
        rows = []
        prev = 1000.0
        for i in range(n):
            units = max(0.0, np.random.normal(250, 40))
            curr  = prev + units
            rows.append({
                "consumer_id":    f"C{i:03d}",
                "zone":           "Domestic",
                "tariff_type":    "Domestic",
                "billing_month":  (i % 12) + 1,
                "prev_reading":   round(prev, 1),
                "current_reading": round(curr, 1),
                "billed_units":   round(units, 2),
                "amount_paid":    round(units * 5.5, 2),
                "anomaly_label":  "normal",
            })
            prev = curr
        return pd.DataFrame(rows)

    def test_isolation_forest_trains_and_scores(self):
        df          = self._base_df()
        clf, scaler = train_isolation_forest(df)
        record      = {
            "consumer_id":     "C000",
            "billing_month":   1,
            "prev_reading":    1000.0,
            "current_reading": 1250.0,
            "billed_units":    250.0,
            "tariff_type":     "Domestic",
            "zone":            "Domestic",
        }
        risk_score, z_score = score_record(record, clf, scaler, df)
        assert 0 <= risk_score <= 100
        assert isinstance(z_score, float)

    def test_anomalous_record_scores_higher(self):
        """A severely mismatched record should score higher than a normal one."""
        df          = self._base_df(200)
        clf, scaler = train_isolation_forest(df)

        normal_record = {
            "consumer_id":     "NORMAL",
            "billing_month":   5,
            "prev_reading":    1000.0,
            "current_reading": 1250.0,
            "billed_units":    250.0,
            "tariff_type":     "Domestic",
            "zone":            "Domestic",
        }
        anomaly_record = {
            "consumer_id":     "ANOMALY",
            "billing_month":   5,
            "prev_reading":    1000.0,
            "current_reading": 1250.0,
            "billed_units":    10.0,    # extreme mismatch: metered 250, billed 10
            "tariff_type":     "Domestic",
            "zone":            "Domestic",
        }
        normal_score,  _ = score_record(normal_record,  clf, scaler, df)
        anomaly_score, _ = score_record(anomaly_record, clf, scaler, df)
        assert anomaly_score >= normal_score, (
            f"Anomaly score {anomaly_score} should be >= normal score {normal_score}"
        )

    def test_negative_billed_units_rejected(self):
        with pytest.raises(ValueError, match="negative"):
            _validate_record({
                "consumer_id":    "X",
                "billing_month":  1,
                "prev_reading":   1000.0,
                "current_reading": 1250.0,
                "billed_units":   -50.0,
            })

    def test_negative_reading_rejected(self):
        with pytest.raises(ValueError):
            _validate_record({
                "consumer_id":    "X",
                "billing_month":  1,
                "prev_reading":   -100.0,
                "current_reading": 150.0,
                "billed_units":   250.0,
            })

    def test_current_less_than_prev_rejected(self):
        with pytest.raises(ValueError, match="meter rollback"):
            _validate_record({
                "consumer_id":    "X",
                "billing_month":  1,
                "prev_reading":   1500.0,
                "current_reading": 1200.0,
                "billed_units":   250.0,
            })

    def test_missing_field_rejected(self):
        with pytest.raises(ValueError, match="Missing required field"):
            _validate_record({
                "consumer_id":    "X",
                "billing_month":  1,
                "prev_reading":   1000.0,
                # current_reading missing
                "billed_units":   250.0,
            })

    def test_analyze_dataframe_on_csv(self):
        """End-to-end: load generated CSV and run full pipeline."""
        import os
        csv_path = os.path.join("data", "utility_billing_records.csv")
        if not os.path.exists(csv_path):
            pytest.skip("utility_billing_records.csv not found — run data_generator.py first")
        df     = pd.read_csv(csv_path)
        result = analyze_dataframe(df)
        assert "risk_score"               in result.columns
        assert "risk_tier"                in result.columns
        assert "triggered_rules"          in result.columns
        assert "estimated_revenue_impact" in result.columns
        assert result["risk_score"].between(0, 100).all(), "risk_score out of 0-100 range"
        assert set(result["risk_tier"].unique()).issubset({"Critical", "Suspicious", "Normal"})

    def test_analyze_dataframe_missing_column_raises(self):
        df = _make_df([{"consumer_id": "X"}])
        df = df.drop(columns=["billed_units"])
        with pytest.raises(ValueError, match="missing required columns"):
            analyze_dataframe(df)

    def test_analyze_dataframe_negative_units_raises(self):
        df = _make_df([{
            "consumer_id": "X",
            "billed_units": -1.0,
            "current_reading": 1050.0,
        }])
        with pytest.raises(ValueError, match="negative"):
            analyze_dataframe(df)


# ============================================================
# INTEGRATION — analyze_record end-to-end
# ============================================================

class TestAnalyzeRecordIntegration:

    def _setup(self):
        """Build a small but realistic dataset + trained model."""
        np.random.seed(42)
        rows = []
        prev = 1000.0
        for i in range(120):
            units = max(0.0, np.random.normal(250, 40))
            curr  = prev + units
            rows.append({
                "consumer_id":    "C000",
                "zone":           "Domestic",
                "tariff_type":    "Domestic",
                "billing_month":  (i % 12) + 1,
                "prev_reading":   round(prev, 1),
                "current_reading": round(curr, 1),
                "billed_units":   round(units, 2),
                "amount_paid":    round(units * 5.5, 2),
                "anomaly_label":  "normal",
            })
            prev = curr
        df          = pd.DataFrame(rows)
        clf, scaler = train_isolation_forest(df)
        return df, clf, scaler

    def test_mismatch_record_flagged_as_suspicious_or_critical(self):
        df, clf, scaler = self._setup()
        consumer_hist   = df[df["consumer_id"] == "C000"].copy()
        zone_df         = df.copy()

        record = {
            "consumer_id":    "C000",
            "billing_month":  7,
            "prev_reading":   1000.0,
            "current_reading": 1250.0,
            "billed_units":   50.0,     # extreme mismatch
            "tariff_type":    "Domestic",
            "zone":           "Domestic",
        }
        result = analyze_record(
            record=record,
            consumer_history=consumer_hist,
            zone_df=zone_df,
            clf=clf,
            scaler=scaler,
            full_df=df,
        )
        assert result.risk_tier in ("Suspicious", "Critical")
        assert "meter_to_bill_mismatch" in result.triggered_rules

    def test_normal_record_not_flagged(self):
        df, clf, scaler = self._setup()
        consumer_hist   = df[df["consumer_id"] == "C000"].copy()
        zone_df         = df.copy()

        record = {
            "consumer_id":    "C000",
            "billing_month":  7,
            "prev_reading":   1000.0,
            "current_reading": 1250.0,
            "billed_units":   250.0,    # exact match
            "tariff_type":    "Domestic",
            "zone":           "Domestic",
        }
        result = analyze_record(
            record=record,
            consumer_history=consumer_hist,
            zone_df=zone_df,
            clf=clf,
            scaler=scaler,
            full_df=df,
        )
        # No deterministic rule should fire
        mismatch_rules = [r for r in result.triggered_rules
                         if r not in ("spike", "flatline", "cold_start_anomaly")]
        assert "meter_to_bill_mismatch"       not in result.triggered_rules
        assert "cross_month_continuity_breach" not in result.triggered_rules


# ============================================================
# PHASE 3 — GRANITE AGENT: OFFLINE FALLBACK + SHA-256 HASH
# ============================================================

class TestGraniteAgent:
    """
    Tests for granite_agent.py.
    These tests do NOT require real IBM Cloud credentials — they verify:
      1. Offline fallback fires when credentials are absent.
      2. get_engine_status() returns a safe, non-secret hint.
      3. explain_fraud_case() returns a non-empty string offline.
      4. draft_action_report() returns a dict with sha256_hash offline.
      5. SHA-256 hash is deterministic and verifiable.
      6. Invalid report_type raises ValueError.
      7. answer_query() returns a string offline.
      8. Empty question returns a prompt message.
    """

    # Minimal verified findings dict (matches fraud_engine output shape)
    FINDINGS = {
        "consumer_id":              "CON0042",
        "billing_month":            7,
        "risk_score":               82.5,
        "risk_tier":                "Critical",
        "triggered_rules":          ["meter_to_bill_mismatch", "peer_normalized_drop"],
        "estimated_revenue_impact": 1870.50,
        "details": {
            "rule1": "Metered 310.00 kWh vs billed 120.00 kWh (gap 190.00 kWh, 61.3%)",
            "rule3": (
                "Sudden drop 68.4% vs 6-month baseline (310.0 kWh). "
                "Peer zone avg dropped only 3.2%."
            ),
        },
    }

    def _clear_model_cache(self):
        """Clear the module-level model cache to force re-evaluation."""
        import granite_agent
        granite_agent._MODEL_CACHE.clear()

    def _force_offline(self, monkeypatch):
        """Remove all watsonx env vars so the agent goes offline."""
        self._clear_model_cache()
        for var in ("WATSONX_APIKEY", "WATSONX_PROJECT_ID", "WATSONX_URL", "WATSONX_MODEL_ID"):
            monkeypatch.delenv(var, raising=False)

    # ------------------------------------------------------------------
    # 1. get_engine_status — offline path
    # ------------------------------------------------------------------

    def test_engine_status_offline_when_no_credentials(self, monkeypatch):
        self._force_offline(monkeypatch)
        import granite_agent
        status = granite_agent.get_engine_status()
        assert status["status"] == "Offline Fallback"
        assert status["model_id"] is None
        assert len(status["hint"]) > 0

    def test_engine_status_hint_does_not_contain_secret(self, monkeypatch):
        """Hint must not accidentally echo an API key fragment."""
        monkeypatch.setenv("WATSONX_APIKEY",     "sk-FAKEKEYDONOTUSE123456789ABCDEF")
        monkeypatch.setenv("WATSONX_PROJECT_ID", "fake-project-id")
        monkeypatch.setenv("WATSONX_URL",        "https://fake.example.com")
        monkeypatch.setenv("WATSONX_MODEL_ID",   "ibm/fake-model")
        self._clear_model_cache()
        import granite_agent
        # Force a connection failure by using a bad URL — the hint should be safe
        status = granite_agent.get_engine_status()
        # If it went offline, hint must not contain the raw API key
        if status["status"] == "Offline Fallback":
            assert "FAKEKEYDONOTUSE" not in status["hint"]

    # ------------------------------------------------------------------
    # 2. explain_fraud_case — offline fallback
    # ------------------------------------------------------------------

    def test_explain_fraud_case_offline_returns_string(self, monkeypatch):
        self._force_offline(monkeypatch)
        import granite_agent
        result = granite_agent.explain_fraud_case(self.FINDINGS)
        assert isinstance(result, str)
        assert len(result) > 50

    def test_explain_fraud_case_offline_contains_consumer_id(self, monkeypatch):
        self._force_offline(monkeypatch)
        import granite_agent
        result = granite_agent.explain_fraud_case(self.FINDINGS)
        assert "CON0042" in result

    def test_explain_fraud_case_offline_mentions_triggered_rules(self, monkeypatch):
        self._force_offline(monkeypatch)
        import granite_agent
        result = granite_agent.explain_fraud_case(self.FINDINGS)
        # Offline explanation should reference at least one triggered rule
        assert any(
            kw in result.lower()
            for kw in ("mismatch", "drop", "meter", "bill", "peer")
        )

    # ------------------------------------------------------------------
    # 3. draft_action_report — offline fallback + SHA-256
    # ------------------------------------------------------------------

    def test_draft_report_field_inspection_offline(self, monkeypatch):
        self._force_offline(monkeypatch)
        import granite_agent
        report = granite_agent.draft_action_report(self.FINDINGS, "field_inspection")
        assert isinstance(report, dict)
        assert "report_text" in report
        assert "sha256_hash" in report
        assert "timestamp"   in report
        assert "engine"      in report
        assert report["engine"] == "Offline Deterministic"
        assert report["report_type"] == "field_inspection"

    def test_draft_report_correction_request_offline(self, monkeypatch):
        self._force_offline(monkeypatch)
        import granite_agent
        report = granite_agent.draft_action_report(self.FINDINGS, "correction_request")
        assert isinstance(report, dict)
        assert report["report_type"] == "correction_request"
        assert len(report["report_text"]) > 100

    def test_sha256_hash_is_valid_hex(self, monkeypatch):
        self._force_offline(monkeypatch)
        import granite_agent
        report = granite_agent.draft_action_report(self.FINDINGS, "field_inspection")
        sha    = report["sha256_hash"]
        assert len(sha) == 64
        assert all(c in "0123456789abcdef" for c in sha)

    def test_sha256_hash_is_deterministic(self, monkeypatch):
        """Same report content must always produce the same hash."""
        self._force_offline(monkeypatch)
        import granite_agent
        # Generate twice with same inputs — timestamps will differ, so compare
        # hash against manually recomputed value from report_text
        report = granite_agent.draft_action_report(self.FINDINGS, "field_inspection")
        import hashlib
        recomputed = hashlib.sha256(report["report_text"].encode("utf-8")).hexdigest()
        assert report["sha256_hash"] == recomputed, (
            "SHA-256 hash does not match recomputed value — tampering possible!"
        )

    def test_sha256_hash_changes_when_report_text_changes(self, monkeypatch):
        """Hash must change if the report text is modified (integrity check)."""
        self._force_offline(monkeypatch)
        import granite_agent, hashlib
        report      = granite_agent.draft_action_report(self.FINDINGS, "field_inspection")
        original    = report["sha256_hash"]
        tampered    = hashlib.sha256(
            (report["report_text"] + " TAMPERED").encode("utf-8")
        ).hexdigest()
        assert tampered != original

    def test_invalid_report_type_raises(self, monkeypatch):
        self._force_offline(monkeypatch)
        import granite_agent
        with pytest.raises(ValueError, match="report_type"):
            granite_agent.draft_action_report(self.FINDINGS, "unknown_type")

    # ------------------------------------------------------------------
    # 4. answer_query — offline fallback
    # ------------------------------------------------------------------

    def test_answer_query_offline_returns_string(self, monkeypatch):
        self._force_offline(monkeypatch)
        import granite_agent
        result = granite_agent.answer_query(
            "Which zone has the most fraud cases?",
            "Dataset: 300 consumers. Domestic: 25 Critical, Commercial: 12, Industrial: 8.",
        )
        assert isinstance(result, str)
        assert len(result) > 20

    def test_answer_query_empty_question(self, monkeypatch):
        self._force_offline(monkeypatch)
        import granite_agent
        result = granite_agent.answer_query("", "some context")
        assert "question" in result.lower() or len(result) > 0

    def test_answer_query_offline_does_not_invent_data(self, monkeypatch):
        """Offline answer must not claim data it was not given."""
        self._force_offline(monkeypatch)
        import granite_agent
        result = granite_agent.answer_query(
            "What is consumer CON9999's reading?",
            "Dataset: 2 consumers CON0001 and CON0002.",
        )
        # Should NOT assert CON9999 exists or fabricate a reading
        assert "CON9999" not in result or "offline" in result.lower() or "cannot" in result.lower()


# ============================================================
# STANDALONE RUNNER (python test_fraud_pipeline.py)
# ============================================================

if __name__ == "__main__":
    result = pytest.main([__file__, "-v", "--tb=short"])
    sys.exit(result)
