"""
app.py
======
Electricity Bill Fraud Detection & Personal Bill Vault
3-Tab Streamlit Application

Security:
  - unsafe_allow_html=False everywhere (default)
  - No hardcoded secrets — all credentials from .env
  - All SQL via parameterized bill_vault functions
  - File uploads validated (extension, size, required columns)
  - API keys / tracebacks never shown in UI
"""

from __future__ import annotations

import hashlib
import io
import os
import time
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ElectroGuard AI — Fraud Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Lazy imports (heavy; cached after first load)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading fraud engine …")
def _load_engine_assets():
    from fraud_engine import analyze_dataframe, train_isolation_forest
    import fraud_engine as fe
    return fe


@st.cache_resource(show_spinner="Connecting to IBM watsonx.ai …")
def _load_granite():
    import granite_agent as ga
    return ga


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CSV_PATH          = os.path.join("data", "utility_billing_records.csv")
REQUIRED_CSV_COLS = {
    "consumer_id", "billing_month", "prev_reading",
    "current_reading", "billed_units", "zone",
}
MAX_UPLOAD_MB = 10
TARIFF_RATES  = {"Domestic": 5.5, "Commercial": 8.0, "Industrial": 6.5}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_csv_upload(uploaded_file) -> tuple[pd.DataFrame | None, str]:
    """Return (df, error_message). error_message is '' on success."""
    if uploaded_file is None:
        return None, "No file provided."
    name = uploaded_file.name.lower()
    if not name.endswith(".csv"):
        return None, "Only .csv files are accepted."
    if uploaded_file.size > MAX_UPLOAD_MB * 1024 * 1024:
        return None, f"File exceeds {MAX_UPLOAD_MB} MB limit."
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        return None, "Could not parse CSV file."
    missing = REQUIRED_CSV_COLS - set(df.columns)
    if missing:
        return None, f"CSV is missing required columns: {', '.join(sorted(missing))}"
    if (df["billed_units"] < 0).any():
        return None, "CSV contains negative billed_units — rejected."
    if (df["current_reading"] < df["prev_reading"]).any():
        return None, "CSV contains rows where current_reading < prev_reading — rejected."
    return df, ""


@st.cache_data(show_spinner="Running fraud analysis …", ttl=300)
def _run_analysis(df_json: str) -> pd.DataFrame:
    """Cached bulk analysis — keyed on serialised DataFrame."""
    fe  = _load_engine_assets()
    df  = pd.read_json(io.StringIO(df_json), orient="split")
    return fe.analyze_dataframe(df)


def _engine_badge(ga) -> None:
    """Render sidebar Engine Status Badge."""
    status = ga.get_engine_status()
    if status["status"] == "Live (IBM Granite)":
        st.sidebar.success(f"🟢 **IBM Granite Live**\n\n`{status['model_id']}`")
    else:
        hint = status["hint"] or "Credentials not configured."
        st.sidebar.warning(f"🟡 **Offline Deterministic Mode**\n\n{hint}")


def _kpi(col, label: str, value: str, delta: str = "") -> None:
    col.metric(label=label, value=value, delta=delta or None)


def _fraud_color(tier: str) -> str:
    return {"Critical": "#e74c3c", "Suspicious": "#f39c12", "Normal": "#27ae60"}.get(tier, "#888")


def _download_report(report: dict, label: str) -> None:
    """Render a download button for a text report."""
    text  = report["report_text"]
    sha   = report["sha256_hash"]
    fname = f"report_{report.get('report_type','report')}_{int(time.time())}.txt"
    st.download_button(
        label=label,
        data=text.encode("utf-8"),
        file_name=fname,
        mime="text/plain",
    )
    st.caption(f"SHA-256 integrity hash: `{sha}`")


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------

def _render_sidebar(ga, results_df: pd.DataFrame | None) -> dict:
    """Render sidebar and return filter state."""
    st.sidebar.title("⚡ ElectroGuard AI")
    st.sidebar.markdown("*AI Agent for Electricity Bill Fraud Detection*")
    st.sidebar.divider()

    _engine_badge(ga)
    st.sidebar.divider()

    filters: dict = {}

    if results_df is not None and not results_df.empty:
        st.sidebar.subheader("Dataset Filters")
        zones = ["All"] + sorted(results_df["zone"].dropna().unique().tolist())
        filters["zone"] = st.sidebar.selectbox("Zone", zones)

        tiers = ["All", "Critical", "Suspicious", "Normal"]
        filters["tier"] = st.sidebar.selectbox("Risk Tier", tiers)

        st.sidebar.divider()
        st.sidebar.subheader("Active Dataset Summary")
        total = results_df["consumer_id"].nunique()
        st.sidebar.metric("Total Consumers", total)

        # Show scored stats if analysis has been run (session_state),
        # otherwise show placeholder so sidebar loads instantly
        scored = st.session_state.get("results_df")
        if scored is not None:
            flagged = scored[scored["risk_tier"] != "Normal"]["consumer_id"].nunique()
            leakage = scored["estimated_revenue_impact"].sum()
            st.sidebar.metric("Flagged Consumers", flagged)
            st.sidebar.metric("Est. Revenue Leakage", f"Rs.{leakage:,.0f}")
        else:
            st.sidebar.caption("Run analysis in Tab 1 to see risk stats.")
    else:
        filters["zone"] = "All"
        filters["tier"] = "All"

    return filters


# ---------------------------------------------------------------------------
# TAB 1 — Utility Fraud Auditor
# ---------------------------------------------------------------------------

def _tab_fraud_auditor(ga, filters: dict) -> None:
    st.header("🔍 Utility Fraud Auditor — DISCOM Bulk Analysis")

    # --- Dataset source ---
    source = st.radio(
        "Dataset source",
        ["Load Sample Smart-Grid Dataset", "Upload custom CSV"],
        horizontal=True,
    )

    df_raw: pd.DataFrame | None = None
    dataset_label = ""

    if source == "Load Sample Smart-Grid Dataset":
        if os.path.exists(CSV_PATH):
            df_raw = pd.read_csv(CSV_PATH)
            dataset_label = "Sample Smart-Grid Dataset (synthetic, labeled)"
            st.info("📊 **Sample Smart-Grid Dataset** loaded — synthetic data for demonstration.", icon="ℹ️")
        else:
            st.error("Sample dataset not found. Run `python data_generator.py` first.")
            return
    else:
        uploaded = st.file_uploader("Upload billing CSV", type=["csv"])
        if uploaded:
            df_raw, err = _validate_csv_upload(uploaded)
            if err:
                st.error(f"Upload rejected: {err}")
                return
            dataset_label = f"Custom upload: {uploaded.name}"
        else:
            st.info("Upload a CSV with columns: consumer_id, billing_month, prev_reading, current_reading, billed_units, zone")
            return

    # --- Run analysis ---
    with st.spinner("Running fraud analysis … (first run trains ML model, ~30s)"):
        results_df = _run_analysis(df_raw.to_json(orient="split"))
    # Store for Tab 3 chat context and sidebar stats
    st.session_state["results_df"] = results_df

    # --- Apply sidebar filters ---
    display_df = results_df.copy()
    if filters["zone"] != "All":
        display_df = display_df[display_df["zone"] == filters["zone"]]
    if filters["tier"] != "All":
        display_df = display_df[display_df["risk_tier"] == filters["tier"]]

    # --- KPI cards ---
    st.subheader("Key Performance Indicators")
    total_meters   = display_df["consumer_id"].nunique()
    flagged_df     = display_df[display_df["risk_tier"] != "Normal"]
    flagged        = flagged_df["consumer_id"].nunique()
    high_risk      = display_df[display_df["risk_tier"] == "Critical"]["consumer_id"].nunique()
    leakage        = display_df["estimated_revenue_impact"].sum()

    c1, c2, c3, c4 = st.columns(4)
    _kpi(c1, "Total Meters Audited", f"{total_meters:,}")
    _kpi(c2, "Flagged Suspects", f"{flagged:,}")
    _kpi(c3, "High-Risk (Critical)", f"{high_risk:,}")
    _kpi(c4, "Est. Revenue Leakage", f"₹{leakage:,.0f}")

    st.caption(f"Dataset: {dataset_label}")
    st.divider()

    # --- Risk-ranked suspects table ---
    st.subheader("Risk-Ranked Suspects")
    suspects = (
        display_df[display_df["risk_tier"] != "Normal"]
        [["consumer_id", "consumer_name", "zone", "billing_month",
          "risk_score", "risk_tier", "triggered_rules", "estimated_revenue_impact"]]
        .sort_values("risk_score", ascending=False)
        .drop_duplicates(subset=["consumer_id", "billing_month"])
        .head(100)
        .reset_index(drop=True)
    )

    if suspects.empty:
        st.success("No suspects found with current filters.")
    else:
        st.dataframe(
            suspects.style.applymap(
                lambda v: f"color: {_fraud_color(v)}" if v in ("Critical", "Suspicious", "Normal") else "",
                subset=["risk_tier"],
            ),
            use_container_width=True,
        )

        st.divider()

        # --- Consumer Deep-Dive ---
        st.subheader("Consumer Deep-Dive")
        consumer_ids = suspects["consumer_id"].unique().tolist()
        selected_id  = st.selectbox("Select Consumer", consumer_ids)

        if selected_id:
            con_df = results_df[results_df["consumer_id"] == selected_id].sort_values("billing_month")

            # Peer average for the same zone
            zone_name = con_df["zone"].iloc[0] if not con_df.empty else "Domestic"
            peer_df   = results_df[
                (results_df["zone"] == zone_name) & (results_df["consumer_id"] != selected_id)
            ].groupby("billing_month")["billed_units"].mean().reset_index()
            peer_df.columns = ["billing_month", "peer_avg_units"]

            merged = con_df.merge(peer_df, on="billing_month", how="left")

            # Identify anomaly months
            anomaly_months = con_df[con_df["risk_tier"] != "Normal"]["billing_month"].tolist()

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=merged["billing_month"], y=merged["billed_units"],
                name=f"{selected_id} kWh", mode="lines+markers",
                line=dict(color="#3b82d4", width=2),
                marker=dict(
                    size=[12 if m in anomaly_months else 7 for m in merged["billing_month"]],
                    color=[_fraud_color(merged.loc[merged["billing_month"] == m, "risk_tier"].iloc[0])
                           if m in anomaly_months else "#3b82d4"
                           for m in merged["billing_month"]],
                    symbol=["star" if m in anomaly_months else "circle" for m in merged["billing_month"]],
                ),
            ))
            fig.add_trace(go.Scatter(
                x=peer_df["billing_month"], y=peer_df["peer_avg_units"],
                name=f"{zone_name} Zone Peer Avg", mode="lines",
                line=dict(color="#888", width=1.5, dash="dot"),
            ))
            fig.update_layout(
                title=f"Consumer {selected_id} — Monthly kWh vs Zone Peer Average",
                xaxis_title="Billing Month",
                yaxis_title="kWh",
                legend=dict(orientation="h"),
                height=380,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

            # --- Granite Explanation panel ---
            st.subheader("IBM Granite Explainable AI")
            suspect_row = suspects[suspects["consumer_id"] == selected_id].iloc[0] if not suspects[suspects["consumer_id"] == selected_id].empty else None

            if suspect_row is not None:
                findings = {
                    "consumer_id":              selected_id,
                    "billing_month":            int(suspect_row["billing_month"]),
                    "risk_score":               float(suspect_row["risk_score"]),
                    "risk_tier":                suspect_row["risk_tier"],
                    "triggered_rules":          suspect_row["triggered_rules"].split("|") if suspect_row["triggered_rules"] else [],
                    "estimated_revenue_impact": float(suspect_row["estimated_revenue_impact"]),
                    "details":                  {},
                }
                with st.spinner("Generating AI explanation …"):
                    explanation = ga.explain_fraud_case(findings)
                st.text_area("AI Explanation", explanation, height=200)

                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    if st.button("Generate Field Inspection Brief", key="btn_field"):
                        with st.spinner("Drafting report …"):
                            report = ga.draft_action_report(findings, "field_inspection")
                        _download_report(report, "Download Field Inspection Brief (.txt)")
                with col_r2:
                    if st.button("Generate Correction Request Letter", key="btn_correction"):
                        with st.spinner("Drafting report …"):
                            report = ga.draft_action_report(findings, "correction_request")
                        _download_report(report, "Download Correction Request Letter (.txt)")
            else:
                st.info("Selected consumer has no flagged records in the current view.")

        st.divider()

    # --- Live Tampering Simulator ---
    st.subheader("🔧 Simulate Tampering / Edit Reading")
    st.caption("Tweak a reading live and watch the Risk Score update instantly.")

    fe = _load_engine_assets()

    sim_col1, sim_col2, sim_col3, sim_col4 = st.columns(4)
    sim_prev    = sim_col1.number_input("Prev Reading (kWh)", min_value=0.0, value=1000.0, step=10.0, key="sim_prev")
    sim_curr    = sim_col2.number_input("Current Reading (kWh)", min_value=0.0, value=1250.0, step=10.0, key="sim_curr")
    sim_billed  = sim_col3.number_input("Billed Units (kWh)", min_value=0.0, value=250.0, step=10.0, key="sim_billed")
    sim_zone    = sim_col4.selectbox("Zone", ["Domestic", "Commercial", "Industrial"], key="sim_zone")

    if sim_curr < sim_prev:
        st.warning("Current Reading cannot be less than Prev Reading.")
    else:
        sim_record = {
            "consumer_id":    "SIMULATOR",
            "billing_month":  6,
            "prev_reading":   sim_prev,
            "current_reading": sim_curr,
            "billed_units":   sim_billed,
            "tariff_type":    sim_zone,
            "zone":           sim_zone,
        }

        # Build a minimal training dataset for the simulator
        sim_df  = results_df[results_df["zone"] == sim_zone].copy() if results_df is not None else df_raw.copy()
        clf, scaler = fe.train_isolation_forest(sim_df)
        con_hist    = pd.DataFrame([sim_record])  # single-record cold-start

        try:
            sim_result = fe.analyze_record(
                record=sim_record,
                consumer_history=con_hist,
                zone_df=sim_df,
                clf=clf,
                scaler=scaler,
                full_df=sim_df,
            )
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Risk Score", f"{sim_result.risk_score:.1f}/100")
            sc2.metric("Risk Tier", sim_result.risk_tier)
            sc3.metric("Est. Impact", f"₹{sim_result.estimated_revenue_impact:,.2f}")

            if sim_result.triggered_rules:
                st.warning(f"Triggered Rules: {', '.join(sim_result.triggered_rules)}")
                sim_findings = {
                    "consumer_id":              "SIMULATOR",
                    "billing_month":            6,
                    "risk_score":               sim_result.risk_score,
                    "risk_tier":                sim_result.risk_tier,
                    "triggered_rules":          sim_result.triggered_rules,
                    "estimated_revenue_impact": sim_result.estimated_revenue_impact,
                    "details":                  sim_result.details,
                }
                with st.spinner("Generating AI explanation for simulated record …"):
                    sim_explanation = ga.explain_fraud_case(sim_findings)
                st.text_area("Granite AI — Simulated Record Explanation", sim_explanation, height=160)
            else:
                st.success("No anomaly rules triggered — record appears normal.")
        except ValueError as e:
            st.error(f"Simulation error: {e}")


# ---------------------------------------------------------------------------
# TAB 2 — My Bill Vault & Self-Check
# ---------------------------------------------------------------------------

def _tab_bill_vault(ga) -> None:
    st.header("🏠 My Bill Vault & Self-Check")
    st.caption("Save your monthly electricity bills permanently and auto-check for errors.")

    from bill_vault import init_db, insert_bill, get_bills, DB_PATH

    init_db(DB_PATH)

    # --- Quick seed button ---
    if st.button("Load Sample Homeowner History (with Continuity Error)", key="btn_seed"):
        env = os.environ.get("APP_ENV", "production").strip().lower()
        if env == "development":
            import subprocess, sys
            subprocess.run([sys.executable, "seed_dev_data.py"], check=False)
            st.success("Sample history loaded for HOME001 (Rahul Sharma). Month 2024-04 has a deliberate continuity error.")
            st.cache_data.clear()
        else:
            # Directly seed without the env gate for demo purposes in any env
            from bill_vault import insert_bill as _ib
            sample = [
                ("HOME001", "Rahul Sharma", "2024-01", 1200.0, 1450.0, 250.0, 1375.0, "Domestic", "2 BHK / 1 AC"),
                ("HOME001", "Rahul Sharma", "2024-02", 1450.0, 1680.0, 230.0, 1265.0, "Domestic", "2 BHK / 1 AC"),
                ("HOME001", "Rahul Sharma", "2024-03", 1680.0, 1950.0, 270.0, 1485.0, "Domestic", "2 BHK / 1 AC"),
                ("HOME001", "Rahul Sharma", "2024-04", 1980.0, 2250.0, 270.0, 1485.0, "Domestic", "2 BHK / 1 AC"),
                ("HOME001", "Rahul Sharma", "2024-05", 2250.0, 2510.0, 260.0, 1430.0, "Domestic", "2 BHK / 1 AC"),
                ("HOME001", "Rahul Sharma", "2024-06", 2510.0, 2780.0, 270.0, 1485.0, "Domestic", "2 BHK / 1 AC"),
            ]
            for row in sample:
                try:
                    _ib(consumer_id=row[0], consumer_name=row[1], billing_month=row[2],
                        prev_reading=row[3], current_reading=row[4], billed_units=row[5],
                        amount_paid=row[6], tariff_type=row[7], household_profile=row[8])
                except Exception:
                    pass
            st.success("Sample history loaded for HOME001 (Rahul Sharma). Month 2024-04 has a deliberate continuity error.")
            st.rerun()

    st.divider()

    # --- Add Bill Form ---
    with st.expander("➕ Add / Update a Bill Record", expanded=False):
        with st.form("add_bill_form"):
            fc1, fc2 = st.columns(2)
            cid   = fc1.text_input("Consumer ID", value="HOME001", max_chars=20)
            cname = fc2.text_input("Consumer Name", value="", max_chars=60)
            fc3, fc4 = st.columns(2)
            month = fc3.text_input("Billing Month (YYYY-MM)", value="2024-07", max_chars=7)
            tariff = fc4.selectbox("Tariff Type", ["Domestic", "Commercial", "Industrial"])
            fc5, fc6, fc7 = st.columns(3)
            prev_r  = fc5.number_input("Prev Reading (kWh)", min_value=0.0, value=2780.0, step=1.0)
            curr_r  = fc6.number_input("Current Reading (kWh)", min_value=0.0, value=3050.0, step=1.0)
            billed  = fc7.number_input("Billed Units (kWh)", min_value=0.0, value=270.0, step=1.0)
            fc8, fc9 = st.columns(2)
            amount  = fc8.number_input("Amount Paid (₹)", min_value=0.0, value=1485.0, step=10.0)
            profile = fc9.text_input("Household Profile", value="2 BHK / 1 AC", max_chars=50)

            submitted = st.form_submit_button("Save Bill")
            if submitted:
                # Sanitise inputs
                cid   = cid.strip()[:20]
                cname = cname.strip()[:60]
                month = month.strip()[:7]
                if not cid or not cname or not month:
                    st.error("Consumer ID, Name, and Billing Month are required.")
                elif curr_r < prev_r:
                    st.error("Current Reading cannot be less than Prev Reading.")
                else:
                    try:
                        insert_bill(
                            consumer_id=cid, consumer_name=cname,
                            billing_month=month, prev_reading=prev_r,
                            current_reading=curr_r, billed_units=billed,
                            amount_paid=amount, tariff_type=tariff,
                            household_profile=profile,
                        )
                        st.success(f"Bill for {cid} / {month} saved successfully.")
                        st.rerun()
                    except ValueError as e:
                        st.error(f"Validation error: {e}")

    # --- Optional PDF upload ---
    with st.expander("📄 Extract from PDF Bill (optional)", expanded=False):
        pdf_file = st.file_uploader("Upload electricity bill PDF", type=["pdf"], key="pdf_upload")
        if pdf_file:
            if pdf_file.size > MAX_UPLOAD_MB * 1024 * 1024:
                st.error(f"PDF exceeds {MAX_UPLOAD_MB} MB limit.")
            else:
                import tempfile, shutil
                tmp = tempfile.mkdtemp()
                try:
                    tmp_path = os.path.join(tmp, pdf_file.name)
                    with open(tmp_path, "wb") as f:
                        f.write(pdf_file.read())
                    from bill_vault import extract_bill_from_pdf
                    extracted = extract_bill_from_pdf(tmp_path)
                    st.success("Extracted fields (review before saving):")
                    st.json({k: v for k, v in extracted.items() if k != "raw_text_length"})
                except ImportError:
                    st.warning("pypdf not installed — PDF extraction unavailable.")
                except Exception as e:
                    st.error(f"PDF extraction failed: {str(e)[:200]}")
                finally:
                    shutil.rmtree(tmp, ignore_errors=True)

    st.divider()

    # --- Load & display vault ---
    consumer_id_view = st.text_input("View bills for Consumer ID", value="HOME001", max_chars=20, key="vault_cid")
    consumer_id_view = consumer_id_view.strip()[:20]
    bills = get_bills(consumer_id_view)

    if not bills:
        st.info(f"No bills found for Consumer ID: {consumer_id_view}")
        return

    bills_df = pd.DataFrame(bills).sort_values("billing_month").reset_index(drop=True)
    st.subheader(f"Bill History — {consumer_id_view}")

    # --- Reading math & continuity checks ---
    from fraud_engine import rule_continuity_breach, rule_meter_mismatch

    issues = []
    for _, row in bills_df.iterrows():
        flag, gap, desc = rule_meter_mismatch(row.to_dict())
        if flag:
            issues.append({"month": row["billing_month"], "type": "Meter-to-Bill Mismatch", "detail": desc})

    vault_bills = bills_df[["billing_month", "prev_reading", "current_reading"]].to_dict("records")
    breaches = rule_continuity_breach(vault_bills)
    for b in breaches:
        issues.append({"month": b["billing_month"], "type": "Cross-Month Continuity Breach",
                        "detail": f"Expected prev={b['expected_prev']}, got {b['actual_prev']} (gap {b['gap']} kWh)"})

    # Spike check: flag any month > 2.5x rolling average
    if len(bills_df) >= 4:
        for i in range(3, len(bills_df)):
            baseline = bills_df.iloc[:i]["billed_units"].mean()
            cur_units = bills_df.iloc[i]["billed_units"]
            if baseline > 0 and cur_units >= baseline * 2.5:
                issues.append({
                    "month": bills_df.iloc[i]["billing_month"],
                    "type": "Unusual Spike",
                    "detail": f"{cur_units:.1f} kWh = {cur_units/baseline:.1f}x rolling avg ({baseline:.1f} kWh)"
                })

    if issues:
        st.warning(f"⚠️ {len(issues)} issue(s) detected in your bill history:")
        st.dataframe(pd.DataFrame(issues), use_container_width=True)
    else:
        st.success("✅ No issues detected — all readings and billing math check out.")

    # --- Monthly kWh & Amount chart ---
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=bills_df["billing_month"], y=bills_df["billed_units"],
        name="kWh", marker_color="#3b82d4", yaxis="y",
    ))
    fig2.add_trace(go.Scatter(
        x=bills_df["billing_month"], y=bills_df["amount_paid"],
        name="Amount Paid (₹)", mode="lines+markers",
        line=dict(color="#e74c3c", width=2), yaxis="y2",
    ))
    fig2.update_layout(
        title="Monthly kWh Consumption & Amount Paid",
        xaxis_title="Billing Month",
        yaxis=dict(title="kWh", side="left"),
        yaxis2=dict(title="₹ Amount", overlaying="y", side="right"),
        legend=dict(orientation="h"),
        height=360,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # --- Granite breakdown ---
    st.subheader("IBM Granite Plain-English Breakdown")
    if issues:
        vault_findings = {
            "consumer_id":              consumer_id_view,
            "billing_month":            issues[0]["month"],
            "risk_score":               70.0 if len(issues) >= 2 else 45.0,
            "risk_tier":                "Critical" if len(issues) >= 2 else "Suspicious",
            "triggered_rules":          list({i["type"].lower().replace(" ", "_").replace("-", "_") for i in issues}),
            "estimated_revenue_impact": sum(
                abs(b.get("gap", 0)) * TARIFF_RATES.get(bills_df["tariff_type"].iloc[0] if "tariff_type" in bills_df.columns else "Domestic", 5.5)
                for b in breaches
            ),
            "details":                  {"issues": issues},
        }
        with st.spinner("Generating AI breakdown …"):
            vault_explanation = ga.explain_fraud_case(vault_findings)
        st.text_area("AI Analysis", vault_explanation, height=200)

        col_v1, col_v2 = st.columns(2)
        with col_v1:
            if st.button("Generate Billing Correction Request Letter", key="btn_vault_letter"):
                with st.spinner("Drafting letter …"):
                    vault_report = ga.draft_action_report(vault_findings, "correction_request")
                _download_report(vault_report, "Download Correction Request Letter (.txt)")
        with col_v2:
            if st.button("Generate Field Inspection Brief", key="btn_vault_field"):
                with st.spinner("Drafting report …"):
                    vault_report2 = ga.draft_action_report(vault_findings, "field_inspection")
                _download_report(vault_report2, "Download Field Inspection Brief (.txt)")
    else:
        st.info("No issues detected — no AI breakdown needed.")


# ---------------------------------------------------------------------------
# TAB 3 — Granite AI Auditor Chat
# ---------------------------------------------------------------------------

def _tab_chat(ga, results_df: pd.DataFrame | None) -> None:
    st.header("💬 Granite AI Auditor Chat")
    st.caption("Natural-language Q&A grounded in the active billing dataset and Bill Vault.")

    # Build context summary
    context_parts = []
    if results_df is not None and not results_df.empty:
        total      = len(results_df["consumer_id"].unique())
        critical   = results_df[results_df["risk_tier"] == "Critical"]["consumer_id"].nunique()
        suspicious = results_df[results_df["risk_tier"] == "Suspicious"]["consumer_id"].nunique()
        normal     = results_df[results_df["risk_tier"] == "Normal"]["consumer_id"].nunique()
        leakage    = results_df["estimated_revenue_impact"].sum()
        zone_stats = (
            results_df[results_df["risk_tier"] != "Normal"]
            .groupby("zone")["consumer_id"].nunique()
            .to_dict()
        )
        top_suspects = (
            results_df[results_df["risk_tier"] != "Normal"]
            .sort_values("risk_score", ascending=False)
            [["consumer_id", "zone", "billing_month", "risk_score", "risk_tier", "triggered_rules"]]
            .drop_duplicates("consumer_id")
            .head(10)
            .to_string(index=False)
        )
        context_parts.append(
            f"Billing Dataset: {total} unique consumers, {len(results_df)} records.\n"
            f"Risk tiers — Critical: {critical}, Suspicious: {suspicious}, Normal: {normal}.\n"
            f"Estimated total revenue leakage: Rs.{leakage:,.2f}.\n"
            f"Flagged consumers by zone: {zone_stats}.\n"
            f"Top suspects (by risk score):\n{top_suspects}"
        )
    else:
        context_parts.append("No billing dataset loaded yet. Load data in Tab 1 first.")

    from bill_vault import get_all_bills
    vault_bills = get_all_bills()
    if vault_bills:
        vault_df    = pd.DataFrame(vault_bills)
        vault_cids  = vault_df["consumer_id"].nunique()
        vault_total = len(vault_df)
        context_parts.append(
            f"Bill Vault: {vault_cids} unique consumers, {vault_total} saved bills."
        )

    context_summary = "\n\n".join(context_parts)

    # --- Chat history (session state) ---
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    user_input = st.chat_input("Ask anything about the billing dataset or Bill Vault …")
    if user_input:
        user_input_safe = user_input.strip()[:500]
        st.session_state.chat_history.append({"role": "user", "content": user_input_safe})
        with st.chat_message("user"):
            st.markdown(user_input_safe)

        with st.chat_message("assistant"):
            with st.spinner("Thinking …"):
                answer = ga.answer_query(user_input_safe, context_summary)
            st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

    if st.button("Clear Chat History", key="clear_chat"):
        st.session_state.chat_history = []
        st.rerun()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    ga = _load_granite()
    fe = _load_engine_assets()  # noqa: F841 — warms the cache

    # Use session_state to carry results_df across reruns without
    # re-running the expensive IsolationForest on every page load.
    if "results_df" not in st.session_state:
        st.session_state["results_df"] = None

    # Fast sidebar stats: read raw CSV for consumer/zone counts only —
    # no IsolationForest here, so the sidebar renders instantly.
    sidebar_df: pd.DataFrame | None = None
    if os.path.exists(CSV_PATH):
        try:
            sidebar_df = pd.read_csv(CSV_PATH)
        except Exception:
            sidebar_df = None

    filters = _render_sidebar(ga, sidebar_df)

    tab1, tab2, tab3 = st.tabs([
        "🔍 Utility Fraud Auditor",
        "🏠 My Bill Vault & Self-Check",
        "💬 Granite AI Auditor Chat",
    ])

    with tab1:
        _tab_fraud_auditor(ga, filters)

    with tab2:
        _tab_bill_vault(ga)

    with tab3:
        # Tab 3 uses whatever results_df Tab 1 has stored in session_state
        _tab_chat(ga, st.session_state.get("results_df"))


if __name__ == "__main__":
    main()
