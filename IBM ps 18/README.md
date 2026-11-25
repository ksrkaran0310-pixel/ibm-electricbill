# ⚡ ElectroGuard AI — Electricity Bill Fraud Detection

> **Problem Statement No. 18** — AI Agent for Electricity Bill Fraud Detection  
> Built with Python · Streamlit · scikit-learn · Plotly · SQLite · IBM watsonx.ai (IBM Granite)

---

## Overview

ElectroGuard AI is a two-stage fraud detection system for electricity utilities and normal consumers:

| Stage | Component | Description |
|---|---|---|
| Stage 1 | `fraud_engine.py` | Deterministic rules + IsolationForest ML — zero LLM dependency |
| Stage 2 | `granite_agent.py` | IBM Granite explainable AI co-pilot — grounded on Stage-1 findings only |

**Three-tab Streamlit app:**
- **Tab 1 — Utility Fraud Auditor:** Bulk DISCOM analysis, risk-ranked suspects, deep-dive charts, live tampering simulator
- **Tab 2 — My Bill Vault:** Personal bill storage, cross-month continuity checks, PDF extraction, downloadable letters
- **Tab 3 — Granite AI Auditor Chat:** Natural-language Q&A over the active dataset

---

## Quick Start

### 1. Prerequisites

```bash
python --version   # 3.10+
```

### 2. Clone & Install

```bash
git clone <your-repo-url>
cd <repo-folder>

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure Credentials

```bash
cp .env.example .env
```

Edit `.env` and fill in your IBM Cloud credentials:

```env
WATSONX_APIKEY=your_ibm_cloud_api_key
WATSONX_PROJECT_ID=your_watsonx_project_id
WATSONX_URL=https://us-south.ml.cloud.ibm.com
WATSONX_MODEL_ID=ibm/granite-13b-instruct-v2
APP_ENV=development
SESSION_SECRET=your_random_32char_secret
```

> **Note:** The app works fully in **Offline Deterministic Mode** without credentials.  
> The sidebar will show a yellow `Offline Fallback` badge with a safe reason.

### 4. Generate Sample Data

```bash
python data_generator.py
```

This creates `data/utility_billing_records.csv` with 300 consumers × 12 months.

### 5. (Optional) Seed Personal Bill Vault Demo Data

```bash
# APP_ENV must be development
python seed_dev_data.py
```

Seeds 6 months of bills for HOME001 with a deliberate cross-month continuity error in 2024-04.

### 6. Run the App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Running Tests

```bash
python -m pytest test_fraud_pipeline.py -v
```

**45 tests** covering all 4 fraud rules, ML layer, Granite offline fallback, and SHA-256 report integrity.

---

## Project Structure

```
.
├── app.py                          # 3-tab Streamlit application
├── fraud_engine.py                 # Stage-1: Rules + IsolationForest
├── granite_agent.py                # Stage-2: IBM Granite AI co-pilot
├── bill_vault.py                   # SQLite Bill Vault CRUD + PDF extractor
├── data_generator.py               # Synthetic smart-grid dataset generator
├── seed_dev_data.py                # Dev-only sample data seeder
├── test_fraud_pipeline.py          # 45 automated tests (pytest)
├── requirements.txt
├── .env.example                    # Credential template (never commit .env)
└── data/
    ├── utility_billing_records.csv # Generated sample dataset
    └── bill_vault.db               # SQLite database (gitignored)
```

---

## Fraud Detection Rules

| Rule | Description |
|---|---|
| **Rule 1 — Meter-to-Bill Mismatch** | Flags when `billed_units ≠ (current_reading − prev_reading)` beyond ±2% tolerance |
| **Rule 2 — Cross-Month Continuity** | Flags when `month_N.prev_reading ≠ month_(N-1).current_reading` |
| **Rule 3 — Peer-Normalized Drop** | Flags ≥55% drop vs 6-month baseline ONLY if zone peers did NOT similarly drop (avoids seasonal false positives) |
| **Rule 4 — Flatline / Spike / Cold-Start** | Flags ≥2 consecutive 0-kWh months, ≥250% spikes, or first-bill reading errors |
| **ML Layer** | IsolationForest + Z-Score → `risk_score` (0–100), `risk_tier` (Critical/Suspicious/Normal) |

---

## Security Highlights

- **Zero hardcoded secrets** — all credentials read exclusively from `.env`
- **Parameterized SQL** — no f-string queries anywhere in `bill_vault.py`
- **Upload validation** — extension allowlist, 10 MB size limit, required columns check, negative-value rejection
- **API key scrubbing** — error messages strip 32+ char alphanumeric tokens before display
- **SHA-256 report integrity** — every downloadable report includes a verifiable hash
- **Dev gate** — `seed_dev_data.py` only runs when `APP_ENV=development`
- **Grounded AI** — Granite receives only verified Stage-1 JSON; never raw readings or free-form user text

---

## IBM Granite Integration

The app connects to `ibm-watsonx-ai` using credentials from `.env`.  
If credentials are absent or the endpoint is unreachable, the app falls back to **Offline Deterministic Mode** automatically — clearly labelled in the sidebar with a safe, non-secret reason.

**Granite is used for:**
1. `explain_fraud_case()` — plain-English explanation of flagged meters
2. `draft_action_report()` — Field Inspection Brief or Billing Correction Request Letter (with SHA-256 hash)
3. `answer_query()` — conversational Q&A grounded in dataset context

---

## Demo Walkthrough

1. **Tab 1:** Click "Load Sample Smart-Grid Dataset" → Review KPI cards → Select a flagged consumer → Scroll to "Simulate Tampering" and reduce Billed Units to see risk score jump live
2. **Tab 2:** Click "Load Sample Homeowner History" → Observe the cross-month continuity error flagged for 2024-04 → Download a Billing Correction Request Letter
3. **Tab 3:** Ask "Which zone has the most fraud cases?" or "Explain the top suspect's anomaly"

---

## License

MIT — for demonstration and educational purposes.
