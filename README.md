# ⚡ ElectroGuard AI
### AI Agent for Electricity Bill Fraud Detection
**Problem Statement No. 18 — IBM Hackathon**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![IBM watsonx](https://img.shields.io/badge/IBM%20watsonx.ai-Granite-0530AD?style=flat&logo=ibm&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-IsolationForest-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-Bill%20Vault-003B57?style=flat&logo=sqlite&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-45%20Passed-27ae60?style=flat&logo=pytest&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## 📌 Problem Statement

Electricity theft and billing fraud cost Indian utilities an estimated **₹20,000+ crore annually**. Fraudulent activities range from meter tampering and illegal bypasses to billing system mismatches and reading falsification. Normal consumers are also affected by double-billing and cross-month reading errors that go undetected for months.

**ElectroGuard AI** addresses both sides of this problem:
- **For DISCOMs / Utilities** — bulk automated fraud audit with explainable AI
- **For Normal Consumers** — personal bill vault with automatic error detection

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT FRONTEND                        │
│  Tab 1: Utility Fraud Auditor  │  Tab 2: My Bill Vault      │
│  Tab 3: Granite AI Chat        │  Sidebar: Engine Status     │
└───────────────────┬─────────────────────────────────────────┘
                    │
         ┌──────────▼──────────┐
         │   STAGE 1 ENGINE    │  fraud_engine.py
         │  (Zero LLM, <100ms) │
         │                     │
         │  Rule 1: Meter-Bill │
         │  Rule 2: Continuity │
         │  Rule 3: Peer Drop  │
         │  Rule 4: Flatline/  │
         │          Spike      │
         │  IsolationForest +  │
         │  Z-Score → JSON     │
         └──────────┬──────────┘
                    │  Verified findings JSON only
         ┌──────────▼──────────┐
         │   STAGE 2 ENGINE    │  granite_agent.py
         │  IBM Granite LLM    │
         │                     │
         │  explain_fraud_case │
         │  draft_action_report│
         │  answer_query       │
         │  SHA-256 integrity  │
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────┐
         │   DATA LAYER        │
         │  SQLite Bill Vault  │  bill_vault.py
         │  Smart-Grid CSV     │  data_generator.py
         └─────────────────────┘
```

---

## ✨ Key Features

### 🔍 Tab 1 — Utility Fraud Auditor (DISCOM)
- Load built-in **Sample Smart-Grid Dataset** (300 consumers × 12 months) or upload custom CSV
- **Dynamic KPI Cards** — Total Meters Audited, Flagged Suspects, High-Risk Alerts, Estimated Revenue Leakage ₹
- **Risk-Ranked Suspects Table** with colour-coded tiers (Critical / Suspicious / Normal)
- **Consumer Deep-Dive Chart** — individual kWh vs zone peer average with anomaly months starred
- **IBM Granite Explainable AI Panel** — plain-English explanation of exactly why a meter was flagged
- **1-click Field Inspection Brief** download with SHA-256 integrity hash
- **Live Tampering Simulator** — tweak any reading and watch risk score + AI explanation update instantly

### 🏠 Tab 2 — My Bill Vault & Self-Check (Consumer)
- Save monthly electricity bills permanently in SQLite
- Optional **PDF bill extraction** with strict numeric plausibility guards
- **1-click sample history** with a pre-loaded cross-month continuity error for demo
- Automatic checks: reading math, cross-month continuity, spike detection
- **Downloadable Billing Correction Request Letter** (IBM Granite drafted, SHA-256 signed)

### 💬 Tab 3 — Granite AI Auditor Chat
- Natural-language Q&A grounded in the active billing dataset and Bill Vault
- Context-aware answers — Granite never fabricates readings or amounts
- Full chat history within the session

---

## 🤖 Two-Stage Fraud Detection Pipeline

### Stage 1 — Deterministic Rules + ML (Zero LLM Dependency)

| Rule | Trigger Condition | What It Catches |
|---|---|---|
| **Rule 1** Meter-to-Bill Mismatch | `billed_units ≠ (current − prev)` beyond ±2% | Bill manipulation, reading falsification |
| **Rule 2** Cross-Month Continuity | `month_N.prev ≠ month_(N-1).current` | Double-billing, reading jumps |
| **Rule 3** Peer-Normalized Drop | ≥55% drop vs 6-month baseline AND peers did NOT drop | Meter bypass / illegal tap |
| **Rule 4** Flatline / Spike | ≥2 consecutive 0-kWh months OR ≥250% spike | Dead meter, unauthorized sub-metering |
| **ML Layer** | IsolationForest + Z-Score | Multivariate anomalies missed by rules |

> **Anti-False-Positive Design:** Rule 3 checks whether the entire zone/category had a similar dip (winter/monsoon seasonal variation). If peers also dropped ≥20%, the flag is suppressed — zero seasonal false positives.

**Output per record:**
```json
{
  "risk_score": 82.5,
  "risk_tier": "Critical",
  "triggered_rules": ["meter_to_bill_mismatch", "peer_normalized_drop"],
  "estimated_revenue_impact": 1870.50,
  "details": { "rule1": "Metered 310 kWh vs billed 120 kWh (gap 190 kWh, 61.3%)" }
}
```

### Stage 2 — IBM Granite Explainable AI

Granite receives **only verified Stage-1 JSON** — never raw CSV data, never user free-text. This eliminates hallucination of meter readings or bill amounts.

| Function | Output |
|---|---|
| `explain_fraud_case()` | Plain-English explanation with exact numerical gaps and ₹ impact |
| `draft_action_report()` | Field Inspection Brief (utility) or Correction Request Letter (consumer) |
| `answer_query()` | Grounded conversational Q&A over dataset context |

All reports include a **SHA-256 verification hash** for tamper detection.

---

## 🚀 Quick Start

### Prerequisites
```bash
python --version   # 3.10 or higher required
git --version
```

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/electroguard-ai.git
cd electroguard-ai
```

### 2. Create Virtual Environment & Install
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure IBM Cloud Credentials
```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Edit `.env` with your credentials:
```env
WATSONX_APIKEY=your_ibm_cloud_api_key
WATSONX_PROJECT_ID=your_watsonx_project_id
WATSONX_URL=https://us-south.ml.cloud.ibm.com
WATSONX_MODEL_ID=ibm/granite-13b-instruct-v2
APP_ENV=development
SESSION_SECRET=any_random_32_character_string
```

> **Works without credentials too.** The app runs fully in **Offline Deterministic Mode** — the sidebar shows a yellow 🟡 badge. Set credentials to switch to 🟢 **IBM Granite Live**.

### 4. Generate Sample Dataset
```bash
python data_generator.py
```
Creates `data/utility_billing_records.csv` — 300 consumers × 12 months, 4 anomaly types.

### 5. (Optional) Seed Bill Vault Demo Data
```bash
# Requires APP_ENV=development in .env
python seed_dev_data.py
```
Seeds 6 months of bills for HOME001 with a deliberate cross-month continuity error in 2024-04.

### 6. Run the App
```bash
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)**

---

## 🧪 Running Tests

```bash
python -m pytest test_fraud_pipeline.py -v
```

**45 tests — 100% pass rate:**

| Test Class | Tests | Coverage |
|---|---|---|
| `TestRule1MeterMismatch` | 5 | Exact match, tolerance, mismatch, zero-meter, overbilling |
| `TestRule2ContinuityBreach` | 5 | Clean history, single bill, breach detection, multiple breaches |
| `TestRule3PeerNormalizedDrop` | 4 | Drop flagged, **winter dip NOT flagged**, peer suppression, insufficient history |
| `TestRule4FlatlineSpike` | 6 | Flatline, single-zero, spike, normal, cold-start OK, cold-start error |
| `TestMLLayer` | 9 | IsoForest training, anomaly scoring, negative/invalid input rejection, end-to-end CSV |
| `TestAnalyzeRecordIntegration` | 2 | Mismatch flagged as Critical, normal record not flagged |
| `TestGraniteAgent` | 14 | Offline fallback, secret scrubbing, explain/report/query, SHA-256 integrity |

---

## 📁 Project Structure

```
electroguard-ai/
│
├── app.py                      # 3-tab Streamlit application (main entry point)
├── fraud_engine.py             # Stage-1: 4 rules + IsolationForest + Z-Score
├── granite_agent.py            # Stage-2: IBM Granite AI co-pilot
├── bill_vault.py               # SQLite Bill Vault CRUD + PDF extractor
├── data_generator.py           # Synthetic smart-grid CSV generator
├── seed_dev_data.py            # Dev-only sample data seeder (gated by APP_ENV)
├── test_fraud_pipeline.py      # 45 automated pytest tests
│
├── requirements.txt            # Python dependencies
├── .env.example                # Credential template (safe to commit)
├── .gitignore                  # Excludes .env, *.db, __pycache__, .venv
├── README.md                   # This file
│
└── data/
    ├── utility_billing_records.csv   # Generated smart-grid dataset (3,600 rows)
    └── bill_vault.db                 # SQLite database (gitignored)
```

---

## 🛡️ Security Design

| Principle | Implementation |
|---|---|
| **Zero hardcoded secrets** | All 6 credentials read exclusively from `.env` via `python-dotenv` |
| **Parameterized SQL** | Every SQLite query uses `?` placeholders — zero f-string SQL anywhere |
| **File upload validation** | Extension allowlist (`.csv`, `.pdf`), 10 MB size limit, required columns check, negative-value rejection |
| **API key scrubbing** | Error messages strip 32+ char alphanumeric tokens with `[REDACTED]` before display |
| **SHA-256 report integrity** | Every downloadable report has a verifiable hash — tamper-detectable |
| **Dev gate** | `seed_dev_data.py` only executes when `APP_ENV=development` |
| **Grounded AI** | Granite receives only verified Stage-1 JSON findings — never raw user input or CSV data |
| **No silent failures** | Engine status badge always shows `Live` or `Offline` with a safe reason |

---

## 🗂️ Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit 1.32+ |
| ML / Anomaly Detection | scikit-learn (IsolationForest), NumPy |
| Data Processing | Pandas |
| Visualization | Plotly |
| Database | SQLite (via Python `sqlite3`) |
| PDF Extraction | pypdf |
| AI Co-Pilot | IBM watsonx.ai — IBM Granite (`ibm-watsonx-ai`) |
| Credential Management | python-dotenv |
| Testing | pytest |
| Language | Python 3.10+ |

---

## 🎬 Demo Walkthrough

**Step 1 — Utility Fraud Auditor (Tab 1)**
1. Click **"Load Sample Smart-Grid Dataset"**
2. Wait ~30s for first-time ML model training (cached after that)
3. Review KPI cards — Flagged Suspects, High-Risk, Revenue Leakage
4. Pick any flagged consumer from the table → see deep-dive chart
5. Scroll to **"Simulate Tampering"** → reduce Billed Units to 50 → watch Risk Score jump to Critical + Granite explanation updates

**Step 2 — My Bill Vault (Tab 2)**
1. Click **"Load Sample Homeowner History (with Continuity Error)"**
2. See the cross-month breach flagged for **2024-04** (March current=1950 vs April prev=1980)
3. Click **"Generate Billing Correction Request Letter"** → download the SHA-256 signed PDF

**Step 3 — AI Auditor Chat (Tab 3)**
- Ask: *"Which zone has the most fraud cases?"*
- Ask: *"What is the total estimated revenue leakage?"*
- Ask: *"Explain why the top suspect was flagged"*

---

## 👥 Team / Author

Built for **IBM Hackathon — Problem Statement No. 18**
*AI Agent for Electricity Bill Fraud Detection*

---

## 📄 License

MIT License — built for demonstration and educational purposes.

---

<p align="center">
  <b>⚡ ElectroGuard AI — Protecting Utilities & Consumers with Explainable AI</b><br>
  <i>Powered by IBM Granite · scikit-learn · Streamlit</i>
</p>
