"""
granite_agent.py
================
Stage-2 IBM Granite Explainable AI Co-Pilot.

Connects to IBM Cloud watsonx.ai using credentials from .env ONLY.
Never hardcodes API keys, project IDs, or model IDs.

Grounded contract:
  All public functions receive ONLY verified Stage-1 JSON findings.
  Granite is never asked to invent readings, units, or bill amounts.

Functions:
  get_engine_status()       -> dict  (Live vs Offline + reason)
  explain_fraud_case(...)   -> str   (plain-English explanation)
  draft_action_report(...)  -> dict  (report text + SHA-256 hash)
  answer_query(...)         -> str   (grounded Q&A)

Security:
  - Credentials loaded exclusively from os.environ (populated by python-dotenv).
  - API keys are never logged, printed, or included in error messages.
  - Tracebacks are caught and replaced with safe, non-secret hints.
  - Offline fallback is deterministic and clearly labelled.
"""

from __future__ import annotations

import hashlib
import json
import os
import textwrap
import time
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Credential loading (read-only; never echoed)
# ---------------------------------------------------------------------------

def _get_creds() -> dict[str, str]:
    """Return watsonx credentials from environment. Raises nothing — returns empties."""
    return {
        "api_key":    os.environ.get("WATSONX_APIKEY",      "").strip(),
        "project_id": os.environ.get("WATSONX_PROJECT_ID",  "").strip(),
        "url":        os.environ.get("WATSONX_URL",          "").strip(),
        "model_id":   os.environ.get("WATSONX_MODEL_ID",    "").strip(),
    }


def _creds_present(creds: dict) -> bool:
    return all(creds[k] for k in ("api_key", "project_id", "url", "model_id"))


# ---------------------------------------------------------------------------
# IBM watsonx.ai client (lazy, cached)
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict[str, Any] = {}   # key -> ModelInference instance


def _get_model():
    """
    Return a cached ModelInference instance, or None if credentials are
    missing / the SDK is not installed / the endpoint is unreachable.

    Returns (model_instance | None, error_hint | None)
    error_hint is a safe, non-secret string safe to display in the UI.
    """
    creds = _get_creds()
    if not _creds_present(creds):
        missing = [k for k, v in creds.items() if not v]
        hint    = f"Missing environment variables: {', '.join(missing).upper().replace('API_KEY','WATSONX_APIKEY')}"
        return None, hint

    cache_key = creds["model_id"] + "|" + creds["project_id"]
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key], None

    try:
        from ibm_watsonx_ai import APIClient, Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference
        from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

        credentials = Credentials(
            url=creds["url"],
            api_key=creds["api_key"],
        )
        model = ModelInference(
            model_id=creds["model_id"],
            credentials=credentials,
            project_id=creds["project_id"],
            params={
                GenParams.MAX_NEW_TOKENS: 800,
                GenParams.TEMPERATURE:    0.2,
                GenParams.TOP_P:          0.9,
            },
        )
        _MODEL_CACHE[cache_key] = model
        return model, None

    except ImportError:
        return None, "ibm-watsonx-ai package not installed."
    except Exception as exc:
        # Safe hint — never reveal the exception message if it contains key fragments
        safe_msg = str(exc)
        # Scrub anything that looks like an API key (32+ hex/alphanum chars)
        import re
        safe_msg = re.sub(r"[A-Za-z0-9_\-]{32,}", "[REDACTED]", safe_msg)
        return None, f"watsonx.ai connection failed: {safe_msg[:120]}"


# ---------------------------------------------------------------------------
# Engine status
# ---------------------------------------------------------------------------

def get_engine_status() -> dict:
    """
    Returns:
      {
        "status":  "Live (IBM Granite)" | "Offline Fallback",
        "model_id": str | None,
        "hint":    str   # safe reason if offline; empty if live
      }
    """
    model, hint = _get_model()
    if model is not None:
        creds = _get_creds()
        return {
            "status":   "Live (IBM Granite)",
            "model_id": creds["model_id"],
            "hint":     "",
        }
    return {
        "status":   "Offline Fallback",
        "model_id": None,
        "hint":     hint or "Unknown reason.",
    }


# ---------------------------------------------------------------------------
# Internal: call Granite with a prompt
# ---------------------------------------------------------------------------

def _generate(prompt: str, max_tokens: int = 800) -> str | None:
    """Call Granite. Returns response text or None on failure."""
    model, _ = _get_model()
    if model is None:
        return None
    try:
        response = model.generate_text(prompt=prompt)
        return response.strip() if isinstance(response, str) else str(response).strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Offline deterministic fallbacks
# ---------------------------------------------------------------------------

def _offline_explain(findings: dict) -> str:
    """Generate a deterministic rule-based explanation when Granite is offline."""
    cid       = findings.get("consumer_id", "Unknown")
    month     = findings.get("billing_month", "Unknown")
    rules     = findings.get("triggered_rules", [])
    score     = findings.get("risk_score", 0)
    impact    = findings.get("estimated_revenue_impact", 0)
    tier      = findings.get("risk_tier", "Normal")
    details   = findings.get("details", {})

    lines = [
        f"[Offline Deterministic Mode] Fraud Analysis — Consumer {cid}, Month {month}",
        f"Risk Tier: {tier}  |  Risk Score: {score}/100  |  Est. Revenue Impact: Rs.{impact:,.2f}",
        "",
        "Triggered Rules:",
    ]

    if not rules:
        lines.append("  None — record appears normal.")
    else:
        for rule in rules:
            if rule == "meter_to_bill_mismatch":
                d = details.get("rule1", "")
                lines.append(f"  * Meter-to-Bill Mismatch: {d}")
            elif rule == "cross_month_continuity_breach":
                breaches = details.get("rule2", [])
                for b in breaches:
                    lines.append(
                        f"  * Cross-Month Continuity Breach in {b.get('billing_month')}: "
                        f"expected prev={b.get('expected_prev')}, "
                        f"got prev={b.get('actual_prev')} (gap {b.get('gap')} kWh)"
                    )
            elif rule == "peer_normalized_drop":
                d = details.get("rule3", "")
                lines.append(f"  * Peer-Normalized Sudden Drop: {d}")
            elif rule == "flatline":
                d = details.get("rule4", "")
                lines.append(f"  * Flatline Detected: {d}")
            elif rule == "spike":
                d = details.get("rule4", "")
                lines.append(f"  * Unusual Spike: {d}")
            elif rule == "cold_start_anomaly":
                d = details.get("rule4", "")
                lines.append(f"  * Cold-Start Anomaly (Day-1 Bill): {d}")
            else:
                lines.append(f"  * {rule}")

    lines += [
        "",
        "Recommended Action:",
        "  Schedule a physical meter inspection and cross-verify readings with "
        "field engineer before issuing a corrected bill.",
    ]
    return "\n".join(lines)


def _offline_report(findings: dict, report_type: str) -> str:
    """Generate a deterministic report skeleton when Granite is offline."""
    cid    = findings.get("consumer_id", "Unknown")
    month  = findings.get("billing_month", "Unknown")
    tier   = findings.get("risk_tier", "Normal")
    impact = findings.get("estimated_revenue_impact", 0)
    rules  = findings.get("triggered_rules", [])
    ts     = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    if report_type == "field_inspection":
        title = "FIELD INSPECTION BRIEF"
        body  = textwrap.dedent(f"""
            Consumer ID   : {cid}
            Billing Month : {month}
            Risk Tier     : {tier}
            Est. Revenue Impact: Rs.{impact:,.2f}
            Triggered Flags: {', '.join(rules) if rules else 'None'}
            Generated     : {ts}

            Action Required:
            1. Dispatch field engineer to verify meter seal integrity.
            2. Cross-check meter reading against billing system entry.
            3. Interview premises occupant regarding usage patterns.
            4. Collect photographic evidence of meter display.
            5. Submit inspection report within 72 hours.
        """).strip()
    else:
        title = "BILLING CORRECTION REQUEST LETTER"
        body  = textwrap.dedent(f"""
            Consumer ID   : {cid}
            Billing Month : {month}
            Risk Tier     : {tier}
            Est. Overcharge: Rs.{impact:,.2f}
            Triggered Flags: {', '.join(rules) if rules else 'None'}
            Generated     : {ts}

            Dear Billing Department,

            I am writing to formally request a review and correction of my
            electricity bill for the above-referenced period. The automated
            billing analysis system has identified the following discrepancies:
            {chr(10).join(f'  - {r}' for r in rules) if rules else '  - See attached analysis.'}

            I request that a qualified engineer review my meter readings and
            issue a corrected bill at the earliest convenience.

            Yours sincerely,
            Consumer {cid}
        """).strip()

    return f"{'='*60}\n{title}\n{'='*60}\n\n{body}\n\n{'='*60}\n[Offline Deterministic Mode]"


def _offline_answer(question: str, context_summary: str) -> str:
    return (
        f"[Offline Deterministic Mode] I cannot answer '{question}' with AI reasoning "
        f"while IBM Granite is offline.\n\n"
        f"Dataset Summary:\n{context_summary}\n\n"
        "Tip: Set WATSONX_APIKEY, WATSONX_PROJECT_ID, WATSONX_URL, and "
        "WATSONX_MODEL_ID in your .env file to enable live AI responses."
    )


# ---------------------------------------------------------------------------
# SHA-256 verification hash
# ---------------------------------------------------------------------------

def _compute_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def explain_fraud_case(findings: dict) -> str:
    """
    Generate a plain-English explanation of why a meter/bill was flagged.

    Parameters
    ----------
    findings : dict — verified Stage-1 JSON output from fraud_engine.analyze_record().
               Must contain: consumer_id, billing_month, risk_score, risk_tier,
               triggered_rules, estimated_revenue_impact, details.

    Returns plain-English explanation string.
    """
    cid     = findings.get("consumer_id", "Unknown")
    month   = findings.get("billing_month", "Unknown")
    rules   = findings.get("triggered_rules", [])
    score   = findings.get("risk_score", 0)
    impact  = findings.get("estimated_revenue_impact", 0)
    tier    = findings.get("risk_tier", "Normal")
    details = findings.get("details", {})

    # Build a tightly scoped, grounded prompt — no fabrication surface
    rules_text = json.dumps(details, indent=2) if details else str(rules)
    prompt = textwrap.dedent(f"""
        You are an electricity billing fraud analyst. Explain the following
        verified fraud detection findings in clear, plain English for a
        field engineer. Do NOT invent any numbers or readings beyond what
        is provided. Be concise (3-5 sentences).

        Consumer ID   : {cid}
        Billing Month : {month}
        Risk Tier     : {tier}
        Risk Score    : {score}/100
        Est. Revenue Impact: Rs.{impact:,.2f}
        Triggered Rules and Evidence:
        {rules_text}

        Explanation:
    """).strip()

    response = _generate(prompt)
    if response:
        return response
    # Graceful offline fallback
    return _offline_explain(findings)


def draft_action_report(findings: dict, report_type: str = "field_inspection") -> dict:
    """
    Generate a downloadable report with a SHA-256 verification hash.

    Parameters
    ----------
    findings    : verified Stage-1 dict from fraud_engine.
    report_type : "field_inspection" (utility DISCOM) |
                  "correction_request" (homeowner)

    Returns
    -------
    {
        "report_text": str,   # full report body
        "sha256_hash": str,   # SHA-256 of report_text
        "timestamp":   str,   # UTC ISO timestamp
        "engine":      str,   # "IBM Granite" | "Offline Deterministic"
        "report_type": str,
    }
    """
    if report_type not in ("field_inspection", "correction_request"):
        raise ValueError(
            f"report_type must be 'field_inspection' or 'correction_request', "
            f"got: {report_type!r}"
        )

    cid    = findings.get("consumer_id", "Unknown")
    month  = findings.get("billing_month", "Unknown")
    rules  = findings.get("triggered_rules", [])
    score  = findings.get("risk_score", 0)
    tier   = findings.get("risk_tier", "Normal")
    impact = findings.get("estimated_revenue_impact", 0)
    ts     = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    if report_type == "field_inspection":
        title    = "FIELD INSPECTION BRIEF"
        audience = "field engineer / DISCOM utility"
        action   = (
            "1. Dispatch field engineer to verify meter seal integrity.\n"
            "2. Cross-check physical meter reading against billing system entry.\n"
            "3. Collect photographic evidence of meter and surrounding premises.\n"
            "4. Submit signed inspection report within 72 hours."
        )
    else:
        title    = "BILLING CORRECTION REQUEST LETTER"
        audience = "billing department / consumer grievance cell"
        action   = (
            "1. Review meter readings for the flagged billing period.\n"
            "2. Issue a corrected bill reflecting accurate consumption.\n"
            "3. Provide written acknowledgment within 7 working days."
        )

    rules_text = json.dumps(findings.get("details", {}), indent=2) if findings.get("details") else str(rules)

    prompt = textwrap.dedent(f"""
        You are an electricity billing compliance officer.
        Draft a professional {title} addressed to the {audience}.
        Use only the verified data below — do NOT add any numbers or
        readings not present here. Format as a formal letter with
        date, subject, body paragraphs, and signature block.

        Consumer ID       : {cid}
        Billing Month     : {month}
        Risk Tier         : {tier}
        Risk Score        : {score}/100
        Est. Revenue Impact: Rs.{impact:,.2f}
        Triggered Flags   : {', '.join(rules) if rules else 'None'}
        Evidence Details  :
        {rules_text}
        Required Actions  :
        {action}
        Report Timestamp  : {ts}

        Begin report:
    """).strip()

    response = _generate(prompt, max_tokens=1000)
    if response:
        report_text = response
        engine      = "IBM Granite"
    else:
        report_text = _offline_report(findings, report_type)
        engine      = "Offline Deterministic"

    # Append metadata footer before hashing
    footer = (
        f"\n\n{'='*60}\n"
        f"Generated     : {ts}\n"
        f"Engine        : {engine}\n"
        f"Consumer ID   : {cid}\n"
        f"Billing Month : {month}\n"
        f"{'='*60}"
    )
    full_text   = report_text + footer
    sha256_hash = _compute_sha256(full_text)

    return {
        "report_text": full_text,
        "sha256_hash": sha256_hash,
        "timestamp":   ts,
        "engine":      engine,
        "report_type": report_type,
    }


def answer_query(question: str, context_summary: str) -> str:
    """
    Conversational Q&A grounded in the active billing dataset / Bill Vault.

    Parameters
    ----------
    question        : user's natural-language question.
    context_summary : pre-computed plain-text summary of the active dataset
                      (never raw CSV; caller must sanitize).

    Returns answer string.
    """
    if not question or not question.strip():
        return "Please enter a question."

    prompt = textwrap.dedent(f"""
        You are an electricity billing fraud analysis assistant.
        Answer the user's question using ONLY the dataset summary provided.
        Do not invent data, readings, consumer names, or amounts not present
        in the summary. If the answer is not in the summary, say so clearly.

        Dataset Summary:
        {context_summary[:3000]}

        User Question: {question.strip()[:500]}

        Answer:
    """).strip()

    response = _generate(prompt, max_tokens=600)
    if response:
        return response
    return _offline_answer(question, context_summary)


# ---------------------------------------------------------------------------
# CLI self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    status = get_engine_status()
    print(f"[granite_agent] Engine status: {status['status']}")
    if status["hint"]:
        print(f"[granite_agent] Hint: {status['hint']}")

    # Minimal test with a synthetic findings dict
    test_findings = {
        "consumer_id":              "CON0042",
        "billing_month":            7,
        "risk_score":               82.5,
        "risk_tier":                "Critical",
        "triggered_rules":          ["meter_to_bill_mismatch", "peer_normalized_drop"],
        "estimated_revenue_impact": 1870.50,
        "details": {
            "rule1": "Metered 310.00 kWh vs billed 120.00 kWh (gap 190.00 kWh, 61.3%)",
            "rule3": "Sudden drop 68.4% vs 6-month baseline (310.0 kWh). Peer zone avg dropped only 3.2%.",
        },
    }

    print("\n--- explain_fraud_case ---")
    explanation = explain_fraud_case(test_findings)
    print(explanation[:500])

    print("\n--- draft_action_report (field_inspection) ---")
    report = draft_action_report(test_findings, "field_inspection")
    print(f"Engine  : {report['engine']}")
    print(f"SHA-256 : {report['sha256_hash']}")
    print(f"Preview : {report['report_text'][:200]}...")

    print("\n--- draft_action_report (correction_request) ---")
    report2 = draft_action_report(test_findings, "correction_request")
    print(f"Engine  : {report2['engine']}")
    print(f"SHA-256 : {report2['sha256_hash']}")

    print("\n--- answer_query ---")
    answer = answer_query(
        "Which zone has the most fraud cases?",
        "Dataset: 300 consumers, 3600 records. Critical: 45, Suspicious: 120, Normal: 3435. "
        "Domestic zone has 25 Critical cases, Commercial 12, Industrial 8.",
    )
    print(answer[:300])

    print("\n[granite_agent] Self-test complete.")
