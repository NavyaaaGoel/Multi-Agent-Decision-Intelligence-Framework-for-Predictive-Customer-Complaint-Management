"""Central configuration: paths, SLA policy, teams and decision-theory constants."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
SEED = 42

# ---- SLA policy -----------------------------------------------------------
SEVERITY_LABELS = ("Low", "Medium", "High", "Critical")
SLA_HOURS = {"Critical": 24, "High": 72, "Medium": 168, "Low": 360}  # 1d / 3d / 7d / 15d
ROUTES = ("auto_resolve", "bank_escalation", "partner_escalation")

# ---- Decision-theory constants -------------------------------------------
# Expected loss of a candidate plan = P(breach) * penalty * tier_multiplier + cost
BREACH_PENALTY = {"Critical": 10.0, "High": 6.0, "Medium": 3.0, "Low": 1.5}
TIER_MULTIPLIER = {"Standard": 1.0, "Premium": 1.3, "Priority": 1.6}
FAST_TRACK_BASE_COST = 0.8      # scaled by the assigned team's load
REROUTE_COST = 0.5              # hand-over overhead of moving to the backup team
MIN_INTAKE_CONFIDENCE = 0.40    # below this -> human triage

# ---- Teams ---------------------------------------------------------------
AUTOMATION_TEAM = "Automation Engine"
PARTNER_TEAM = "Partner Desk"
DEFAULT_TEAM = "Branch & Service Ops"

PRODUCT_TEAM = {
    "UPI": "Cards & Payments Ops", "Credit Card": "Cards & Payments Ops",
    "Debit Card": "Cards & Payments Ops", "ATM": "Cards & Payments Ops",
    "Wallet / Prepaid Card": "Cards & Payments Ops",
    "Merchant Services / POS": "Cards & Payments Ops",
    "Internet Banking": "Digital Channels", "Mobile Banking": "Digital Channels",
    "Loans": "Lending Ops", "Agriculture / Priority Sector Loans": "Lending Ops",
    "Savings Account": "Deposits & Accounts", "Current Account": "Deposits & Accounts",
    "Cheque Services": "Deposits & Accounts", "Fixed Deposit": "Deposits & Accounts",
    "Recurring Deposit": "Deposits & Accounts",
    "KYC / AML / Compliance": "Compliance",
    "Insurance": "Treasury & Wealth", "Demat / Trading Account": "Treasury & Wealth",
    "Forex / International Transactions": "Treasury & Wealth",
    "Customer Service": "Branch & Service Ops", "Branch Services": "Branch & Service Ops",
    "Pension / Government Schemes": "Branch & Service Ops", "Lockers": "Branch & Service Ops",
    "General Service Requests": "Branch & Service Ops",
}
BACKUP_TEAM = {
    "Cards & Payments Ops": "Digital Channels", "Digital Channels": "Cards & Payments Ops",
    "Lending Ops": "Deposits & Accounts", "Deposits & Accounts": "Branch & Service Ops",
    "Branch & Service Ops": "Deposits & Accounts", "Fraud & Risk": "Compliance",
    "Compliance": "Fraud & Risk", "Treasury & Wealth": "Deposits & Accounts",
}
FRAUD_PATTERN = re.compile(r"fraud|unauthori[sz]ed|phishing|skimming|stolen", re.I)
BANK_TEAMS = sorted(set(PRODUCT_TEAM.values()) | {"Fraud & Risk"})
ALL_TEAMS = BANK_TEAMS + [AUTOMATION_TEAM, PARTNER_TEAM]
TEAM_CAPACITY_CASES = 25  # open cases that correspond to load = 1.0 in the live provider

CHANNELS = ("Web", "Email", "Chatbot", "Voice", "Branch", "OCR")
TIERS = ("Standard", "Premium", "Priority")
