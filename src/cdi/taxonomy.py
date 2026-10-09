"""Issue taxonomy: wraps the hackathon ACTION_LOOKUP and enriches it with severity & teams."""
from __future__ import annotations

import re
from dataclasses import dataclass

from cdi import config, severity
from cdi.data.action_lookup import ACTION_LOOKUP


@dataclass(frozen=True)
class IssueSpec:
    product: str
    issue: str
    key: str                 # "product::issue" (unique; issue names repeat across products)
    route: str
    actions: tuple
    base_severity: int
    team: str
    backup_team: str         # "" if rerouting is not possible
    partner: str             # "" unless route == partner_escalation


def _partner_from(actions) -> str:
    for a in actions:
        m = re.match(r"forward_to_(\w+)", a)
        if m:
            return {"npci": "NPCI"}.get(m.group(1), m.group(1).replace("_", " ").title())
    return "Partner"


def _build() -> dict:
    out = {}
    for product, issues in ACTION_LOOKUP.items():
        for issue, entry in issues.items():
            route, actions = entry["decision"], tuple(entry["actions"])
            if route == "auto_resolve":
                team, backup = config.AUTOMATION_TEAM, ""
            elif route == "partner_escalation":
                team, backup = config.PARTNER_TEAM, ""
            else:
                team = ("Fraud & Risk" if config.FRAUD_PATTERN.search(issue)
                        else config.PRODUCT_TEAM.get(product, config.DEFAULT_TEAM))
                backup = config.BACKUP_TEAM.get(team, "")
            out[f"{product}::{issue}"] = IssueSpec(
                product, issue, f"{product}::{issue}", route, actions,
                severity.base_severity(issue), team, backup,
                _partner_from(actions) if route == "partner_escalation" else "")
    return out


TAXONOMY = _build()
PRODUCTS = sorted({s.product for s in TAXONOMY.values()})


def lookup(product: str, issue: str) -> IssueSpec | None:
    return TAXONOMY.get(f"{product}::{issue}")


def issues_of(product: str) -> list:
    return [s for s in TAXONOMY.values() if s.product == product]
