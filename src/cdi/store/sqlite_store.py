"""SQLite persistence: complaints, immutable audit log. Use ':memory:' for tests / simulation."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta

CLOSED = ("Resolved", "Closed")


class Store:
    def __init__(self, path: str = ":memory:"):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS complaints(
            complaint_id TEXT PRIMARY KEY, customer_id TEXT, submitted_at TEXT, team TEXT,
            status TEXT, sla_status TEXT, escalation_level TEXT, deadline TEXT, payload TEXT);
        CREATE TABLE IF NOT EXISTS audit_log(
            id INTEGER PRIMARY KEY AUTOINCREMENT, complaint_id TEXT, ts TEXT, actor TEXT,
            action TEXT, details TEXT);
        CREATE INDEX IF NOT EXISTS ix_c_customer ON complaints(customer_id, submitted_at);
        CREATE INDEX IF NOT EXISTS ix_a_complaint ON audit_log(complaint_id);
        """)

    # ---- complaints
    def upsert(self, d: dict):
        self.db.execute(
            "INSERT OR REPLACE INTO complaints VALUES (?,?,?,?,?,?,?,?,?)",
            (d["complaint_id"], d["customer_id"], d["submitted_at"], d.get("assigned_team") or d.get("team"),
             d["status"], d.get("sla_status"), d.get("escalation_level"), d.get("deadline"), json.dumps(d)))
        self.db.commit()

    def get(self, complaint_id: str):
        r = self.db.execute("SELECT payload FROM complaints WHERE complaint_id=?", (complaint_id,)).fetchone()
        return json.loads(r["payload"]) if r else None

    def list(self, limit: int = 100):
        rows = self.db.execute("SELECT payload FROM complaints ORDER BY submitted_at DESC LIMIT ?", (limit,))
        return [json.loads(r["payload"]) for r in rows]

    def open_complaints(self):
        q = "SELECT payload FROM complaints WHERE status NOT IN (%s)" % ",".join("?" * len(CLOSED))
        return [json.loads(r["payload"]) for r in self.db.execute(q, CLOSED)]

    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) c FROM complaints").fetchone()["c"]

    def recent_count(self, customer_id: str, before: datetime, days: int = 30) -> int:
        lo = (before - timedelta(days=days)).isoformat()
        r = self.db.execute("SELECT COUNT(*) c FROM complaints WHERE customer_id=? AND submitted_at>=? AND submitted_at<?",
                            (customer_id, lo, before.isoformat())).fetchone()
        return r["c"]

    def team_open_count(self, team: str) -> int:
        q = "SELECT COUNT(*) c FROM complaints WHERE team=? AND status NOT IN (%s)" % ",".join("?" * len(CLOSED))
        return self.db.execute(q, (team, *CLOSED)).fetchone()["c"]

    # ---- audit log (append-only)
    def log_action(self, complaint_id: str, ts, actor: str, action: str, details: dict | None = None):
        ts = ts.isoformat() if isinstance(ts, datetime) else str(ts)
        self.db.execute("INSERT INTO audit_log(complaint_id,ts,actor,action,details) VALUES (?,?,?,?,?)",
                        (complaint_id, ts, actor, action, json.dumps(details or {})))
        self.db.commit()

    def audit(self, complaint_id: str):
        rows = self.db.execute("SELECT ts,actor,action,details FROM audit_log WHERE complaint_id=? ORDER BY id",
                               (complaint_id,))
        return [{"ts": r["ts"], "actor": r["actor"], "action": r["action"], "details": json.loads(r["details"])}
                for r in rows]

    def audit_count(self, complaint_id: str) -> int:
        return self.db.execute("SELECT COUNT(*) c FROM audit_log WHERE complaint_id=?", (complaint_id,)).fetchone()["c"]
