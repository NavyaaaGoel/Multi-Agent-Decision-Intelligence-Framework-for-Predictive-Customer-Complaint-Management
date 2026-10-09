"""FastAPI wrapper (optional). Run:  uvicorn cdi.api.app:app --reload   ->  http://localhost:8000/docs

NOTE: written against FastAPI >= 0.100 / Pydantic v2. All business logic lives in
cdi.service.ComplaintService, which is what the unit tests exercise.
"""
from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from cdi.service import ComplaintService

app = FastAPI(title="Complaint Decision Intelligence", version="1.0.0")
_svc: Optional[ComplaintService] = None


def svc() -> ComplaintService:
    global _svc
    if _svc is None:
        _svc = ComplaintService.from_artifacts("complaints.db")
    return _svc


class ComplaintIn(BaseModel):
    text: str = Field(min_length=3)
    customer_id: str = "ANON"
    channel: str = "Web"
    tier: str = "Standard"
    submitted_at: Optional[str] = None
    amount: Optional[float] = None


class ScanIn(BaseModel):
    now: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/complaints")
def submit(c: ComplaintIn):
    try:
        return svc().submit(c.model_dump())
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/complaints/{cid}")
def get(cid: str):
    d = svc().get(cid)
    if d is None:
        raise HTTPException(404, "not found")
    return d


@app.get("/complaints/{cid}/trace")
def trace(cid: str):
    d = svc().trace(cid)
    if d is None:
        raise HTTPException(404, "not found")
    return d


@app.post("/sla/scan")
def scan(body: ScanIn = ScanIn()):
    return {"events": svc().scan_sla(body.now)}


@app.get("/teams/load")
def teams():
    return svc().team_load()


@app.get("/metrics")
def metrics():
    return svc().metrics() or {}
