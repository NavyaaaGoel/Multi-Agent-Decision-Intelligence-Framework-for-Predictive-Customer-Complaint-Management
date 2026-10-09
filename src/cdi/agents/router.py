from __future__ import annotations

from datetime import timedelta

from cdi import config
from cdi.agents.base import Agent


class ResourceRouterAgent(Agent):
    """Assigns the owning team (primary or backup) and the post-intervention ETA."""
    name = "resource_router"

    def run(self, state, ctx):
        state.assigned_team = state.intervention.get("reroute_to") or state.team
        state.expected_completion = state.submitted_at + timedelta(hours=state.final_risk["median_hours"])
        return {"assigned_team": state.assigned_team, "partner": state.partner or None,
                "expected_completion": state.expected_completion.isoformat(),
                "load_at_assignment": round(ctx.load.load(state.assigned_team, state.submitted_at), 3)}
