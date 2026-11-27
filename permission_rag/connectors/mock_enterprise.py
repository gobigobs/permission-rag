"""Mock enterprise connector (Drive / Notion-style API).

Stands in for a real enterprise source (Google Drive, Notion, Confluence)
that a production deployment would call over HTTP with OAuth. The mock
keeps the same shape a real connector would have — a client object with a
``list_documents()`` API call returning id / title / body / acl per doc —
so swapping in the real API means replacing one class, not the pipeline.

Seeded docs deliberately include an exec-only compensation document: it is
the denial target in the demo and the ACL test.
"""

from __future__ import annotations

from ..acl import Document
from . import Connector


class MockEnterpriseAPI:
    """Fake Drive/Notion-style API: list_documents() returns seeded docs."""

    def list_documents(self) -> list:
        return [
            {
                "id": "exec-strategy-2026",
                "title": "Exec Strategy 2026 — Growth Plan (CONFIDENTIAL)",
                "body": (
                    "The 2026 growth plan targets Rs 5 crore in revenue, up from "
                    "Rs 3.1 crore in 2025. The plan has three bets: expand the "
                    "Salem shop to a second location, launch a catering line for "
                    "corporate clients, and grow the online ordering share to 40% "
                    "of sales. Hiring is frozen in Q1 except for two senior "
                    "engineering roles. The board reviews progress quarterly."
                ),
                "acl": ["exec"],
            },
            {
                "id": "exec-comp-review",
                "title": "Executive Compensation Review 2026 (CONFIDENTIAL)",
                "body": (
                    "Executive compensation bands for 2026: senior engineers "
                    "Rs 18–24 lakh, engineering managers Rs 30–40 lakh, directors "
                    "Rs 55–70 lakh, and the CEO band Rs 90 lakh–1.2 crore. Annual "
                    "bonuses are 10–20% of base, paid in March. Equity grants vest "
                    "over four years with a one-year cliff. These bands are "
                    "confidential and must not be shared outside the exec team."
                ),
                "acl": ["exec"],
            },
            {
                "id": "fin-vendor-contracts",
                "title": "Vendor Contracts — Renewals and Approval Limits",
                "body": (
                    "Vendor contracts renew every April. Contracts above Rs 2 lakh "
                    "a year need finance approval before signing, and contracts "
                    "above Rs 10 lakh a year need the CFO's sign-off. Current key "
                    "vendors: Varagi Trades for bulk supplies, Savitha Mart for "
                    "packaging, and All Food for frozen goods. Renegotiate the "
                    "Varagi contract before April — prices rose 8% last year. "
                    "Expense claims above Rs 10,000 need finance approval."
                ),
                "acl": ["finance"],
            },
            {
                "id": "eng-architecture",
                "title": "System Architecture — Services and Data Flow",
                "body": (
                    "The platform has three services: the web app, the API, and "
                    "the worker that processes orders. The API talks to Postgres "
                    "for orders and Redis for the session cache. All services run "
                    "in Docker containers behind a load balancer. Deploys go "
                    "through the CI pipeline with a canary rollout: 5% of traffic "
                    "for 10 minutes, then 50%, then 100%. Logs go to the central "
                    "logging service and are kept for 30 days."
                ),
                "acl": ["engineering"],
            },
            {
                "id": "hr-performance-review",
                "title": "Performance Review Process — Ratings and Timeline",
                "body": (
                    "Performance reviews run twice a year, in April and October. "
                    "Managers write a review, employees add a self-review, and "
                    "ratings are calibrated in a meeting with HR. Ratings are: "
                    "Exceeds, Meets, and Needs Improvement. A Needs Improvement "
                    "rating triggers a 3-month improvement plan with monthly "
                    "check-ins. Promotion decisions are made in the April cycle "
                    "only. Review scores are confidential between the employee, "
                    "their manager, and HR."
                ),
                "acl": ["hr"],
            },
            {
                "id": "all-remote-work-policy",
                "title": "Remote Work Policy — Hybrid Working Rules",
                "body": (
                    "The company works hybrid: two days a week in the office, "
                    "three days remote. Teams pick their own office days, but "
                    "everyone is in on Wednesdays for the all-hands meeting. "
                    "Remote work needs a stable internet connection and a quiet "
                    "workspace. Equipment — laptop, monitor, headset — is provided "
                    "by the company. Working from outside India needs HR approval "
                    "two weeks in advance."
                ),
                "acl": ["all"],
            },
        ]


class MockEnterpriseConnector(Connector):
    """Connector over the mock enterprise API (Drive/Notion-shaped)."""

    name = "mock-enterprise"

    def __init__(self, api=None):
        self.api = api or MockEnterpriseAPI()

    def fetch(self) -> list:
        docs = []
        for item in self.api.list_documents():
            if "acl" not in item or not item["acl"]:
                continue  # fail closed: no ACL, no ingest
            docs.append(Document(
                doc_id=item["id"],
                title=item["title"],
                text=item["body"],
                source=self.name,
                acl=item["acl"],
            ))
        return docs
