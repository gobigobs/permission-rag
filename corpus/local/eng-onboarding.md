---
id: eng-onboarding
title: Engineering Onboarding — Dev Environment Setup
source: local-filesystem
acl: [engineering]
---

# Engineering Onboarding — Dev Environment Setup

New engineers set up their laptop on day one. Install Python 3.12, Node 20,
and Docker Desktop. Clone the monorepo from the internal Git server and run
`make bootstrap`, which installs dependencies and creates the local database.

Every service runs locally with `docker compose up`. The API listens on
port 8000 and the web app on port 3000. Run the test suite with `make test`;
all tests must pass before your first pull request.

Ask in the #eng-help channel if setup fails. Your onboarding buddy reviews
your first pull request within one working day.
