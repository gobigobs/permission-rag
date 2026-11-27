---
id: eng-deploy-runbook
title: Deploy Runbook — Shipping to Production
source: local-filesystem
acl: [engineering]
---

# Deploy Runbook — Shipping to Production

Deploys happen through the CI pipeline only — never deploy from a laptop.
Merge to `main`, wait for the green build, then click **Promote to prod** in
the deploy dashboard. The rollout is canary-first: 5% of traffic for 10
minutes, then 50%, then 100%.

If error rates rise above 1% during the canary, the pipeline rolls back
automatically. To roll back manually, click **Rollback** in the dashboard and
pick the previous green build. The on-call engineer is paged automatically
if the canary fails twice in a row.

Every deploy posts a summary to #eng-deploys with the build number, the
author, and the rollback link.
