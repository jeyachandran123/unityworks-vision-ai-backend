# product-analyst — Vision AI backend brief

## Evidence to read
`docs/architecture/NOT_YET_CONNECTED.md`, `app/api/capability.py`, `app/api/analytics.py`,
`app/api/integrations.py`, `app/api/patron.py`, `app/domain/modules.py`,
`app/authorization/model.py` (roles → permissions), `README.md`. `docs/architecture/FINAL_*.md` are
decisions; phase reports are history. The frontend's brief covers navigation and pages.

## What breaks here
- A module route answering `available: true` with no data source; seven modules are deliberately
  unconnected.
- `NOT_YET_CONNECTED.md` requirements differing from the wording the module's route serves (one source).
- A permission defined but used on no route; `REGISTER_DEMAND` wired to anything.
- Role intent: `kitchen_supervisor` reaching evidence; `auditor` reaching live views.
- `README.md` claiming product routes do not exist (it predates them).
- `awaiting` vs `blocked` conflated.

## How to prove it
List routes by importing the app with `.venv/Scripts/python.exe`; grep the module catalogue.
