# Screenshots — capture guide

**Status: not captured.** Screenshot capture requires a displayed browser window, which the build
environment could not provide. Rather than leave this as an untracked gap, this file lists exactly
what to capture and the state to capture it in. Week 3's retrospective (§7.9) records the same tool
limitation, so it is documented rather than rediscovered.

Start the app:

```bash
streamlit run app/main.py
```

Capture at **1440×900**, dark theme (the default), and save with the filenames below — the README and
the demo video both refer to them.

| # | Filename | What to show | How to get that state |
|---|---|---|---|
| 1 | `01_console_idle.png` | The landing state: sidebar request panel, run options, live provider status, budget meter at 0/50 with the caps line | Open the app. Nothing to click. |
| 2 | `02_plan_checkpoint.png` | **Checkpoint 1** — the research plan awaiting approval, before anything expensive has run | Leave *Human checkpoints* ticked, pick the LangGraph vs CrewAI sample, press **Run workflow**, wait for the pause |
| 3 | `03_plan_edit.png` | The operator editing a task in the plan, showing a human can change it | At checkpoint 1, edit one task's research question |
| 4 | `04_parallel_research.png` | Per-agent live status with several researchers running at once | Approve the plan; capture while the fan-out is in flight |
| 5 | `05_critic_revision.png` | The Critic's objections and the revision counter showing a cycle was used | Wait for the Critic to reject once |
| 6 | `06_report_checkpoint.png` | **Checkpoint 2** — the full report shown for review, evidence section separate from the recommendation | Continue to the end of the run |
| 7 | `07_evidence_panel.png` | Stored evidence with source, reliability and the quote that was verified against the document | Expand the evidence section of the report |
| 8 | `08_budget_and_usage.png` | Per-agent token and call usage after a complete run | Sidebar, after the run finishes |
| 9 | `09_light_mode.png` | The same console in light theme | Toggle **🌙 Dark Mode** off |
| 10 | `10_permission_error.png` | A tool permission refusal surfaced in the trace | Open the run trace and find a `permission_denied` entry |

## Cost warning

Screenshots 2–8 require a **real workflow run**, which costs roughly 100k tokens. The free-tier
ceiling is tokens *per day* and allows about nine full runs, so capture all of them in a single run
rather than restarting between shots. Screenshots 1 and 9 cost nothing — the app makes no model call
until **Run workflow** is pressed.

## After capturing

Nothing references these files yet, so add them where they carry weight — the human-checkpoint and
evaluation sections of the [README](../README.md) are the natural homes for 2, 6 and 7.
