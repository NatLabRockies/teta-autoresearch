# USRSE'26 Abstract Outline — RouteE AutoResearch

## Working Title
- **Primary:** "LLM-in-the-Loop AutoResearch for Vehicle Energy Models: A Small-Team RSE Experiment"
- **Alternatives:**
  - "When the Team Has No Data Scientist: Using LLM Agents to Drive ML Model Discovery for RouteE-Powertrain"
  - "Autoresearch with LLMs: Improving Link-Level Accuracy in Vehicle Energy Prediction"

## Authors (to confirm)
- Rob Fitzgerald <rob.fitzgerald@nrel.gov>, Computational Science Center, National Renewable Energy Laboratory, ORCID: TBD
- Co-authors from the RouteE / RouteE-Compass team (TBD)

## Keywords (need ≥3)
- LLM agents / autonomous research
- AutoML / metaheuristic search
- Vehicle energy modeling (RouteE-Powertrain)
- Research software engineering for small teams
- Link-level prediction accuracy

---

## Abstract Outline (~1 page)

### 1. Hook & Problem Framing (2–3 sentences)
- RSE teams often own ML-driven scientific software but lack dedicated data scientists.
- Our team maintains RouteE-Powertrain (vehicle energy estimation) used inside RouteE-Compass (energy-aware route planning); model improvements have stagnated for years while engineering work took priority.
- Existing random-forest models trained on vehicle trajectories + FASTSim energy targets are accurate at trip level but inconsistent at link level — the level that actually matters for routing decisions.

### 2. Research Question
- Can an LLM agent, placed in-the-loop of the modeling workflow, act as a stand-in "intern data scientist" to discover feature-engineering and model-architecture improvements that a small RSE team would otherwise not have time to explore?

### 3. Approach — The AutoResearch Loop
- Built a harness (this repo) around RouteE-Powertrain with a fixed dataset and evaluation protocol.
- LLM agent iterates over a search space of features and model families, proposing → training → evaluating → critiquing → revising.
- Implemented as a loose metaheuristic search (TPE / CMA-ES / random samplers in `optimizers/`) wrapped with an LLM-driven planner that can also make *structural* changes to the experiment — expanding the search beyond our team's prior assumptions.
- Briefly note RSE concerns: reproducibility of agent runs, cost/token budgeting, logging of decisions, separation of domain code (`domains/routee/`) from optimizer/agent infrastructure.

### 4. Findings
- The agent rediscovered and surfaced ideas the team had not previously used:
  - History features from the previous 5 links along the trajectory.
  - Road-geometry **sinuosity** as a predictive feature.
  - Sequential deep-learning architectures, with a **1D CNN** emerging as the best performer (captures sequence structure of link observations).
- Result: trip-level accuracy maintained, **link-level accuracy substantially improved** vs. the production random-forest baseline.

### 5. Significance / "What's the lesson?"
- An LLM-driven autoresearch loop can act as a force-multiplier for small RSE teams, unblocking stalled scientific-modeling work without hiring a dedicated ML specialist.
- The agent's value was not just hyperparameter tuning — it proposed structural and feature-space changes outside our prior framing.
- Transferable pattern: domain-fixed harness + pluggable optimizers + LLM planner, with the RSE team owning evaluation rigor and reproducibility.

### 6. Future Work (1–2 sentences)
- Extend the loop across powertrain classes (ICEV / HEV / BEV / PHEV) to test whether each needs distinct treatments.
- Analyze trends in agent-discovered features across vehicle types.
- Deploy next-generation RouteE-Powertrain models into RouteE-Compass.

---

## Connection to US-RSE Mission, Goals & Interests (<300 words)
Talking points to assemble:
- **Empowering RSEs in scientific software:** demonstrates how RSEs without a dedicated data-science role can still drive measurable scientific-model improvement by treating an LLM agent as a collaborator inside a well-engineered harness.
- **Reusable RSE patterns:** clear separation of (a) domain code, (b) optimizer/search infrastructure, (c) agent-driven planning — a template other RSE groups can adapt.
- **Reproducibility & rigor:** even with a non-deterministic LLM in the loop, the harness enforces a fixed dataset, fixed evaluation protocol, and logged trial history (see `example/powertrain/tpe/results/`).
- **Community relevance:** speaks to a growing US-RSE conversation about responsible, practical use of LLMs in research workflows — moving past "chat with my code" toward agents that contribute to the scientific result.
- **Open science:** repo, harness, and findings shared so other small teams maintaining domain ML models can replicate the pattern.

---

## References (placeholders to fill)
- RouteE-Powertrain — NREL. https://github.com/NREL/routee-powertrain
- RouteE-Compass — NREL. https://github.com/NREL/routee-compass
- FASTSim — NREL. https://www.nrel.gov/transportation/fastsim.html
- Optuna (TPE / CMA-ES samplers used in `optimizers/`). Akiba et al., KDD 2019.
- Any LLM-agent / autoresearch prior work to cite (e.g., AutoML-GPT, MLAgentBench, Sakana AI Scientist) — TBD which to include.

---

## Open Questions for the Author Before Drafting
1. Final author list, affiliations, and ORCIDs.
2. Concrete accuracy numbers (RF baseline vs. CNN-best) — trip-level and link-level — to put in the abstract.
3. Which LLM(s) / agent framework were used; do we want to name them in the abstract?
4. Are we comfortable framing this as "autoresearch" vs. "LLM-assisted AutoML"? (affects keywords)
5. Any IP / clearance constraints on naming the discovered features or sharing the harness publicly by submission time?
