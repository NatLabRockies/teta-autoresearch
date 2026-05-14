## LLM-in-the-Loop AutoResearch for Vehicle Energy Models: A Small-Team RSE Experiment

## Authors

Nicholas Reinicke | nicholas.reinicke@nlr.gov | Center for Integrated Mobility Science, National Laboratory of the Rockies | <orcid-id>
Robert Fitzgerald | robert.fitzgerald@nlr.gov | Center for Integrated Mobility Science, National Laboratory of the Rockies | 0000-0003-0740-5118

## Keywords

- AutoML
- LLM Agents
- Vehicle Energy Modeling

## Abstract

### 1. Hook & Problem Framing (2–3 sentences)
- RSE teams often own ML-driven scientific software but lack dedicated data scientists.
- Our team maintains RouteE-Powertrain (vehicle energy estimation) used inside RouteE-Compass (energy-aware route planning); model improvements have stagnated for years while engineering work took priority.
- Existing random-forest models trained on vehicle trajectories + FASTSim energy targets are accurate at trip level but inconsistent at link level — the level that actually matters for routing decisions.

### 2. Research Question
- Can an LLM agent, placed in-the-loop of the modeling workflow, act as a stand-in "intern data scientist" to discover feature-engineering and model-architecture improvements that a small RSE team would otherwise not have time to explore?

### 3. Approach — The AutoResearch Loop
- Built a harness (this repo) around RouteE-Powertrain with a fixed dataset and evaluation protocol.
- LLM agent iterates over a search space of features and model families, proposing → training → evaluating → critiquing → revising.
- Implemented as an LLM-driven planner that can make *structural* changes to the experiment — modifying features, model families, and even the experimental design itself — expanding the search beyond our team's prior assumptions.
- After the agent converges on a preferred architecture and feature set, traditional Bayesian / evolutionary optimizers (TPE, CMA-ES, random — `optimizers/`) take over hyperparameter tuning to squeeze out remaining gains under a fixed cost function.
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

## References

these are placeholders to fill + format:
- RouteE-Powertrain — NREL. https://github.com/NREL/routee-powertrain
- RouteE-Compass — NREL. https://github.com/NREL/routee-compass
- FASTSim — NREL. https://www.nrel.gov/transportation/fastsim.html
- Optuna (TPE / CMA-ES samplers used in `optimizers/`). Akiba et al., KDD 2019.
- Any LLM-agent / autoresearch prior work to cite (e.g., AutoML-GPT, MLAgentBench, Sakana AI Scientist) — TBD which to include.

## Connection to Mission, Goals, & Interests of US-RSE Community  

- **Empowering RSEs in scientific software:** demonstrates how RSEs without a dedicated data science role can still drive measurable scientific model improvement by treating an LLM agent as a collaborator inside a well-engineered harness.
- **Reusable RSE patterns:** clear separation of (a) domain code, (b) optimizer/search infrastructure, (c) agent-driven planning — a template other RSE groups can adapt.
- **Reproducibility & rigor:** even with a non-deterministic LLM in the loop, the harness enforces a fixed dataset, fixed evaluation protocol, and logged trial history (see `example/powertrain/tpe/results/`).
- **Community relevance:** speaks to a growing US-RSE conversation about responsible, practical use of LLMs in research workflows — moving past "chat with my code" toward agents that contribute to the scientific result.
- **Open science:** repo, harness, and findings shared so other small teams maintaining domain ML models can replicate the pattern.
