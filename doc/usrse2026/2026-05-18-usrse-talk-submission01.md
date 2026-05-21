# US-RSE RouteE Paper

## Title 

Human-Guided AutoResearch for Vehicle Energy Models: An Experiment with LLM Agents

## Authors

Nicholas Reinicke \<nicholas.reinicke@nrel.gov\>, Center for Integrated Mobility Sciences, National Renewable Energy Laboratory

Robert Fitzgerald \<robert.fitzgerald@nrel.gov\>, Center for Integrated Mobility Sciences, National Renewable Energy Laboratory, 0000-0003-0740-5118

## Keywords

AutoML, LLM Agents, Transportation Engineering

## Abstract

Research software engineering (RSE) teams often own machine-learning-driven scientific software without having a dedicated ML architecture specialist on staff. Our team maintains RouteE-Powertrain [1], a vehicle energy estimation library consumed by RouteE-Compass [2] for energy-aware route planning. Existing random-forest models, trained on vehicle trajectories with FASTSim [3] energy targets, are accurate at the *trip* level but fail to capture the link-level energy dynamics that explicit route search now demands. Advancing link-level accuracy requires structural changes to features and model families — work that had stalled given the team's capacity.

We ask: *Can a human-guided AutoResearch workflow, steering an LLM agent with domain constraints, reliably surface ML architecture improvements that would otherwise exceed a small RSE team's bandwidth?*

To answer this, we built an AutoResearch harness (forked from Andrej Karpathy's `autoresearch` [4]) around RouteE-Powertrain with a fixed dataset and evaluation protocol, emphasizing human-in-the-loop design. RSEs guide the agent via a session "seed" highlighting domain insights (e.g., link geometry) and a `domain.md` file that makes domain constraints explicit (e.g., bounds on negative energy from BEV regeneration, the scope of valid link states). The LLM planner iterates: propose -> train -> evaluate -> critique -> revise, and is allowed to make *structural* changes — new features, new model families, and even adjustments to experimental design — expanding the search beyond our team's prior assumptions. Once the agent converges on a preferred architecture, traditional Bayesian and evolutionary optimizers (TPE, CMA-ES, random; via Optuna [5]) take over hyperparameter tuning under a fixed cost function. Throughout, we treat reproducibility, token-budget accounting, and decision logging as first-class RSE concerns, and we keep domain code (`domains/routee/`) cleanly separated from optimizer and agent infrastructure.

The agent surfaced ideas the team had not previously pursued: history features summarizing the previous five links along the trajectory, road-geometry **sinuosity** as a predictive feature, and sequential deep-learning architectures — with a **1D CNN** emerging as the best performer by capturing the sequence structure of link observations. The resulting model preserved trip-level accuracy while **substantially improving link-level accuracy** over the production random-forest baseline.

The lesson: an LLM-driven autoresearch loop can be a force-multiplier for small RSE teams, unblocking stalled scientific-modeling work without hiring a dedicated ML specialist. Crucially, the agent's value was not just hyperparameter search — it proposed structural and feature-space changes outside our prior framing. We argue this is a transferable pattern: a domain-fixed harness, pluggable optimizers, and an LLM planner, with the RSE team owning evaluation rigor and reproducibility. Future work extends the loop across powertrain classes (ICEV / HEV / BEV / PHEV), analyzes trends in agent-discovered features across vehicle types, and deploys next-generation RouteE-Powertrain models into RouteE-Compass.

## References

1. *RouteE-Powertrain.* National Renewable Energy Laboratory. https://github.com/NatLabRockies/routee-powertrain
2. *RouteE-Compass.* National Renewable Energy Laboratory. https://github.com/NatLabRockies/routee-compass
3. *FASTSim: Future Automotive Systems Technology Simulator.* National Renewable Energy Laboratory. https://www.nlr.gov/transportation/fastsim.html
4. Karpathy, A. *autoresearch.* https://github.com/karpathy/autoresearch
5. Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. *Optuna: A Next-generation Hyperparameter Optimization Framework.* KDD 2019. https://optuna.org

## Connection to Mission, Goals, & Interests of US-RSE Community

This work speaks directly to the US-RSE community's interest in how RSEs can responsibly and practically apply LLMs inside research workflows — moving beyond "chat with my code" toward agents that contribute to the scientific result. It demonstrates how a small RSE team without a dedicated data science role can still drive measurable scientific model improvement by treating an LLM agent as a collaborator inside a well-engineered harness. The architecture cleanly separates (a) domain code, (b) optimizer and search infrastructure, and (c) agent-driven planning, giving other RSE groups a reusable template they can adapt to their own domain models. Reproducibility and rigor are preserved despite a non-deterministic LLM in the loop: the harness enforces a fixed dataset, a fixed evaluation protocol, and a logged trial history. Finally, in keeping with open-science values, the harness and findings are shared so that other small teams maintaining domain ML software can replicate the pattern and adapt it to their own scientific contexts.

# Appendix

## Alternative Titles

Option 1 (Direct & Balanced)

> Human-Guided AutoResearch for Vehicle Energy Models: An RSE Experiment with LLM Agents

Option 2 (Focus on Framework Expansion & Domain)

> Expanding AutoResearch with Domain Oversight: A Human-in-the-Loop LLM Experiment

Option 3 (Action-Oriented)

> Steering AutoResearch with Domain Expertise: Human-in-the-Loop ML for Vehicle Energy Models

Option 4 (Emphasizing the Small Team Benefit)

> Human-in-the-Loop AutoResearch: Empowering Small RSE Teams with Domain-Guided LLM Agents

## Alternative Hypotheses

These options can replace the second paragraph of the abstract depending on our tone/objective:

Option 1 (Directly integrates the new framing)

> We ask: Can an LLM agent—guided by a human-in-the-loop providing strict domain context—expand an AutoResearch workflow to discover feature-engineering and model-architecture improvements that a small RSE team lacks the capacity to explore?

Option 2 (Focuses on the workflow rather than the agent)

> We ask: Can a human-guided AutoResearch workflow, steering an LLM agent with domain constraints, reliably surface ML architecture improvements that would otherwise exceed a small RSE team's bandwidth?

Option 3 (Slightly punchier / more concise)

> We ask: When guided by RSE domain oversight, can an LLM agent expand an AutoResearch loop to discover structural ML improvements that a small team otherwise has no time to pursue?
