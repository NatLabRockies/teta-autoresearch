# US-RSE RouteE Talk 

## Title

Human-Agent Synergy in AutoResearch: An RSE Experiment Accelerating Vehicle Energy Model Discovery

## Authors

Nicholas Reinicke \<nicholas.reinicke@nlr.gov\>, Center for Integrated Mobility Sciences, National Renewable Energy Laboratory, 0000-0003-3763-3031

Robert Fitzgerald \<robert.fitzgerald@nlr.gov\>, Center for Integrated Mobility Sciences, National Renewable Energy Laboratory, 0000-0003-0740-5118

## Keywords

AutoResearch, LLM Agents, Human-In-The-Loop, Transportation Engineering

## Abstract

Research Software Engineers (RSEs) increasingly maintain machine learning (ML) components embedded in larger systems whose evolving requirements demand structural model improvements. Our team maintains RouteE-Powertrain [1], a vehicle energy estimation library consumed by RouteE-Compass [2] for energy-aware route planning. The existing random-forest models, trained on vehicle trajectories with FASTSim [3] energy targets, are accurate at the _trip_ level but fail to capture certain link-level energy dynamics that are important in route finding applications. RSEs often have the domain expertise and modeling intuitions needed to direct structural improvement, but the per-experiment cost of designing, implementing, training, and evaluating each variant makes running the hundreds of trials needed prohibitive.

We ask: _when an RSE team supplies domain expertise and search direction, can an LLM-driven AutoResearch loop accelerate structural ML improvements by absorbing the per-experiment context-switching cost — and does the agent's contribution survive when that human direction is removed?_

To answer this, we built an AutoResearch harness (forked from Andrej Karpathy's `autoresearch` [4]) around RouteE-Powertrain with a fixed dataset and evaluation protocol. RSEs contribute scientific direction through a per-session `seed.md` which represents a hypothesis the agent will pursue (e.g., "geometry features may matter for link-level energy", "survey state-of-the-art architectures for link sequence prediction"). In addition, a persistent `domain.md` fixes domain constraints (e.g., bounds on negative energy from BEV regeneration, the scope of valid link states). The LLM agent iterates a fixed loop of: propose → train → evaluate → critique → revise. In this way, the agent performs the per-experiment work that is expensive in human time: coding feature pipelines, wiring up new model families, training under a fixed 10-minute budget per trial, and reading metrics. Across hundreds of such trials, the agent expands the search well beyond what the team could exercise manually in the same time budget. 

The most informative findings came from this division of labor. After we seeded the agent with context about road geometry, it identified _sinuosity_ (link length divided by straight-line distance) as a feature our team had not previously tried. After we asked it to survey state of the art sequence models, it converged on a 1D CNN as the best performer, capturing the sequence structure of link observations. The resulting model preserved trip-level accuracy while substantially improving link-level RMSE accuracy by over 50%. In a parallel run where we withheld the human seed context and let the agent explore freely under the same time budget, it did not converge on these improvements, providing some evidence that the directional context, not the agent acting alone, is an important component for this type of problem.

The major lesson that this seems to point to is that the value isn't exclusively the agent autonomy but rather, it's a synergy in which the RSE supplies hypotheses, domain constraints, and review, while the agent absorbs the prohibitive context-switching cost of running hundreds of short, structurally varied experiments. The agent expands the feasibility frontier of the team's existing scientific judgment; it does not replace it. We argue this is a transferable pattern: a fixed domain harness with an LLM agent, where the RSE owns the human aspects like scientific direction and evaluation rigor. Future work extends the framework across powertrain classes (i.e. Conventional Vehicles and Plug-in Hybrid Electric Vehicles), integrates classical hyperparameter optimizers downstream of the agent's architectural choices for fine tuning, and deploys these new RouteE-Powertrain models into RouteE-Compass.

## References

1. _RouteE-Powertrain._ National Renewable Energy Laboratory. https://github.com/NatLabRockies/routee-powertrain
2. _RouteE-Compass._ National Renewable Energy Laboratory. https://github.com/NatLabRockies/routee-compass
3. _FASTSim: Future Automotive Systems Technology Simulator._ National Renewable Energy Laboratory. https://www.nlr.gov/transportation/fastsim.html
4. Karpathy, A. _autoresearch._ https://github.com/karpathy/autoresearch

## Connection to Mission, Goals, & Interests of US-RSE Community

This work speaks directly to the US-RSE community's interest in how RSEs can responsibly and practically apply LLMs inside research workflows. It moves beyond "autonomous experimentation" toward an honest model of human–agent collaboration where each side contributes what it does best. We demonstrate a pattern in which the RSE team retains scientific direction and evaluation authority while the LLM agent absorbs the per-experiment engineering burden that would otherwise gate structural ML improvements. The architecture cleanly separates domain knowledge and agent-driven planning, giving other RSE groups a reusable template they can adapt to scale structural exploration of their own domain specific ML models. Reproducibility and rigor are preserved despite a non-deterministic LLM in the loop: the harness enforces a fixed dataset, a fixed evaluation protocol, and a logged trial history. Finally, in keeping with open-science values, the harness and findings are shared so that other teams navigating the challenges of maintaining domain ML software can replicate the pattern and adapt it to their own systems.
