# US-RSE RouteE Talk

## Title

Human-Agent Synergy in AutoResearch: An RSE Experiment Accelerating Vehicle Energy Model Discovery

## Authors

Nicholas Reinicke \<nicholas.reinicke@nlr.gov\>, Center for Integrated Mobility Sciences, National Renewable Energy Laboratory, 0000-0003-3763-3031

Robert Fitzgerald \<robert.fitzgerald@nlr.gov\>, Center for Integrated Mobility Sciences, National Renewable Energy Laboratory, 0000-0003-0740-5118

## Keywords

AutoResearch, LLM Agents, Human-In-The-Loop, Transportation Engineering

## Abstract

Research Software Engineers (RSEs) maintain complex software and often work with machine learning (ML) models integrated into their systems. At the same time, advances in ML architectures and available data are improving rapidly and these models may have significant room for improvement. Our RSE team maintains the RouteE-Powertrain [1] software which predicts vehicle energy consumption at a mesoscopic level and is consumed by RouteE-Compass [2] which conducts shortest path searches with energy context. Our existing energy models have used a random forest based architecture which is accurate at the trip level but failes to capture certain link-level energy dynamics which are important in route finding applications. While RSEs might have the domain expertise and modeling intuitions needed to improve their models, the overall time and attention cost of designing, implementing, training, and evaluating model improvements makes running extensive trials difficult when there are already significant demands on the RSE's time.

To address this challenge, and with the recent advances in agentic large language model (LLM) based agents, we proposed a hypothesis: When the RSE can supply the domain expertise and the search direction, can an LLM agent help accelerate structural model improvements by absorbing certain elements that are difficult for a human, like context switching and limited time budgets? In addition, would the agent's contributions still survive if we took away the human direction and oversight?

To answer this, we expanded an existing auto research harness (forked from Andrej Karpathy's `autoresearch` [4]) to include elements that were specific to our domain. In this expanded harness, the RSE can provide specific scientific direction through a per-session `seed.md` file that represents a specific hypotheisis the agent should persue (e.g., "geometry features likely impact link level energy", "survey state-of-the-art architectures for link sequence prediction"). In addition, we added a fixed `domain.md` that includes domain specific knowledge and constraints that guide the agent towards solutions that are practical in our domain. For example, Battery Electric Vehicles (BEVs) have regenerative braking that can produce negative link energies and require special handling. The harness then iterates over a fixed loop of proposing an experiment, training the model, and evaluating the model with fixed metrics. This particular loop is expensive in human time and attention and as we expand to hundereds of trials, it can become prohibitive.

After running this modified harness for our particular problem, we find that the most important contribution is the synergy between the RSE and the agent. For example, we seeded the agent with context about road geometry and it identified a new geometry based feature, link sinusoity (a measure of how curvy a road is), that we had not considered before but reduced our error metrics. We also asked the agent to survey some state of the art sequnce models and it converged on a variant of a Convolutional Nueral Network (CNN) with a historical link lookback. The resulting model and feature set cut our link level errors by 50% and showed significant improvements in capturing complex link level dynamics like regenerative braking. Moreover, we also attempted a run where we withheld the human seed context and just let the agent explore freely with the same overall time budget. In this particular case, the agent did not converge on the same improvements which provides some anecdotal evidence that the human context, not the agent acting alone, is an important component for this type of problem.

Overall, these explorations seem to suggest that the value of running agent based auto research isn't the agent autonomy exclusively but rather, a synergy where the RSE can supply direction, domain knowledge and review and the agent can absorb the prohibitive context switching cost of running hundereds of short and varied experiments. The agent expands the possibilities of what the RSE can explore and enhances their scientific judgemnt rather than replacing it. We argue this can be a transferable pattern where a domain context aware LLM agent handles the experiment burden while the RSE owns the invaluable human aspects. Future work is planned to expand the framework across other complex powertrain types like Plug-in Hybrid Electric Vehicles and integrate classical hyperparameter optimizers downstream of the agent for fine tuning.

## References

1. _RouteE-Powertrain._ National Renewable Energy Laboratory. https://github.com/NatLabRockies/routee-powertrain
2. _RouteE-Compass._ National Renewable Energy Laboratory. https://github.com/NatLabRockies/routee-compass
3. _FASTSim: Future Automotive Systems Technology Simulator._ National Renewable Energy Laboratory. https://www.nlr.gov/transportation/fastsim.html
4. Karpathy, A. _autoresearch._ https://github.com/karpathy/autoresearch

## Connection to Mission, Goals, & Interests of US-RSE Community

This work speaks directly to the US-RSE community's interest in how RSEs can responsibly and practically apply LLMs inside research workflows. It moves beyond "autonomous experimentation" toward an honest model of human–agent collaboration where each side contributes what it does best. We demonstrate a pattern in which the RSE team retains scientific direction and evaluation authority while the LLM agent absorbs the per-experiment engineering burden that would otherwise gate structural ML improvements. The architecture cleanly separates domain knowledge and agent-driven planning, giving other RSE groups a reusable template they can adapt to scale structural exploration of their own domain specific ML models. Reproducibility and rigor are preserved despite a non-deterministic LLM in the loop: the harness enforces a fixed dataset, a fixed evaluation protocol, and a logged trial history. Finally, in keeping with open-science values, the harness and findings are shared so that other teams navigating the challenges of maintaining domain ML software can replicate the pattern and adapt it to their own systems.
