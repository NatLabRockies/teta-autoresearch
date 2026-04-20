# Experiment Plan: <tag>

## Starting point

- Powertrain: `<bev|ice|phev>`
- Branch: `routee-autoresearch/<tag>` (where `<tag>` is `<powertrain>-<date>`)
- Forked from: `<powertrain>/best` at `<commit>` (or `bev/best` / `main` if no prior best exists for this powertrain)
- Best known RMSE for this powertrain: <from learnings.md or fresh baseline>

## Cross-cutting insights consulted

List the specific items from `learnings.md → Cross-cutting insights` you're carrying into this session (e.g. "MSE aligns with RMSE, BatchNorm hurts under short budget, predict energy_rate not energy_gge").

- <insight 1 and how it shapes your baseline>
- <insight 2>

## Session goals

1. <primary goal>
2. <secondary goal>

## Planned experiments (rough order)

1. Baseline run
2. <first experiment idea — what and why>
3. <second experiment idea>

## Constraints / focus areas

- <any session-specific constraints>
- <focus areas from seed.md, powertrain-specific hypotheses from learnings.md>

## Progress log

Updated after each experiment. Format: `- [x] expN: description -> rmse (status)`
