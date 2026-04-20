# PLAN

Run a comparison of 

### Methodology

##### Trials

- 40 hour overall runtime per experiment:
  - TPE with cold start
  - TPE with warm start
  - LLM with no guidance
  - LLM with human guidance

##### Inputs
- Powertrain model_type: Bolt

##### Environment + Tools

- Arnaud (specs?)
  - RAM 
  - CPUs
  - GPU type
  - GPUs
- LLM
  - dockerized run context for LLM agent 
  - name of LLM (Claude 4.7 Sauceypants?) 
  - extentions (thinking/reasoning LOW/MED/HIGH)
  - web access
  - token count (per trial?) - add opentelemetry integration as part of docker container
  - parallelism
- TPE
  - parallelism
  - 

### Publishing goals
- Poster @ USRSE 
  - submission date for Talk: Friday May 22, 2026
  - for Poster: Friday August 7, 2026
- open-source data (+ code?) repository