# Autoscientist

> **A deterministic research engine for tabular machine learning — designed to let LLMs propose experiments while the engine controls execution, evaluation, and evidence.**

Autoscientist is an experimental framework for **human-in-the-loop agentic data science research**.

The project is built around a simple principle:

> **The LLM proposes. The engine decides.**

Instead of allowing an LLM to freely generate code, run arbitrary experiments, and decide whether a model improved, Autoscientist provides a deterministic execution layer for:

- Dataset understanding and data contracts
- Data validation and leakage detection
- Validation strategy selection
- Baseline construction
- Feature engineering
- Gradient-boosted decision tree experiments
- Cross-validation
- Statistical comparison
- Practical significance
- Complexity-aware model selection
- Experiment governance
- Sandboxed execution
- Reproducible research state

The long-term goal is to place an LLM on top of this engine as a **research orchestrator**: generating hypotheses, proposing experiments, and interpreting evidence while deterministic components remain responsible for execution and scientific decision rules.

> **Status: Early-stage research prototype / actively under development.**

---

## Why Autoscientist?

Modern LLMs are increasingly capable of writing machine-learning code. But generating code is not the same as conducting reliable ML research.

A naive LLM-driven workflow looks like:

```text
LLM
 │
 ├── "Try target encoding"
 ├── "Try another model"
 ├── "Add these features"
 ├── "Score improved!"
 └── "Let's continue..."
```

This creates several problems:

- Experiments may not be comparable
- Validation can be inconsistent
- Data leakage can go unnoticed
- Small metric differences can be overinterpreted
- Increasingly complex pipelines can win without meaningful evidence
- Previous decisions become difficult to audit
- The LLM can become both the proposer and judge of its own experiments

Autoscientist separates these responsibilities.

```text
                     ┌──────────────────────┐
                     │         LLM          │
                     │                      │
                     │ Hypothesis generation│
                     │ Experiment proposals │
                     │ Research reasoning   │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │ Deterministic Engine │
                     │                      │
                     │ Validate             │
                     │ Execute              │
                     │ Evaluate             │
                     │ Compare              │
                     │ Govern               │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │       Evidence       │
                     │                      │
                     │ Metrics              │
                     │ Confidence intervals │
                     │ Statistical verdicts  │
                     │ Complexity           │
                     │ Experiment history   │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │    Research State    │
                     └──────────┬───────────┘
                                │
                                └──────► LLM
```

The LLM provides **creativity and hypothesis generation**.

The engine provides **reproducibility, constraints, and evidence**.

---

# Core Design Principles

## 1. The LLM Proposes; Deterministic Policy Decides

Research decisions are represented explicitly rather than hidden inside prompts.

The policy layer determines:

- Which research directions are open
- Which decisions still need evidence
- Whether a proposal is valid
- Whether a previous result should be confirmed
- When simplification should be attempted
- When the research process should stop

The policy is deterministic and replayable.

```text
LLM Proposal
      │
      ▼
Deterministic Policy
      │
 ┌────┴─────┐
 │          │
valid     rejected
 │
 ▼
Experiment
```

This makes research decisions auditable instead of depending entirely on what an LLM happened to say.

---

## 2. Build the Baseline Before Optimizing

Autoscientist uses a baseline ladder rather than immediately searching for the strongest possible model.

Conceptually:

```text
E000  Trivial baseline
  │
  ▼
E001  Reference model
  │
  ▼
E002  Tuned reference
  │
  ▼
Research experiments
```

This provides context for later improvements and helps distinguish:

- Preprocessing effects
- Feature engineering effects
- Model-family effects
- Hyperparameter effects

A complicated pipeline should not receive credit simply because the starting point was weak.

---

## 3. Statistical Evidence Over Raw Metric Differences

An experiment is not considered successful merely because:

```text
score_new > score_old
```

Autoscientist compares experiments on the **same validation folds** and tracks:

- Paired fold differences
- Corrected standard errors
- Confidence intervals
- Practical improvement thresholds
- Minimum detectable effect
- Secondary metric constraints
- Statistical verdicts

Possible outcomes include:

```text
SUPPORTED
REJECTED
INCONCLUSIVE
PARTIALLY_SUPPORTED
```

This is important because:

```text
+0.003 metric improvement
```

does not automatically mean:

```text
"The new approach is better."
```

If the available data cannot reliably resolve the effect, the system should be able to say:

> **INCONCLUSIVE**

rather than manufacture confidence.

---

## 4. Research for Coverage, Not Just Leaderboard Gain

The research policy maintains explicit **decision slots** representing common modeling decisions.

Examples include:

```text
missing_handling
categorical_encoding
model_family
imbalance_handling
target_transform
feature_interaction
hyperparameter_tuning
```

The goal is not simply:

> "Find the highest score."

Instead:

> **Determine which modeling decisions actually matter for this dataset.**

A decision that has been tested and found not to matter can be closed, reducing unnecessary experimentation.

---

## 5. Complexity Is Part of Model Selection

Higher performance is not automatically better.

Autoscientist tracks pipeline complexity and can penalize unnecessarily complicated solutions.

Conceptually:

```text
Research utility
    =
    predictive performance
    − complexity penalty
```

This encourages the system to prefer a simpler pipeline when two approaches provide practically equivalent evidence.

---

# Data Understanding

Before modeling, Autoscientist builds a structured `DataContract` describing the dataset.

The EDA layer covers several areas.

### Dataset Integrity

- Row and column counts
- Duplicate rows
- Constant columns
- Empty columns
- Memory footprint

### Target Understanding

- Automatic task inference
- Binary classification
- Multiclass classification
- Regression
- Target distribution
- Class prevalence

### Schema Inference

Columns are classified into roles such as:

```text
numeric
categorical
datetime
id
```

The system also attempts to recognize numeric and datetime values stored as strings.

### Leakage Screening

The EDA layer includes deterministic checks for signals such as:

- ID-like columns
- Unusually predictive individual features
- Near-perfect feature-target association
- Informative missingness
- Duplicate rows

Suspicious findings are surfaced as **warnings or blocking findings**, rather than silently being treated as valid modeling signals.

### Validation Structure

The system also examines whether the dataset may require:

```text
Stratified K-Fold
K-Fold
Group K-Fold
Time Series validation
```

Some structural ambiguities are deliberately surfaced for confirmation rather than automatically decided from heuristics.

---

# Validation and Statistical Layer

Autoscientist treats validation as part of the research protocol rather than a configurable afterthought.

Experiments use a consistent validation specification so competing pipelines can be compared on the same folds.

The statistical layer includes:

- Paired cross-validation comparison
- Nadeau–Bengio corrected standard error
- Confidence intervals
- Minimum detectable effect
- Statistical power profiling
- Practical significance thresholds
- Secondary metric gates
- Empirical variance calibration after sufficient comparisons

The system can therefore distinguish between:

```text
Observed improvement
        │
        ├── Statistically supported
        ├── Rejected
        └── Insufficient evidence
```

rather than treating every positive delta as a discovery.

---

# Feature Engineering Sandbox

LLM-generated feature engineering introduces another problem:

> How do we execute generated transformations without allowing an invalid transformation to silently contaminate the research process?

Autoscientist therefore provides a sandboxed execution layer with runtime checks.

The sandbox evaluates generated transformations for properties such as:

- Schema validity
- Deterministic behaviour
- Row independence
- Batch invariance
- Fit/transform separation
- Resource limits
- Execution time

For example, a transformation such as:

```python
def transform(self, X):
    X["normalized"] = X["income"] / X["income"].mean()
    return X
```

may produce valid-looking output while depending on the composition of the batch being transformed.

Runtime evidence can detect this class of problem.

The sandbox is intended primarily as a **correctness and containment layer for generated code**, while the broader execution environment is isolated through Linux/Docker.

---

# Supported Modeling Direction

The current modeling scope is intentionally focused on **gradient-boosted decision tree models**:

- LightGBM
- XGBoost
- CatBoost

The goal is not to implement every possible ML algorithm.

Instead, Autoscientist focuses on making experimentation around a constrained modeling family:

```text
Data
 │
 ▼
EDA / Data Contract
 │
 ▼
Baseline
 │
 ▼
Feature Engineering
 │
 ▼
GBDT Model
 │
 ▼
Cross Validation
 │
 ▼
Statistical Evaluation
 │
 ▼
Evidence
```

This constrained scope makes it possible to focus on **research methodology and experiment governance** rather than broad algorithm coverage.

---

# Research Policy

The research policy currently supports several types of actions:

```text
PROBE
FEATURE
MODEL
CONFIRM
SIMPLIFY
STOP
```

The routing policy is deterministic.

A simplified research loop looks like:

```text
                  ┌─────────────┐
                  │ Research    │
                  │ State       │
                  └──────┬──────┘
                         │
                         ▼
                  ┌─────────────┐
                  │ Policy      │
                  │ Router      │
                  └──────┬──────┘
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
       Open decision            Confirmation
             │                       │
             └───────────┬───────────┘
                         ▼
                  LLM proposal
                         │
                         ▼
                    Validation
                         │
                         ▼
                    Experiment
                         │
                         ▼
                     Evidence
                         │
                         ▼
                  Research State
```

The policy also reserves part of the experiment budget for confirming earlier results against a newer incumbent. This helps avoid treating a result measured against an outdated control as permanently settled.

---

# Configuration

A research session is configured through YAML.

Example:

```yaml
dataset: data/telco.csv
target: Churn

objective: "Identifikasi pelanggan yang akan berhenti berlangganan bulan depan."

budget:
  max_experiments: 50
  max_wall_time_min: 90
  max_tokens: 1500000

evaluation:
  primary_metric: auto
  delta_practical: 0.01

constraints:
  max_features: 200
  max_pipeline_depth: 5
  complexity_lambda: 0.02

governance:
  approval_mode: interactive

seed: 42
```

The configuration controls:

- Research budget
- Evaluation criteria
- Complexity constraints
- Governance
- Reproducibility

---

# Project Structure

```text
Autoscientist/
│
├── configs/
│   └── research.yaml
│
├── data/
│   └── ...
│
├── scripts/
│   └── ...
│
├── src/
│   └── dsar/
│       │
│       ├── adapters/
│       │   └── sandbox.py
│       │
│       ├── core/
│       │   ├── contracts.py
│       │   ├── statistics.py
│       │   ├── policy.py
│       │   ├── complexity.py
│       │   ├── fingerprint.py
│       │   └── ...
│       │
│       ├── modeling/
│       │   └── cv.py
│       │
│       ├── stages/
│       │   ├── eda.py
│       │   ├── baseline.py
│       │   └── loop.py
│       │
│       └── ...
│
├── tests/
│
├── Dockerfile
├── compose.yaml
├── pyproject.toml
└── uv.lock
```

---

# Installation

Autoscientist currently targets Python 3.11+ and is developed around a Linux execution environment.

The project uses [uv](https://docs.astral.sh/uv/) for dependency management.

## Clone

```bash
git clone https://github.com/Nauviii/Autoscientist.git
cd Autoscientist
```

## Install Dependencies

```bash
uv sync --extra dev
```

## Run Tests

```bash
uv run pytest -q
```

---

# Docker

For reproducible execution, the project also provides a Linux-based Docker environment.

Build and run the test suite:

```bash
docker compose run --rm dsar
```

Or open an interactive shell:

```bash
docker compose run --rm shell
```

The Docker environment pins execution to Linux and provides the system dependencies required by the GBDT stack.

---

# Current Status

Autoscientist is **actively under development**.

The current implementation focuses primarily on the deterministic research engine and its experimental infrastructure.

### Implemented / Under Active Development

- [x] Dataset profiling and EDA
- [x] Data contract
- [x] Schema inference
- [x] Leakage screening
- [x] Validation strategy inference
- [x] Statistical power analysis
- [x] Baseline ladder
- [x] Paired CV comparison
- [x] Practical significance
- [x] Secondary metric gates
- [x] Complexity-aware proposal selection
- [x] Deterministic research policy
- [x] Experiment fingerprinting
- [x] Feature engineering sandbox
- [x] Runtime transformation guards
- [x] Docker/WSL-oriented execution environment
- [x] Unit-test coverage for core research logic

### In Progress

- [ ] End-to-end research loop
- [ ] LLM proposal generation
- [ ] LLM-to-engine orchestration
- [ ] Persistent experiment ledger
- [ ] Human approval workflow
- [ ] Rich experiment reports
- [ ] Full CLI workflow

---

# Roadmap

The project is intentionally being developed from the bottom up.

```text
Phase 1
Deterministic research primitives
        │
        ▼
Phase 2
End-to-end experiment engine
        │
        ▼
Phase 3
LLM hypothesis & proposal layer
        │
        ▼
Phase 4
Human-in-the-loop research loop
        │
        ▼
Phase 5
Long-running autonomous research
```

The long-term architecture is:

```text
                ┌──────────────────────┐
                │       Researcher     │
                │        / Human       │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │         LLM          │
                │ Hypothesis & Proposal│
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │    Research Engine   │
                │                      │
                │ Policy               │
                │ Validation           │
                │ Sandbox              │
                │ Modeling             │
                │ Statistics           │
                │ Governance            │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │       Evidence       │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │   Research State     │
                └──────────────────────┘
```

The objective is not simply to build an agent that can run more experiments.

The objective is to build an agent that can **conduct experiments whose conclusions are constrained by evidence**.

---

# Design Philosophy

Autoscientist is built around four principles:

> **Creativity belongs to the proposer.**  
> **Execution belongs to the engine.**  
> **Evidence belongs to the statistics.**  
> **Accountability belongs to the research state.**

The LLM should be allowed to explore the hypothesis space.

It should not be allowed to redefine the rules by which its own experiments are judged.

---

# Research Scope

Autoscientist is currently focused on:

- Tabular predictive modeling
- Classification and regression
- Gradient-boosted decision trees
- Feature engineering
- Experiment comparison
- Statistical evidence
- Human-in-the-loop research

It is **not yet a general-purpose autonomous scientist**.

The current project is intentionally narrower:

> **Build a reliable research engine first, then place increasingly capable reasoning on top of it.**

---

# License

License information will be added as the project matures.
