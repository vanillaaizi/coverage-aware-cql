# Coverage-Aware Conservative Q-Learning for Offline RL

**KAIST IE540 — Dynamic Programming & Reinforcement Learning · Solo Term Project · Spring 2026**

A controlled offline reinforcement learning experiment asking:

> **Should Conservative Q-Learning be equally pessimistic everywhere when dataset coverage is highly uneven?**

I extend fixed-penalty CQL with a state-dependent conservative coefficient that increases in rarely observed states and decreases in well-covered states.

## Key result

Across **8 randomized offline datasets**, Coverage-Aware CQL achieved the strongest performance in the submitted 7×7 Gridworld experiment:

| Method | Success rate | Average return |
|---|---:|---:|
| Offline Q-learning | 0.0% | -79.88 ± 0.35 |
| Fixed-alpha CQL | 62.5% | -36.25 ± 34.85 |
| **Coverage-Aware CQL** | **87.5%** | **-14.62 ± 10.25** |

The included experiment script reproduces these aggregate results.

## Motivation

Offline RL learns only from a fixed dataset. Standard Q-learning can exploit actions that are poorly represented—or completely absent—because their Q-values cannot be corrected by new interaction.

Conservative Q-Learning (CQL) addresses this by penalizing unsupported actions. This project studies a simple limitation of using one global penalty everywhere: **rare states and well-covered states do not have the same uncertainty.**

## Coverage-aware penalty

```text
alpha(s) = clip(alpha0 * sqrt(mean_N / (N(s) + 1)), 0.05, 1.20)
```

where:

- `N(s)` is the number of offline transitions observed from state `s`;
- `mean_N` is the mean visitation count among visited states;
- `alpha0 = 0.50` is the base penalty.

Interpretation:

- **low coverage → larger alpha → stronger pessimism**
- **high coverage → smaller alpha → less unnecessary pessimism**

This is an interpretable heuristic extension, not a claim of state-of-the-art performance.

## Experimental setup

- **Environment:** deterministic 7×7 Gridworld, 49 states, 4 actions, 6 obstacles
- **Rewards:** -1 normal step, -2 wall/obstacle collision, 0 at terminal goal
- **Offline data:** 50 episodes per seed from a mostly-good behavior policy with mild exploration
- **Behavior success:** 100%
- **State-action coverage:** 16.6% ± 1.6%
- **Random dataset seeds:** 8
- **Discount factor:** 0.97
- **Learning rate:** 0.05
- **Training epochs:** 80
- **Fixed CQL alpha:** 0.50

All methods use the same fixed dataset and evaluation protocol for each seed.

## Methods compared

1. **Offline Q-learning** — Bellman optimality backup with no conservative regularization.
2. **Fixed-alpha CQL** — CQL-style log-sum-exp penalty with `alpha = 0.50` for every state.
3. **Coverage-Aware CQL** — the same CQL penalty, but `alpha` depends on empirical state visitation.

## Repository structure

```text
.
├── src/
│   ├── __init__.py
│   └── experiment.py
├── tests/
│   └── test_experiment.py
├── results/
│   └── reported_summary.csv
├── report/
│   └── IE540_TermProject_AizereSeitjan.pdf
├── .github/workflows/
│   └── ci.yml
├── requirements.txt
├── CITATION.cff
└── README.md
```

## Course presentation

A sanitized copy of the final 16-slide KAIST term-project presentation is included at `report/IE540_TermProject_AizereSeitjan.pdf`. The student ID has been removed from the public copy.

## Run locally

```bash
git clone https://github.com/vanillaaizi/coverage-aware-cql.git
cd coverage-aware-cql

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/experiment.py
```

The script generates:

- `results_raw.csv`
- `results_summary.csv`
- `fig_success_rate.png`
- `fig_average_return.png`
- `fig_paths.png`

Run tests with:

```bash
pytest -q
```

## What the experiment supports

The controlled study shows that:

- standard offline Q-learning can fail when unsupported actions retain optimistic values;
- CQL-style pessimism reduces this failure mode;
- empirical dataset coverage can be a useful low-cost signal for adapting conservatism in a tabular setting.

## Limitations

This is intentionally a **small mechanism study**, not a D4RL or deep continuous-control benchmark.

Important next steps include:

- replacing state coverage `N(s)` with state-action coverage `N(s, a)`;
- sensitivity analysis over `alpha0`;
- neural function approximation;
- continuous-control benchmarks;
- comparison with IQL and TD3+BC.

## References

- Kumar, A., Zhou, A., Tucker, G., & Levine, S. (2020). *Conservative Q-Learning for Offline Reinforcement Learning.*
- Lange, S., Gabel, T., & Riedmiller, M. (2012). *Batch Reinforcement Learning.*
- Fujimoto, S., Meger, D., & Precup, D. (2019). *Off-Policy Deep Reinforcement Learning without Exploration.*
- Fujimoto, S., & Gu, S. (2021). *A Minimalist Approach to Offline Reinforcement Learning.*

---

**Author:** Aizere Seitjan · NYU Shanghai / KAIST exchange
