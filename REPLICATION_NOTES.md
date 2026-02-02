# Rouault et al. (2019) Replication Analyses
Python replication of Experiment 3 analyses from Rouault, Dayan, & Fleming (2019).

# Background
First time working through a full computational cognitive modelling analysis. Python and statistics were familiar, but needed support for MATLAB code interpretation and HMeta-d' environment setup. Main challenge was data wrangling: identifying variables, and understanding nested MATLAB structures.

## Reference
Rouault, M., Dayan, P., & Fleming, S. M. (2019). Forming global estimates of self-performance from local confidence. *Nature Communications*, 10(1), 1141.

## Completed Analyses

### 1. Confidence by Task Difficulty
Confidence ratings track task difficulty: analysed per-subject mean confidence for Easy vs Difficult conditions (N=46) [manipulation check]
- Script: `analyses/01_confidence_by_difficulty/exp3_confidence_subject_means.py`
- Outputs: `outputs/01_confidence_distributions.png`, `outputs/01_confidence_easy_vs_different.png`

### 2. Task Choice Prediction
Confidence difference predicts task choice beyond accuracy and RT (Pairing 6 analysis).
- Script: `analyses/02_task_choice_regression/exp3_pair6_logrep.py`
- Reference: Figure 5b analysis

### 3. Metacognitive Efficiency Correlation
Metacognitive efficiency (M-ratio) correlates with global self-performance estimates (Spearman ρ = 0.340, p = 0.021).
- Script: `analyses/03_metad_correlation/analyse_metad.py`
- Reference: Figure 5d
Note: Used pre-computed hierarchical Bayesian meta-d' from paper's data file. Did not re-implement MCMC estimation (requires MATLAB HMeta-d' toolbox), could not load properly (>2h).

## Structure
```
analyses/
  01_confidence_by_difficulty/
  02_task_choice_regression/
  03_metad_correlation/
data/
  Exp3.mat
outputs/
  01_confidence_distributions.png
  01_confidence_easy_vs_difficult.png
utils/
  load_exp3.py
  inspect_x_ser6_all.py
```

## Data Source

Original: https://github.com/marionrouault/RouaultDayanFleming