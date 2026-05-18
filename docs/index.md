---
layout: default
title: "Forming global estimates of self-performance from local confidence - extension"
description: "A Python replication + extension of Rouault, Dayan & Fleming (2019)"
---

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {
    inlineMath: [],
    displayMath: [],
    processEscapes: true
  },
  TeX: {
    extensions: ["AMSmath.js", "AMSsymbols.js"]
  }
});
</script>

<script defer src="https://cdn.jsdelivr.net/npm/mathjax@2/MathJax.js?config=TeX-AMS_CHTML"></script>

# Metacognitive sensitivity, local confidence, and uncertainty
*Nina Edgley · May 2026*
*Rouault M., Dayan P. & Fleming S. M. Forming global estimates of self-performance from local confidence. Nature Communications (2019)*

---

Self-representation refers to an agent's set of self-referential models, spanning from embodiment, interoceptive inference, to narrative models. Metacognition is a key process in the development, sophistication, and change of these models. It grants agents the capacity to learn, to reflect, and to adapt to changing circumstances - or, to adapt circumstances to one's goals.    

I've been studying self-representation through the lens of active inference, exploring how our models come to be. Specifically, I'm curious to understand the degree of potential flexibility in our models - to what extent are we shaped by them, or shape them reflexively? Statistical learning, active inference, and metacognition have all been key frameworks through which to approach this question.    
These interests motivated a replication and extension of Rouault, Dayan, & Fleming's *Forming global estimates of self-performance from local confidence. (2019)*, which explores higher-order behavioural control through metacognitive sensitivity, an SDT-derived metric indicating how well subjects' confidence ratings discriminate and correspond to their objective performance. The paper discusses the relationship between feedback, uncertainty, and metacognition in decision-making, and ultimately, how confidence aggregates into global beliefs about our abilities. It surfaces certain discrepancies, and open questions : what follows is an attempt to address them.

I replicated key findings from Experiment 3, translated the MLE metacognitive sensitivity estimation (fit_meta_d_MLE; Maniscalco & Lau, 2012) to Python, and extended the computational model proposed in the original paper. Specifically, I've compared 2 model classes, and 4 models overall : I propose that uncertainty-aversion, when implemented through changes to the decision module, provides a better account of the systematic feedback preference that the original model underestimates.

Below, I describe the replication of Experiment 3 (original experiment design - replication - results), then the extension (motivation - model architecture - fitting - discussion), before concluding on my takeaways.

---

   
## Original experiment design

Rouault, Dayan, & Fleming developed a perceptual decision-making paradigm to investigate how external feedback and local decision confidence relate to global self-performance estimates, and how these factors are changed in the absence of feedback. Over the course of 3 different experiments, participants were asked to complete 6 short learning blocks that interleaved 2 tasks. Potential tasks varied on 2 dimensions: difficulty (Easy, Difficult), and feedback (Feedback, No Feedback), resulting in 6 task pairings. At the end of each learning block, participants rated (1) which task should be used to calculate a monetary bonus based on their performance at the chosen task, and (2) their overall ability at each task on a continuous scale. 

Both measures operationalised global SPEs, and - when evaluated across the 6 potential task pairings (Difficult+Feedback, Easy+Feedback, Difficult+NoFeedback, etc.) - gave a broader view of how SPE interacted with difficulty and feedback. The candidate hierarchical learning model contained a perceptual module (generating a choice and confidence on each trial) and a learning module (updating global SPEs across trials from local confidence and feedback).

- Experiment 1 focused on testing the exact paradigm described above.
- Experiment 2 incorporated variations in learning block duration.
- Experiment 3 extended the former by adding confidence ratings of participants' perceptual judgements on no-feedback trials. This was done to obtain direct evidence that changes in local confidence were predictive of end-of-block SPEs. It then used these data to evaluate metacognitive sensitivity.

For the purposes of this replication, I focused on Experiment 3, replicating key figures, metacognitive sensitivity fits, and statistical analyses, described below.

    

## Replication

I took the [publicly available data from the Rouault repository](https://github.com/marionrouault/RouaultDayanFleming) and reimplemented the core analyses in Python (paired t-tests, ANOVA, logistic regression model). The original code was in MATLAB.

The main technical challenge was the meta-d' computation, rather than the statistical analyses - the repository contained `scripts/`, which held MATLAB code for replication. I translated these into standard Python, prioritising the key figures for Experiment 3.
The original paper used the MATLAB fit_meta_d_MLE toolbox [Maniscalco & Lau, 2012](https://psycnet.apa.org/record/2012-05338-034), whilst the hierarchical Bayesian extension ([Fleming, 2017](https://github.com/metacoglab/HMeta-d)) required JAGS or R. I translated the MLE fitting procedure into Python, implemented the SDT model and the constrained optimisation over meta-d' and the type-2 criteria. I ran into a blocker on the coordinate-shifted parameterisation, which took several attempts to get right. I debugged this with AI coding assistance, and checked whether the nuts and bolts were correct (tested outputs against the pre-computed ratios contained in the original repository (`mratios`), and got a near 1-1 correspondence).
As an aside, data wrangling also took more time than expected : the original data lived in nested MATLAB structures. It was my first time working with it, so I took a second to apply the different in format, indexing, and arrays. Once the syntax was figured out, the process was straightforward.

As mentioned above, I validated the implementation by comparing the Python MLE m-ratio estimates against the pre-computed values in the original data file. They match to the ~third decimal place (Easy: 0.858 vs 0.858; Difficult: 0.738 vs 0.740).

       

## Results

The primary analyses and figures run for Experiment 3 focused on assessing the interaction between feedback, confidence, difficulty, and objective performance. 
  
### Confidence predicts task choice
This relationship was tested using 2 paired within-subjects t-tests. Confidence was significantly higher for easy than difficult trials (p < 0.001), confirming the manipulation works.

![Confidence by difficulty](figures/02_fig5a_confidence_accuracy_difficulty.png)
*Figure 1: Confidence tracks task difficulty — higher for easy trials (green) than difficult trials (red), confirming the difficulty manipulation worked.*

The original analysis proposed a logistic regression to further assess the contribution that different factors exerted on task choice in the absence of feedback. Two models were run (full and reduced) to control for the different regressors. Both models focused on the trial data unique to Experiment 3 : task choices in no-feedback ratings (with explicit confidence ratings). BIC values were computed for both, and used as the basis for a standard model comparison.
- Full model : 3 regressors - xAcc, xRT, xConf
- Reduced model : 2 regressors - xAcc, xRT

A model comparison (ΔBIC = 22.4) confirmed that confidence significantly improves prediction of task choice, beyond accuracy and reaction time alone. This replicates Rouault, Dayan, & Fleming's initial result, that local confidence was a key input, aggregating to form global SPE (task choice being its proxy). 

![Regression coefficients](figures/02_fig5b_logistic_regression.png)
*Figure 2: Logistic regression coefficients predicting task choice. Confidence difference (confDiff) is a significant predictor above and beyond accuracy difference (accDiff) and reaction time difference (rtDiff).*

Finally, no-feedback conditions were also analysed through the lens of task choice frequencies, relatively to difficulty and confidence. As shown below, mean trial-by-trial confidence values for the 2 tasks were subtracted from one another, then sorted into 3 buckets : small, average, and large differences in confidence between the tasks. As Figure 3 shows, participants increasingly chose the task where they had higher confidence, in step with the relative difference in confidence levels across both tasks.

![Task choice frequency](figures/02_fig5c_task_choice_performance.png)
*Figure 3: Participants increasingly chose the task where they had higher confidence.*

   
### Metacognitive efficiency predicts global self-evaluation

Metacognitive efficiency, measured here as M-ratio (meta-d' / d'), captures how well someone's confidence ratings distinguish their own correct from incorrect responses, independent of actual task performance. An M-ratio of 1 means confidence is informationally optimal, sub-1 that information is lost between deciding and rating confidence. The key results for the paper was that metacognitive efficiency correlated with global SPEs for both MLE and AUROC-2 (non-parametric alternative), with AUROC-2 showing a stronger correlation.
- MLE : ρ = 0.34, p = 0.021
- AUROC-2 : ρ = 0.45, p = 0.002

![Metacognitive efficiency correlation](figures/02_fig5ed_metacog_scatter.png)
*Figure 4: Left — Metacognitive efficiency (M-ratio, MLE) correlates with global self-performance estimates (ρ = 0.34, p = 0.021). AUROC2 shows a stronger version of the same relationship (ρ = 0.45, p = 0.002).*

---
     

## Extension Motivation

In Rouault et al.'s paper, a curious relationship between feedback, confidence, and global SPEs emerges. Confidence tracks performance and hypothetically supports SPE formation. SPE, meanwhile, integrates feedback and confidence. Both depend on performance. 

The paper's hierarchical Bayesian learning model (hereafter <script type="math/tex">\mathcal{M}_0</script>) explains how subjects form global self-performance estimates by updating Beta-distributed beliefs about task performance. Its learning module conditions update rules on feedback availability: on feedback trials, the posterior is updated via binary outcomes (correct → <script type="math/tex">\alpha + 1</script>; incorrect → <script type="math/tex">\beta + 1</script>), while on no-feedback trials, it uses graded confidence signals derived from the internal decision variable (<script type="math/tex">\alpha + p_{\text{correct}}</script>, <script type="math/tex">\beta + (1 - p_{\text{correct}})</script>).
The model captures the core behavioural patterns well, but consistently underestimates the observed preference for feedback tasks. This, along with several empirical regularities in the data suggest that feedback plays a richer role in SPE formation than a binary update rule can express:

- Trial count only influences SPE formation if feedback is present - no learning effects are shown in no-feedback trials
- The difference in performance between E-NF and D-F tasks is tempered by feedback: when performance difference are small, feedback tasks are preferred. When they're large, choice follows performance.
- 2x2 ANOVA in Experiment 1 shows that feedback does not have a significant effect on objective task performance.

The original reasoning proposed that local confidence forms trial-to-trial, bolstered by metacognitive sensitivity, which enables better performance tracking. Confidence is therefore tightly coupled to performance, and in turn supports the formation of global SPEs. However, the paper finds that feedback does not have a significant on effect performance - but its absence does significantly impact task choices (SPE assay). The impact of feedback on SPEs must therefore come from elsewhere in the causal chain : update rules in the decision module, or a broader parameter affecting the distributions. 

I hypothesised two candidate alternative mechanisms, operating at different stages of the model:
1. **Bayesian Cue Integration Model, learning module**: feedback and confidence are integrated as separate, weighted sources of information for feedback trials in the learning module. A free parameter (η) controls the weights accorded.
2. **Uncertainty Aversion, decision module**: the comparison between accumulated beliefs is sensitive to their precision, rather than their means - it penalises uncertain beliefs at the point of choice, biasing decisions towards feedback (mean-variance utility, λ).

What followed was a comparison of four candidate models, which I've specified and fit. Details below:
- **<script type="math/tex">\mathcal{M}_0</script>** - original Bayesian hierarchical learning model
- **<script type="math/tex">\mathcal{M}_\lambda</script>** - uncertainty-averse model
- **<script type="math/tex">\mathcal{M}_\eta</script>** - cue-integration model
- **<script type="math/tex">\mathcal{M}_{\eta\lambda}</script>** - combined model (uncertainty-aversion, cue integration)
    

## Model Architecture
   
All four models share the same perceptual module, but differ along 2 dimensions - how evidence is accumulated (perceptual learning module), and how it then drives the task choice (SPE).
    

### Perceptual Module

On each trial, the observer receives a noisy sensory signal where <script type="math/tex">d_t \in \{-1, +1\}</script> encodes the target side, <script type="math/tex">k_{ch}</script> is perceptual sensitivity (estimated per subject from overall accuracy), and <script type="math/tex">\Delta_t</script> is the stimulus strength (easy = 60, hard = 24). The observer responds left if <script type="math/tex">X_t \lt 0</script>, right otherwise:
<script type="math/tex; mode=display">
X_t \sim \mathcal{N}(d_t \cdot k_{ch} \cdot \Delta_t, \ 1)
</script>

Confidence is derived from the observer's internal model of their own sensitivity (<script type="math/tex">k_{conf}</script>, a free parameter) where <script type="math/tex">\sigma(\cdot)</script> is the logistic function. This is identical across all four models:
<script type="math/tex; mode=display">
p(\text{correct}_t)
=
\begin{cases}
\sigma\bigl(2 k_{conf} \Delta_t X_t\bigr)
\quad \text{if } X_t \gt 0 \\
1 - \sigma\bigl(2 k_{conf} \Delta_t X_t\bigr)
\quad \text{if } X_t \le 0
\end{cases}
</script>

   
### Learning Module

Each task <script type="math/tex">j \in \{1, 2\}</script> maintains a Beta posterior <script type="math/tex">\text{Beta}(\alpha_j, \beta_j)</script>, initialised at <script type="math/tex">\text{Beta}(6, 3)</script>.

**<script type="math/tex">\mathcal{M}_0</script> and <script type="math/tex">\mathcal{M}_\lambda</script>** use the original update rule:
<script type="math/tex; mode=display">
\text{Feedback trial:} \quad \alpha_j \mathrel{+}= \mathbb{1}[\text{correct}], \quad \beta_j \mathrel{+}= \mathbb{1}[\text{incorrect}]
</script>
<script type="math/tex; mode=display">
\text{No-feedback trial:} \quad \alpha_j \mathrel{+}= p(\text{correct}_t), \quad \beta_j \mathrel{+}= 1 - p(\text{correct}_t)
</script>

**<script type="math/tex">\mathcal{M}_\eta</script> and <script type="math/tex">\mathcal{M}_{\eta\lambda}</script>** introduce a cue-integration parameter <script type="math/tex">\eta \in [0, 1]</script> that governs the relative weight of feedback versus confidence on feedback trials:
<script type="math/tex; mode=display">
\text{Feedback trial:} \quad \alpha_j \mathrel{+}= \eta \cdot \mathbb{1}[\text{correct}] + (1 - \eta) \cdot p(\text{correct}_t)
</script>
<script type="math/tex; mode=display">
\phantom{\text{Feedback trial:}} \quad \beta_j \mathrel{+}= \eta \cdot \mathbb{1}[\text{incorrect}] + (1 - \eta) \cdot (1 - p(\text{correct}_t))
</script>
<script type="math/tex; mode=display">
\text{No-feedback trial:} \quad \text{unchanged}
</script>

When <script type="math/tex">\eta = 1</script>, this recovers the original binary rule (as confidence is weighted at 0, leaving the original binary update intact). When <script type="math/tex">\eta \lt 1</script>, even feedback trials incorporate the graded confidence signal. Feedback and confidence are treated as two cues to be integrated rather than as distinct update regimes on feedback v. no-feedback tasks. The total weight increment per trial remains 1 regardless of <script type="math/tex">\eta</script>, preserving the rate of evidence accumulation assumed in the original model.

    
### Decision
  
At the end of each learning block, the observer chooses between tasks based on their accumulated beliefs.

**<script type="math/tex">\mathcal{M}_0</script> and <script type="math/tex">\mathcal{M}_\eta</script>** use the original Monte Carlo comparison (drawing samples from each Beta posterior and computing the probability that one exceeds the other):
<script type="math/tex; mode=display">
P(\text{choose } T_1) = P\bigl(\theta_1 \gt \theta_2\bigr), \quad \theta_j \sim \text{Beta}(\alpha_j, \beta_j)
</script>

**<script type="math/tex">\mathcal{M}_\lambda</script> and <script type="math/tex">\mathcal{M}_{\eta\lambda}</script>** replace this with a mean–variance utility approach, introducing uncertainty-aversion via <script type="math/tex">\lambda \geq 0</script>:
<script type="math/tex; mode=display">
\mu_j = \frac{\alpha_j}{\alpha_j + \beta_j}, \qquad \sigma^2_j = \frac{\mu_j(1 - \mu_j)}{\alpha_j + \beta_j + 1}
</script>
<script type="math/tex; mode=display">
U_j = \mu_j - \lambda \cdot \sigma_j
</script>

To compute the probability of task selection for these two models, I computed the approx. Beta posterior analytically (using mean and variance), rather than using the Monte Carlo sampling method. When <script type="math/tex">\lambda \gt 0</script>, the agent penalises tasks with uncertain posteriors — those with fewer observations or more ambiguous evidence. When <script type="math/tex">\lambda = 0</script>, the mean–variance formulation approximates the Monte Carlo comparison (both essentially reduce to comparing posterior means under the Gaussian approximation to the Beta). This directly captures the preference for feedback tasks at equal performance: feedback trials produce binary (0 or 1) increments, yielding higher effective weighted counts and tighter posteriors than the confidence-based increments on no-feedback trials.
<script type="math/tex; mode=display">
P(\text{choose } T_2) = \Phi\!\left(\frac{U_2 - U_1}{\sqrt{\sigma^2_1 + \sigma^2_2}}\right)
</script>

In short, the summary for each model change can be reduced to:
<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Label</th>
      <th>Learning</th>
      <th>Decision</th>
      <th>Free parameters</th>
      <th>Nests within</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><script type="math/tex">\mathcal{M}_0</script></td>
      <td>Original (BAY)</td>
      <td>Binary / graded</td>
      <td>Monte Carlo</td>
      <td><script type="math/tex">k_{conf}</script></td>
      <td><script type="math/tex">\mathcal{M}_\lambda, \mathcal{M}_\eta, \mathcal{M}_{\eta\lambda}</script></td>
    </tr>
    <tr>
      <td><script type="math/tex">\mathcal{M}_\lambda</script></td>
      <td>Uncertainty-averse</td>
      <td>Binary / graded</td>
      <td>Mean–variance</td>
      <td><script type="math/tex">k_{conf}, \lambda</script></td>
      <td><script type="math/tex">\mathcal{M}_{\eta\lambda}</script></td>
    </tr>
    <tr>
      <td><script type="math/tex">\mathcal{M}_\eta</script></td>
      <td>Cue-integration</td>
      <td><script type="math/tex">\eta</script>-blended</td>
      <td>Monte Carlo</td>
      <td><script type="math/tex">k_{conf}, \eta</script></td>
      <td><script type="math/tex">\mathcal{M}_{\eta\lambda}</script></td>
    </tr>
    <tr>
      <td><script type="math/tex">\mathcal{M}_{\eta\lambda}</script></td>
      <td>Combined</td>
      <td><script type="math/tex">\eta</script>-blended</td>
      <td>Mean–variance</td>
      <td><script type="math/tex">k_{conf}, \eta, \lambda</script></td>
      <td>—</td>
    </tr>
  </tbody>
</table>

Given the nested structure of the different models, this approach also makes it clear where potentially improvements in fitting procedures come from - parameter, module, decision rules. 
- If <script type="math/tex">\mathcal{M}_\eta</script> wins, the feedback bias is best explained by how evidence is accumulated. 
- If <script type="math/tex">\mathcal{M}_\lambda</script> wins, it is explained by how evidence is compared. 
- If <script type="math/tex">\mathcal{M}_{\eta\lambda}</script> wins, both stages contribute. 
- If <script type="math/tex">\mathcal{M}_0</script> wins, the original specification is already sufficient and the bias is adequately captured by the structural difference between binary and graded updates.

   
## Model Fitting

All models were fit to the Experiment 2 data (<script type="math/tex">n = 29</script>, 30 task choices per subject, as in the original paper). Since the MATLAB files contained aggregate data (analyses, summary stats, etc.), I reconstructed trial-level data manually. Specifically, I derived <script type="math/tex">k_{ch}</script> per subject from mean accuracy via

<script type="math/tex; mode=display">
d' = \Phi^{-1}(\bar{p}) - \Phi^{-1}(1 - \bar{p}), \qquad k_{ch} = \frac{d'}{2\,\bar{\delta}}
</script>

where <script type="math/tex">\bar{p}</script> is mean accuracy pooled across tasks and pairings, and <script type="math/tex">\bar{\delta} = 42</script> is the average dot difference across easy (<script type="math/tex">\delta = 60</script>) and difficult (<script type="math/tex">\delta = 24</script>) conditions. Task choices were taken from `task1val`. This can be found in `04_extension.ipynb`: I tested it with a check on the kch range, and various subject summaries.

Otherwise, I initialised the model fits using coarse-grid parameter combinations first (over <script type="math/tex">k_{conf}</script>, <script type="math/tex">\eta</script>, <script type="math/tex">\lambda</script> grids), then followed with Limited-memory Broyden-Fletcher-Goldfarb-Shanno with Bounds refinement (L-BFGS-B!). For model comparison, I used standard BIC at the subject level and a fixed-effects approximation for group-level inference.

![Lambda x Eta distribution + BIC comparison](figures/04_model_comparison_params.png)
*Figure 5: Parameter fits - individual and combined - and BIC comparison*
  

### Results
  
A summary of the results can be compressed to:

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th><script type="math/tex">k</script></th>
      <th>Mean NLL</th>
      <th>Mean BIC</th>
      <th>N best (BIC)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><script type="math/tex">\mathcal{M}_0</script> (Original)</td>
      <td>1</td>
      <td>17.31</td>
      <td>38.03</td>
      <td><strong>17</strong></td>
    </tr>
    <tr>
      <td><script type="math/tex">\mathcal{M}_\lambda</script> (Uncertainty-averse)</td>
      <td>2</td>
      <td>15.74</td>
      <td>38.28</td>
      <td><strong>12</strong></td>
    </tr>
    <tr>
      <td><script type="math/tex">\mathcal{M}_\eta</script> (Cue-integration)</td>
      <td>2</td>
      <td>17.10</td>
      <td>41.00</td>
      <td>0</td>
    </tr>
    <tr>
      <td><script type="math/tex">\mathcal{M}_{\eta\lambda}</script> (Combined)</td>
      <td>3</td>
      <td>15.76</td>
      <td>41.72</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

At the group level, the original model remains preferred (BIC weight = 0.976). However, the picture is more nuanced at the individual level: 12 of 29 subjects are best fit by <script type="math/tex">\mathcal{M}_\lambda</script>, with several showing substantial improvements in likelihood (e.g., <script type="math/tex">\Delta</script>NLL = 6.0, 5.2, 3.5 for the strongest cases).
More importantly, <script type="math/tex">\mathcal{M}_\lambda</script> predicts the feedback-biased task selection (see below - cases ENF-EF, ENF-DF, and EF-DNF, in which the selection of the feedback-driven task is more visible). 

Group-level mean absolute error confirms the same ordering: <script type="math/tex">\mathcal{M}_\lambda</script> (MAE = 0.087) substantially outperforms <script type="math/tex">\mathcal{M}_0</script> (0.127), <script type="math/tex">\mathcal{M}_\eta</script> (0.134), and <script type="math/tex">\mathcal{M}_{\eta\lambda}</script> (0.097) in reproducing observed task choice frequencies across pairings and durations.

![Observed T1 choice frequency by pairing × duration](figures/04_4model_task_choice_frequency_comparison.png)
*Figure 6: Observed task choice frequency comparison across models, relative to pairing and duration - stronger fit on feedback-driven task selection for the <script type="math/tex">\mathcal{M}_\lambda</script>.*

The model fits show a strong positive skew in uncertainty aversion: most participants show none, with <script type="math/tex">\lambda = 0</script>, with a long positive tail for the remaining ~41%. My interpretation is that the <script type="math/tex">\mathcal{M}_\lambda</script> captures and better explains the behaviour of participants with positive <script type="math/tex">\lambda</script> values, hence the better individual performance. 
For subjects with no uncertainty aversion (<script type="math/tex">\lambda \approx 0</script>), NLL improvements relative to <script type="math/tex">\mathcal{M}_0</script> likely reflect the deterministic Gaussian approximation being more stable than Monte Carlo sampling rather than a genuine effect of lambda. Subjects with <script type="math/tex">\lambda \gt 0</script> provide the cleaner evidence for the extended mechanism, with the largest NLL change values (6.0, 5.2, 3.5) concentrated in this group.

The cue-integration model <script type="math/tex">\mathcal{M}_\eta</script> and combined model <script type="math/tex">\mathcal{M}_{\eta\lambda}</script> are decisively ruled out. Neither win for any subjects.
- <script type="math/tex">\mathcal{M}_\eta</script> and its mean NLL (17.10) barely improves on the original (17.31), which isn't sufficient to justify the additional parameter. The fitted <script type="math/tex">\eta</script> distribution is bimodal, clustering at the bounds (<script type="math/tex">\eta \approx 0</script> or <script type="math/tex">\eta \approx 1</script>) with almost no subjects in between. From my understanding, this points to non-identifiability: the likelihood surface is essentially flat with respect to <script type="math/tex">\eta</script>, with the parameter not doing meaningful work.
- <script type="math/tex">\mathcal{M}_{\eta\lambda}</script> almost always recovers the <script type="math/tex">\mathcal{M}_\lambda</script> fit exactly (<script type="math/tex">\eta</script> collapses to 1.0 whenever <script type="math/tex">\lambda \gt 0</script>), confirming that the two mechanisms do not interact. Adding <script type="math/tex">\eta</script> to a model that already has <script type="math/tex">\lambda</script> does not provide additional explanatory power.
   

## Conclusions

The four-way comparison provides a clean dissociation between parameters, modules, and the effect of various architectures and decision rules on the final predictions. To the extent that the original model's feedback bias can be reduced, the mechanism sits in the **decision module**, not the learning module.

The original model already distinguishes between feedback and no-feedback trials at the level of evidence accumulation. The <script type="math/tex">\eta</script> extension tested whether this categorical distinction was too rigid, proposing instead a graded / weighted blend of confidence and feedback on each trial. This hypothesis returned null : the learning module's original specification (feedback as a qualitatively different type of evidence) was already appropriate. 

What does have an effect is changing how accumulated evidence is evaluated, rather than formed. The <script type="math/tex">\lambda</script> parameter captures uncertainty aversion at the point of task choice: subjects with high <script type="math/tex">\lambda</script> penalise tasks where their performance belief is imprecise. Because feedback trials produce binary increments (0 or 1) while no-feedback trials produce fractional increments (<script type="math/tex">p_{\text{correct}} \in (0,1)</script>), feedback posteriors are tighter for feedback trials.   
An uncertainty-averse agent will therefore prefer the feedback task even when both posteriors have similar means, which resolves the feedback discrepancy the original model underestimates.

This is consistent with broader findings in the metacognition literature, providing preliminary evidence that the precision of self-performance beliefs, not just their content, influences higher-order evaluation. 

That said, the original model remains the group-level winner. The BIC penalty for <script type="math/tex">\lambda</script> (<script type="math/tex">\ln(30) \approx 3.4</script> nats) outweighs the likelihood improvement for the majority of subjects. This means that for most participants, the original model's implicit precision asymmetry is already sufficient to explain task choice. Only a subgroup (~41%) shows an additional, explicit sensitivity to posterior uncertainty at the decision stage. Whether this subgroup can be characterised by independent measures (e.g., metacognitive efficiency, anxiety, intolerance of uncertainty) would be the natural next question I'd want to explore.

---

## Code

All analyses are in Jupyter notebooks: [github.com/ninaedgley/rouault-2019-extension](https://github.com/ninaedgley/rouault-2019-extension).

- `01_load_data.ipynb` : preprocessing, MATLAB file extraction
- `02_replication.ipynb` : statistical analyses and figure replications
- `03_metad.ipynb` : meta-d' translation to Python, MLE and validation
- `04_extension.ipynb` : extension - specification and comparison of 4 candidate learning models

Figures can be found under `docs/figures/`.

---

*Reference: Rouault, M., Dayan, P., & Fleming, S. M. (2019). Forming global estimates of self-performance from local confidence. Nature Communications, 10(1), 1141.*
