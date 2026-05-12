# Bayesian logistic wheat direction — MCMC run log

Companion to **`bayes_logit_mcmc_macro_pca.py`** (package root) and **Appendix K** of **`docs/wheat_direction_forecast.pdf`**.

Notation uses **LaTeX math**: `$…$` for inline, **`$$ … $$`** for display equations (parses reliably in Cursor/VS Code and many MathJax-based Markdown previews; avoid raw `\[...\]` blocks here because some engines strip thin-space commands like `\,`/`\!`). PDF-ready typography lives in the LaTeX manuscript under **`docs/`**.

## Purpose

Exploratory **Bayesian logistic regression** with **NUTS** (PyMC). **Default** (`BAYES_DESIGN=wheat_pca`): same engineered wheat + cross-asset features, `create_target` labels, and train+val-only scaler+PCA as `TCN_PCA_WHEAT_ONLY`, flattened to `lookback × K` columns per row. **Legacy** (`BAYES_DESIGN=fred_macro`): FRED-MD macro + five lagged wheat log-return features as in `TCN_Revised.py`. **Not** the primary graded trading strategy; documents posterior inference and probability metrics for the VIP appendix. Strategy hook: `STRATEGY_PANEL=bayes` runs inner-val hurdle tuning (`STRATEGY_TUNE_HURDLE=1`, two NUTS passes) unless disabled.

## What is NUTS?

**NUTS** stands for **No-U-Turn Sampler**. It is the default Hamiltonian Monte Carlo (**HMC**)-style engine PyMC uses (via a NUTS implementation similar in spirit to Hoffman & Gelman, 2011) to draw correlated samples whose empirical distribution approaches the posterior
$p(\boldsymbol{\theta}\mid \mathcal{D})$
(the distribution of unknown parameters $\boldsymbol{\theta}=\big(\alpha,\beta_0,\ldots,\beta_{9}\big)^{\mathsf{T}}$ conditional on observations $\mathcal{D}$) once chains have mixed.

- **Hamiltonian idea:** Introduce auxiliary “momentum” variables and simulate trajectories guided by gradients of $\log p(\boldsymbol{\theta}\mid \mathcal{D})$. Proposals travel farther in parameter space than a naive random walk—useful once there are moderately many correlated coefficients (here: intercept plus ten $\beta$-slopes inside $\boldsymbol{\theta}$).

- **“No-U-Turn”:** The sampler extends the simulated trajectory forward and backward until a criterion detects the path would fold back (“make a U-turn”) toward its start. That **adapts** how long each trajectory runs instead of fixing a trajectory length by hand—a common headache in plain HMC.

- **Tune vs draw:** Lines like `tune=2000` adapt step size and trajectory behavior; `draw=2000` keeps **posterior samples** after adaptation. Divergence warnings usually mean proposals are too aggressive for some geometry region (stable runs report **zero divergences**).

Here, **NUTS is only the numerical tool that approximates Bayesian posteriors on a small logistic model**; trading performance depends on predictors and the decision rule, not on the sampler name.

## Metropolis, Gibbs, and how NUTS fits

**Markov chain Monte Carlo (MCMC)** constructs a dependent sequence $\boldsymbol{\theta}^{(1)}, \boldsymbol{\theta}^{(2)}, \ldots$. After **mixing**, the empirical distribution approximates the posterior **$p(\boldsymbol{\theta}\mid \mathcal{D})$**, even though we typically only evaluate an **unnormalized** proportional $\tilde{p}(\boldsymbol{\theta}\mid \mathcal{D})$.

**Metropolis-Hastings (often called “Metropolis” colloquially):**

- Let the current draw be $\boldsymbol{\theta}^{(t)}$. Propose $\boldsymbol{\theta}^{\star}$ from a simple perturbation—for example Gaussian **random walk** $\boldsymbol{\theta}^{\star} = \boldsymbol{\theta}^{(t)} + \text{noise}$.

- Accept or reject $\boldsymbol{\theta}^{\star}$ using Metropolis acceptance so the chain leaves $p(\boldsymbol{\theta}\mid \mathcal{D})$ invariant; densities need only coincide up to a constant. If rejected, set $\boldsymbol{\theta}^{(t+1)}\leftarrow \boldsymbol{\theta}^{(t)}$.

- **Strength:** applies broadly wherever you can **evaluate** proportional posterior density.

- **Weakness:** in high dimension or along narrow ridges, naive random proposals mix slowly (“**poor mixing**”) unless cleverly tuned.

**Gibbs sampling:**

- Write $\boldsymbol{\theta}=(\theta_1,\ldots,\theta_J)$—scalar blocks or subsets. Gibbs cycles through $j$ redrawing $\theta_j$ **exactly** from its **full conditional**
  $p\big(\theta_j \bigm| \boldsymbol{\theta}_{-j},\,\mathcal{D}\big)$,
  where $\boldsymbol{\theta}_{-j}$ freezes every coefficient except coordinate $j$.

- Moves are **automatically accepted** (conditionals imply correct marginal updates whenever the partitioning is valid).

- **Strength:** conjugate hierarchies often yield closed-form $p(\theta_j \mid \boldsymbol{\theta}_{-j},\, \mathcal{D})$ draws.

- **Weakness:** many nonlinear regressions admit no convenient conditionals; **strong dependence** leaves $\boldsymbol{\theta}_{-j}$ pinning $\theta_j$ unless you block-update or augment the state space.

**Relation to NUTS:**

NUTS is still **MCMC**: draws correlate until mixing ends. Hamiltonian drift proposes distant $\boldsymbol{\theta}^{\star}$; MH acceptance keeps the equilibrium law exactly equal to $p(\boldsymbol{\theta}\mid \mathcal{D})$. Unlike Gibbs sweeping, gradients steer correlated **whole-vector** updates jointly on $\boldsymbol{\theta}$.

**Diagnostics:** **R-hat** (multiple chains agreeing) and **ESS** (“effective sample size” after correcting for autocorrelation) summarize **sampler health** and Monte Carlo uncertainty—not whether forecasts beat a coin flip economically.

## Design (10 predictors)

| Block | Count | Description |
|--------|------|-------------|
| Returns | 5 | Scaled log returns at last lag: $\displaystyle \ln\frac{P_{t-1}}{P_{t-1-j}}$ for $j=1,\ldots,5$ |
| Macro PCA | 5 | `StandardScaler` + `PCA(5)` on 31 macro **lag-1** columns; **fit only on training rows** before 15% holdout |

**Label (next-day up):** let $y_t = \mathbf{1}\{ C_t > C_{t-1}\}$ where $C_t$ is the settlement close—the same dichotomy as `TCN_Revised.py`.

**Warm-start / lookback parity:** aggregated rows omit the earliest $29$ trading days (`BAYES_LOOKBACK=30`) so calendar alignment mirrors TCN preprocessing. With covariates $\mathbf{x}_t\in\mathbb{R}^{10}$ stacked per day $t$, the latent linear predictor $\eta_t := \alpha + \boldsymbol{\beta}^{\!\mathsf{T}}\mathbf{x}_t$ enters the canonical logistic cdf $\sigma(\eta_t)=(1+e^{-\eta_t})^{-1}$ for $\boldsymbol{\beta}=(\beta_0,\ldots,\beta_9)^{\mathsf{T}}$, yielding the eleven-parameter block $\boldsymbol{\theta}=[\alpha\;\boldsymbol{\beta}^{\!\mathsf{T}}]^{\mathsf{T}}$ shown in the Gaussian prior display.

**Gaussian priors in PyMC (vector notation):**

$$
\begin{aligned}
\boldsymbol{\theta} &= (\alpha, \beta_0,\ldots,\beta_9)^{\mathsf{T}}, \\[6pt]
\alpha &\sim \mathcal{N}\bigl(0, (1.5)^2 \bigr), \\[6pt]
\beta_j &\overset{\text{i.i.d.}}{\sim} \mathcal{N}\bigl(0, (0.8)^2 \bigr),
\quad j \in \{0,1,\ldots,9\}.
\end{aligned}
$$

NUTS then targets the joint posterior density $p(\boldsymbol{\theta}\mid\mathcal{D})$ over intercept $\alpha$, slopes $\{\beta_j\}$, and (implicitly) the preprocessing of inputs used to build covariates $\mathbf{x}_t$.

**Baseline:** `sklearn.linear_model.LogisticRegression(C=1.0)` minimizes the same logistic loss on stacked design matrix $\mathbf{X}\in\mathbb{R}^{n\times 10}$ and responses $y_t\in\{0,1\}$, reporting frequentist contrasts to Bayes-drawn $\boldsymbol{\theta}$.

## Environment

- **Python:** project `venv` (`./venv/bin/python`)
- **PyMC:** `pip install 'pymc>=5.16,<6'` (e.g. pymc 5.28.x)
- **Hardware:** Apple Silicon / MPS irrelevant for this script (CPU NUTS)

## Reproducibility commands

```bash
cd /path/to/vip

# Default-ish (full pre-test train, moderate MCMC)
BAYES_DRAWS=2000 BAYES_TUNE=2000 ./venv/bin/python bayes_logit_mcmc_macro_pca.py

# Faster smoke (subsample train)
BAYES_MAX_TRAIN=4000 BAYES_DRAWS=200 BAYES_TUNE=200 ./venv/bin/python bayes_logit_mcmc_macro_pca.py

# Long chain (diminishing returns for forecast metrics; good for ESS / appendix)
BAYES_DRAWS=10000 BAYES_TUNE=10000 ./venv/bin/python bayes_logit_mcmc_macro_pca.py
```

**Env vars:** `BAYES_LOOKBACK`, `BAYES_PCA_COMPONENTS`, `BAYES_TEST_FRAC`, `BAYES_MAX_TRAIN`, `BAYES_DRAWS`, `BAYES_TUNE`, `BAYES_CHAINS`, `BAYES_TARGET_ACCEPT`.

---

## Canonical numbers for `wheat_direction_forecast.tex` (PDF)

**Default `BAYES_DESIGN=wheat_pca` (May~11,~2026 rerun):** `n_train=5401`, holdout `n_test=958`, design dimension **150** ($L{=}30$ flattened PCA scores with $K{=}5$), within-sample PCA EV on the wheat-only block **0.836**, hierarchical slope prior (global $\sigma_\beta$), **0 NUTS divergences**, $\hat R\approx 1$.

| Role | sklearn `LogisticRegression(C=1)` test | Bayes posterior-mean $\bar p$ test | Bayes posterior-mean train |
|------|----------------------------------------|-----------------------------------|-----------------------------|
| Typographic Table (`tab:bayes`, 3 decimals) | **0.500 / 0.509 / 0.255** | **0.542 / 0.508 / 0.249** | acc **0.520**, Brier **0.249** |
| Representative 10k/10k $\times$2 chains (terminal log) | acc 0.5000, AUC 0.5086, Brier 0.2547 | acc 0.5418, AUC 0.5080, Brier 0.2487 | acc 0.5195, Brier 0.2492 |

*Posterior mean* here is **`posterior_mean_probs`**: average of `sigmoid(α_s + X β_s)` across draws (see Appendix K in the PDF), not `sigmoid` of averaged coefficients.

**Legacy `BAYES_DESIGN=fred_macro` (archived May~7 sweep, dim 10):** MLE and Bayes both near **AUC 0.528** / **Brier 0.251** with **coin-flip accuracy** at 0.5---historical rows below retained for reproducibility only.

---

### Legacy table: `fred_macro` frozen May~7,~2026

| Role | sklearn test | Bayes posterior-mean test | Bayes train (in-sample) |
|------|-------------|---------------------------|-------------------------|
| 10-feature design, `n_train=5413`, `n_test=956` | acc 0.5000, AUC 0.5284, Brier 0.2510 | acc 0.4979, AUC 0.5286, Brier 0.2510 | acc 0.5282, Brier 0.2487 |

Three-decimal `tab:bayes` *before* the wheat-default switch: MLE **0.500 / 0.528 / 0.251**; Bayes **0.498 / 0.529 / 0.251**.

---

## Run history (chronological lab notes)

| Date (local) | `BAYES_MAX_TRAIN` | tune / draw / chains | Train *n* | Test *n* | Sklearn test (acc, AUC, Brier) | Bayes mean *p* test | Notes |
|--------------|-------------------|----------------------|-----------|----------|--------------------------------|---------------------|-------|
| 2026-05-07 | 4000 | 200 / 200 / 2 | 4000 | 956 | 0.491, 0.514, 0.252 | 0.489, 0.515, 0.252 | Subsampled train |
| 2026-05-07 | — (full) | 600 / 600 / 2 | 5413 | 956 | 0.500, 0.528, 0.251 | 0.498, 0.529, 0.251 | Legacy row; superseded by verification sweep below |
| 2026-05-07 | — | **2000 / 2000 / 2** | 5413 | 956 | **0.5000, 0.5284, 0.2510** | **0.4990, 0.5283, 0.2510** | Train in-sample Bayes: acc **0.5284**, Brier **0.2487**; ~11 s NUTS; 0 divergences |
| 2026-05-07 | — | **6000 / 6000 / 2** | 5413 | 956 | **0.5000, 0.5284, 0.2510** | **0.5000, 0.5284, 0.2510** | Train in-sample: acc **0.5278**, Brier **0.2487**; ~30 s |
| 2026-05-07 | — | **10000 / 10000 / 2** | 5413 | 956 | **0.5000, 0.5284, 0.2510** | **0.4979, 0.5286, 0.2510** | Train in-sample: acc **0.5282**, Brier **0.2487**; ~50\,s |

**Interpretation.** Holdout **accuracy** for Bayes posterior-mean classifications moves by **≤1** ppt across MCMC budgets because probabilities sit near the **0.5** threshold; **AUC** and **Brier** are the stable summary (see Appendix K chain-length table in the PDF).

**PCA macro variance explained (train):** **0.630** on full train (slight variation when `BAYES_MAX_TRAIN` subsamples).

### Convergence snapshot (representative: 2000 / 2000, full train)

- **Divergences:** 0  
- **R-hat (Gelman–Rubin):** $\hat{R}\approx 1$ separately for intercept $\alpha$ and each logistic slope $\beta_j$ (reported under `beta[j]` in PyMC summaries)  
- **Bulk ESS:** on the order of **10^3** to **10^4** (increases with more draws)  
- **ArviZ `summary`:** printed after sampling (`alpha`, `beta[0]` … `beta[9]`); indices 0–4 = return block; 5–9 = macro PCA block.

### Diminishing returns (longer chains)

Increasing tuning and draws **per chain** boosts the minimum **bulk ESS** among $\bigl(\alpha,\beta_{0},\ldots,\beta_{9}\bigr)$ almost linearly in wall-clock CPU time, yet **ROC-AUC** and **Brier** anchored on posterior-mean logistic forecasts barely budge—the limitation is empirical signal in $(\mathcal{D},\mathbf{X})$, not additional posterior precision.

| Tune + draw / chain | Wall (approx.) | Min ESS_bulk | Acc_test (PM) | AUC_test (PM) | Brier_test (PM) | Acc_train (PM) | Brier_train (PM) |
|---------------------|----------------|--------------|---------------|---------------|-----------------|----------------|------------------|
| 2000 + 2000 | ~17 s | ~2.5e3 | **0.5418** | **0.508** (→0.508 in PDF) | **0.2487** | **0.5195** | **0.2493** |
| 6000 + 6000 | ~52 s | ~8.7e3 | **0.5418** | **0.508** | **0.2487** | **0.5195** | **0.2492** |
| 10000 + 10000 | ~93 s | ~2.0e4 | **0.5418** | **0.508** | **0.2487** | **0.5195** | **0.2492** |

*wheat\_pca* default; sklearn test metrics stable near **acc 0.499--0.500**, **AUC 0.5086**, **Brier 0.2547** (see `tab:bayes` in PDF for rounded headline).

### Takeaway

- **`tab:bayes` headline metrics** mirror **10000 tune + 10000 draw** per chain (May 2026 verification sweep); **2000** / **6000** runs match three-decimal **AUC** / **Brier** and vary by at most **1** ppt on holdout accuracy (threshold at 0.5).
- **More draws** keep raising **ESS** / lowering **MCSE** roughly linearly in sampling time even when holdout **AUC/Brier** barely move.
- **Posterior mean** predictive probabilities match **penalized MLE logit** at printed precision → weak priors + small linear model.  
- **Next semester:** retain draws $\boldsymbol{\theta}^{(m)}\sim p(\boldsymbol{\theta}\mid\mathcal{D})$ so decisions reflect ensembles mapped through $\sigma(\alpha+\boldsymbol{\beta}^{\!\mathsf{T}}\mathbf{x}_t)$—not only collapsing uncertainty to posterior means $\mathbb{E}[\boldsymbol{\theta}\mid\mathcal{D}]$.

---

## Files

| File | Role |
|------|------|
| `bayes_logit_mcmc_macro_pca.py` | Pipeline + PyMC + ArviZ summary |
| `bayes_mcmc_wheat_run_log.md` | This log |
| LaTeX manuscript under **`docs/`** (`wheat_direction_forecast.tex`) | Manuscript source |
| Built PDF **`docs/wheat_direction_forecast.pdf`** (via **`docs/ieee/build_report.sh`**) | Output |
| **`docs/figs/*.png`** (Bayes script) | MCMC visuals (ROC, calibration, predictive SD histogram, coefficient forest) |

Regenerate PDF with **`bash docs/ieee/build_report.sh`** (from repo root); **`docs/wheat_direction_forecast.pdf`**; **`.tex`** and **`.aux`/`.log`** stay under **`docs/ieee/`**.

After running `bayes_logit_mcmc_macro_pca.py`, set `BAYES_SAVE_FIG=1` (default) to refresh those PNGs; set `BAYES_SAVE_FIG=0` to skip writing figures.
