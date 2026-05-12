# Fall 2026 VIP — hand-in package

Revised **ARX**, **ELM**, **FAVAR**, and **TCN** pipelines (Python + Jupyter), **PCA-augmented TCN** scripts, an **exploratory Bayesian logistic (MCMC)** script, a preliminary **long-flat investment-strategy proxy**, final report/slides assets, bundled figures, and memos. This folder sits **inside** the repository tree below and is the Fall 2026 submission bundle.

---

## Repository

Top-level layout **of this repository** (paths are relative to the repository root—the directory that contains **`data/`**, **`requirements.txt`**, **`requirements-handin.txt`**, and this README):

| Path | Role |
|------|------|
| `ARX/`, `ELM/`, `FAVAR/`, `TCN/` | Main `*_Revised.py` training scripts; optional **`TCN/TCN_PCA*.py`**, **`exploratory/bayes_logit_mcmc_macro_pca.py`**, and **`figs/`** (may be symlinks into this hand-in bundle). |
| Hand-in bundle (this README’s directory) | Per-model **`*.ipynb`**, memos, logs — the **`README.md`** you are reading lives here. |
| `data/` | Ingestion utilities (`download_wheat_futures.py`, `download_fred_md.py`, `process_fred_md_data.py`) plus `data/wheat-futures/`, `data/fred-md/`, … — **full clone** includes downloaded files; the **hand-in** copy of **`data/`** (next to this README) ships **only scripts + docs + empty folders** and **`README_NO_DATA.txt`** placeholders (**no** CSV/NPY/ZIP in the bundle). |
| `Decomposition/` | EMD / decomposition experiments and lab notes — **same folder name at the repository root and duplicated here** in the hand-in bundle so the zip matches the root layout. |
| `exploratory/` | Optional scratch scripts (e.g. VIX panel, `VIP_Abstract_Class.ipynb`) — **same name at the repository root and duplicated here**; **`bayes_logit_mcmc_macro_pca.py`** here is a symlink to **`../bayes_logit_mcmc_macro_pca.py`**. |
| `docs/` | LaTeX/IEEE manuscript and **`build_report.sh`** — **canonical tree at the repository root**; **duplicate** next to this README for the hand-in bundle. |

**Working directory for commands below:** the **package directory** that contains **`ARX/`**, **`TCN/`**, and this **`README.md`** (see **Appendix B** in `docs/ieee/wheat_direction_forecast.tex` if your checkout nests an extra parent folder).

---

## Folder layout: repository root vs. this bundle

| Location | What it is | Typical contents |
|----------|------------|------------------|
| **Repository root** (`ARX/`, `ELM/`, `FAVAR/`, `TCN/`) | **Canonical CLI runners** — `*_Revised.py` used for training from the clone root. | Root **`ARX/`** has **no** course Jupyter notebooks; root **`TCN/`** may include extra lab notebooks. |
| **This folder** | **Hand-in copy** of the same four models, **plus** course-specific extras. | Under **`ARX/`**, **`ELM/`**, **`FAVAR/`**, **`TCN/`** here: **both** `*_Revised.py` **and** `*_Revised.ipynb` for grading. **`Decomposition/`**, **`exploratory/`**, **`data/`**, and **`docs/`** mirror the **repository-root** layout; **`data/`** here has **no** real datasets; **`docs/`** is a full copy for submission. Also: `TCN_PCA*.py`, `bayes_logit_mcmc_macro_pca.py`, `figs/`, `MEMO.md`, `LAB_REPORT_LOG.md`, `convert_to_ipynb.py`, `bayes_mcmc_wheat_run_log.md`. |
| **Symlinks at repo root** (optional) | Same files, stable short paths. | e.g. **`TCN/TCN_PCA.py`**, **`exploratory/bayes_logit_mcmc_macro_pca.py`**, **`figs/`** may be links; **`python`** still resolves them as normal entry points. |

Use the **repository-root** paths in sections 5–6 for runs unless your grader specifies otherwise.

---

## Contents (paths from repository root)

| Path | Role |
|------|------|
| **`requirements-handin.txt`** (repo root) | **Hand-in dependencies** — models, PCA, PyMC/ArviZ (Bayes), **`nbformat`** (`convert_to_ipynb`). **`pip install -r requirements-handin.txt`** from the repository root. A **`requirements.txt`** copy also ships **next to this README** for grading bundles. |
| `ARX/ARX_Revised.py`, … | Four primary revised trainers (mirrored **here** with **`*.ipynb`** for hand-in). |
| `TCN/TCN_PCA.py`, `TCN/TCN_PCA_WHEAT_ONLY.py` | PCA + TCN extensions — run from repo root as in §6. |
| `exploratory/bayes_logit_mcmc_macro_pca.py` | Bayesian logistic (PyMC). |
| `figs/` | PNG diagnostics (repo root; may be a symlink). |
| `scripts/download_public_data.sh` | One-shot wheat + FRED-MD download from repo root. |
| `docs/wheat_direction_forecast.pdf` | Final IEEE-style report PDF. |
| `docs/slides/` | Final presentation deck variants and speaker notes. Treat **`.pptx` files as manually edited artifacts**; do not regenerate unless intentionally rebuilding from a known script/template. |
| **Hand-in only** (this folder) | **`*.ipynb`** per model, **`MEMO.md`**, **`LAB_REPORT_LOG.md`**, **`convert_to_ipynb.py`**, **`bayes_mcmc_wheat_run_log.md`**, duplicate **`Decomposition/`**, **`exploratory/`**, **`data/`** (scripts only), **`docs/`** (manuscript mirror). |

---

## 1. Environment

After **cloning this repository** to your machine, from the **repository root**:

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements-handin.txt
```

**`requirements-handin.txt`** lists everything for this bundle: **core stack** (including **`seaborn`** used by the revised runners), **`nbformat`** for **`convert_to_ipynb.py`**, and **`pymc`** / **`arviz`** for the Bayesian script. The repository root **`requirements.txt`** is slimmer (four models + **data/** only, no Bayes).

If **`torch`** does not install, add a **PyTorch** wheel for your OS/GPU using the official PyTorch install flow, then run **`pip install -r requirements-handin.txt`** again.

**Bayes tuning:** **`BAYES_DRAWS`**, **`BAYES_TUNE`**, **`BAYES_MAX_TRAIN`**, **`BAYES_SAVE_FIG`** are documented in **`bayes_logit_mcmc_macro_pca.py`**.

### Optional: Jupyter for bundled **`*.ipynb`**

```bash
pip install ipykernel jupyter
```

---

## 2. Data policy (summary)

| Source | Role |
|--------|------|
| **Yahoo Finance** via **`yfinance`** | Daily **Chicago SRW wheat** continuous contract **`ZW=F`**. |
| **FRED-MD** (McCracken & Ng, St. Louis Fed) | Macroeconomic panel; vintage ZIPs/CSVs under **`data/fred-md/`**. |

**Preprocessing (course-style):** **31** FRED‑MD series as in **`SELECTED_FEATURES`** in the revised code, **t‑codes**, **~1‑month publication shift** (no same‑day lookahead), **forward‑fill** monthly to trading days **after** that shift. Details appear in **`data/download_fred_md.py`**, **`data/process_fred_md_data.py`**, and **`data/README_FRED_MD.md`**.

**Large generated arrays:** precomputed tensors such as **`X_train.npy` / `X_val.npy`** are usually **gitignored** (>100 MB). Regenerate them with this project’s preprocessing or training scripts after cloning.

---

## 3. Download into **`data/`**

```bash
./venv/bin/python data/download_wheat_futures.py
./venv/bin/python data/download_fred_md.py
```

Or:

```bash
bash scripts/download_public_data.sh
```

**Outputs:** **`data/wheat-futures/wheat_futures_daily.csv`** (and related outputs); **`data/fred-md/`** as produced by **`download_fred_md.py`**. If Fed filenames change, update URLs inside **`data/download_fred_md.py`**.

**Manual fallback:** follow the CSV/schema notes inside **`data/download_wheat_futures.py`** and **`data/README_FRED_MD.md`**.

---

## 4. Preprocess / align

```bash
./venv/bin/python data/process_fred_md_data.py
```

Large intermediates may land in **`data/processed/`** (often gitignored).

---

## 5. Run the four primary models

From repository root:

```bash
./venv/bin/python ARX/ARX_Revised.py     --mode both
./venv/bin/python ELM/ELM_Revised.py     --mode both
./venv/bin/python FAVAR/FAVAR_Revised.py --mode both
./venv/bin/python TCN/TCN_Revised.py     --mode both
```

**Shared conventions**

| Item | Convention |
|------|------------|
| Wheat | **`ZW=F`** close |
| Macro | FRED‑MD selection + t‑codes + **~1‑month shift** |
| Window | **30‑day** lookback where the revised TCN path uses sequences |
| CV | **5‑fold** time‑series split on the training tail |
| Scaling | **StandardScaler** fit **only** on training folds |

---

## 6. Optional extensions

**PCA + TCN**

Both scripts (**`TCN_PCA_WHEAT_ONLY.py`** and **`TCN_PCA.py`** at package root, or symlinked **`TCN/TCN_PCA*.py`** for short paths) evaluate **five-fold time-series CV on the train+val slice**, then (**by default**) retrain once and print **terminal holdout metrics** reserving the **same 15% tail** chronology used in **`TCN_Revised`**. Tune **`TCN_LR`**, **`TCN_EPOCHS`**, **`TCN_CHANNELS`** (comma list), **`TCN_HOLDOUT_FRAC`** (default **`0.15`**; **`0`** = omit holdout / CV-on-full-sample legacy), and **`TCN_PATIENCE`** (early-stopping epochs without val-loss improvement; default **`20`**, e.g. **`TCN_PATIENCE=10`** to match older runs).

```bash
# Example replay; ensure cwd matches Appendix B hierarchy in IEEE PDF sources.
TCN_LR=1e-6 TCN_EPOCHS=50 TCN_CHANNELS=128,64 ./venv/bin/python TCN/TCN_PCA_WHEAT_ONLY.py
TCN_LR=1e-6 TCN_EPOCHS=50 TCN_CHANNELS=128,64 ./venv/bin/python TCN/TCN_PCA.py
```

**Verification:** rebuild **`docs/wheat_direction_forecast.pdf`** (**`bash docs/ieee/build_report.sh`**); in **§Alternative Data** locate the holdout PCA table; follow **Appendix B** verbatim shell (**verifier checklist** there). Prerequisites: **`pip install -r requirements-handin.txt`**, populated **`data/wheat-futures`** + **`data/fred-md`**, working directory aligned so each **`TCN_PCA*.py`** resolves to **`data/`** at its grandparent (same rule as Appendix B prose), then export **`TCN_LR` / `TCN_EPOCHS` / `TCN_CHANNELS`** exactly as printed in that table caption and diff against **Holdout test** stdout.

`TCN_PCA.py` pulls cross‑asset Yahoo series via **`yfinance`** (**network required**).

**Result highlight (Fall 2026 MEMO — May replay, with corn/soy/UUP/CAD daily returns).**
**`TCN_PCA_WHEAT_ONLY.py`** (**7** wheat-engineered features **+** **four** Yahoo **1-day log-returns**: **ZC=F**, **ZS=F**, **UUP**, **CAD=X**, calendar-aligned to wheat) reaches **terminal holdout accuracy ~0.543 @0.5**, **just above** **`TCN_Revised` ~0.541** on the same chronology. **`TCN_PCA.py`** (**24** channels per bar: **7** wheat **+** **12** Yahoo overlays **+** **5** macro PCs, where macro PCA includes **FRED-MD** **plus** those **same four** daily returns) **underperforms** on **that** tail in our replay (**~0.482 @0.5**). Asset choice and the **1-day log-return** form are documented in **`MEMO.md`** and **§Alternative Data** in **`docs/ieee/wheat_direction_forecast.tex`**; ROC-AUC stays **near 0.5**—see **`MEMO.md`** for CV/holdout tables and caveats.

**Bayesian logistic**

```bash
./venv/bin/python bayes_logit_mcmc_macro_pca.py
```

Defaults to **`BAYES_DESIGN=wheat_pca`** (same scaler+PCA + `create_target` alignment as **`TCN_PCA_WHEAT_ONLY`**, flattened windows for the linear logit). Use **`BAYES_DESIGN=fred_macro`** for the legacy FRED + lag-return ten-feature block. Bayesian diagnostics go to **`docs/figs/`** unless **`BAYES_SAVE_FIG=0`**.

**Investment strategy (long–flat) + holdout validation**

Course deliverable for **tactical positioning**: **`scripts/wheat_strategy_holdout_validation.py`** maps **`TCN_PCA_WHEAT_ONLY`** hold-out **`predict_proba`** into **long–flat** weights versus **buy-and-hold** on **realized** next-day **simple returns** (same horizon as **`create_target`**). This semester’s run is **intentionally minimal** (B&H vs long–flat only): **no** margin, listed options, full commission/roll/slippage stack—optional **`STRATEGY_COST_BPS`** is a toy turnover knob only; a **follow-on term** can pair **model finetuning** with **closer-to-live transaction** assumptions. Scaler + PCA + TCN fit **train+val only**; the **terminal 15%** sequence slice is the default test tail (**`HOLDOUT_FRAC`**, aligned with **`TCN_HOLDOUT_FRAC`**). Rules: **`STRATEGY_HURDLE`** (default **0.52**), **P ≥ 0.5**, **P ≥ TCN validation threshold**, **momentum**, plus optional **`STRATEGY_COST_BPS`**. Optional **`STRATEGY_PANEL=both`** adds the **`TCN_Revised`** **(30×37)** panel + **sklearn** logistic (needs **`data/fred-md`**). **`STRATEGY_PANEL=bayes`** or **`tcn_bayes`** runs the same long–flat rules on **posterior-mean** probabilities from **`bayes_logit_mcmc_macro_pca.py`** (PyMC + ArviZ required); tune **`STRATEGY_BAYES_DRAWS`** / **`STRATEGY_BAYES_TUNE`** for speed. Wheat path: **`WHEAT_CSV`** or **`data/wheat-futures/`** here **or** parent **`../data/wheat-futures/`** (Yahoo required for cross-asset returns).

```bash
export WHEAT_CSV="${WHEAT_CSV:-../data/wheat-futures/wheat_futures_daily.csv}"
./venv/bin/python scripts/wheat_strategy_holdout_validation.py
STRATEGY_PANEL=both STRATEGY_HURDLE=0.55 STRATEGY_COST_BPS=5 ./venv/bin/python scripts/wheat_strategy_holdout_validation.py
```

---

## 7. Write-up inside the repository

LaTeX lives under **`docs/ieee/`**; **`bash docs/ieee/build_report.sh`** must complete **after** any edit to **`wheat_direction_forecast.tex`**. The refreshed PDF lands as **`docs/wheat_direction_forecast.pdf`** (**always reopen or close-and-re-open** that file—the IDE preview often caches stale bytes). **Aux/log** remain under **`docs/ieee/`**. Figures pull from **`docs/figs/`** (Bayesian) and **`figs/`** (model composites) as set in **`wheat_direction_forecast.tex`**.

### Final presentation

Presentation materials live under **`docs/slides/`**. The current manually edited deck is:

- **`docs/slides/vip_spring_2026_final_kf2623.pptx`**
- Speaker notes: **`docs/slides/vip_final_5_slides_template_speaker_notes.md`**

PowerPoint files in this folder are treated as **manual deliverables**. If they are opened and edited in PowerPoint, do not regenerate them from helper scripts unless you first save a separate backup copy.

---

## 8. Create a standalone `vip_fall2026` repository and push

If you want this folder to become its **own GitHub repository**, use the commands below from your local machine. Replace `YOUR_GITHUB_USER` and the remote URL with your actual GitHub account/repository.

### Option A: GitHub CLI (`gh`) recommended

```bash
cd /Users/kfunaki/Projects/vip/vip_fall2026
git init
git status
git add .
git commit -m "Initial VIP Fall 2026 hand-in package"
gh repo create vip_fall2026 --private --source=. --remote=origin --push
```

Use `--public` instead of `--private` only if you are comfortable making the coursework repository public.

### Option B: Create the GitHub repository in the browser first

1. Create a new empty repository on GitHub named **`vip_fall2026`**.
2. Do **not** initialize it with a README, license, or `.gitignore` if this local folder already has those files.
3. Run:

```bash
cd /Users/kfunaki/Projects/vip/vip_fall2026
git init
git status
git add .
git commit -m "Initial VIP Fall 2026 hand-in package"
git branch -M main
git remote add origin git@github.com:YOUR_GITHUB_USER/vip_fall2026.git
git push -u origin main
```

If you use HTTPS instead of SSH, use:

```bash
git remote add origin https://github.com/YOUR_GITHUB_USER/vip_fall2026.git
```

### Before pushing

- Check **`git status`** and confirm that no private raw datasets, credentials, or oversized generated arrays are staged.
- Keep real downloaded data out of Git unless the instructor explicitly requested it and redistribution is allowed.
- The bundled placeholder files under **`data/*/README_NO_DATA.txt`** are safe to commit.

---

## 9. Reproducibility

- **Python ≥ 3.10** recommended; use script flags for seeds where available.
- Run **downloads → preprocessing → training** in order; do **not** refit scalers on the full sample before splitting inside each script’s train/validation logic.
- **GPU vs CPU** (e.g. Apple MPS) can change metrics slightly.

---

## 10. Ethics, data compliance, references, license

Academic / VIP coursework only — **not** investment advice or a production trading system.

**Reuse public data under their terms** (St. Louis Fed, Yahoo / **`yfinance`**). This work is for coursework and reproducibility only.

### Data & policy

1. M. W. McCracken and S. Ng, “FRED-MD: A monthly database for macroeconomic research,” *Journal of Business & Economic Statistics*, vol. 34, no. 4, pp. 574–589, 2016. DOI 10.1080/07350015.2015.1086654  
2. Federal Reserve Bank of St. Louis, **FRED-MD** database and documentation (see St. Louis Fed economic databases).  
3. **Yahoo Finance** **`ZW=F`** via **`yfinance`** — comply with Yahoo/API terms when redistributing or citing.

### Core methods (papers)

4. B. S. Bernanke, J. Boivin, and P. Eliasz, “Measuring the effects of monetary policy: A factor-augmented vector autoregressive (FAVAR) approach,” *Quarterly Journal of Economics*, vol. 120, no. 1, pp. 387–422, 2005.  
5. S. Bai, J. Z. Kolter, and V. Koltun, “An empirical evaluation of generic convolutional and recurrent networks for sequence modeling,” arXiv:1803.01271, 2018.  
6. G.-B. Huang, Q.-Y. Zhu, and C.-K. Siew, “Extreme learning machine: Theory and applications,” *Neurocomputing*, vol. 70, pp. 489–501, 2006.

### Tutorials & docs (non-exhaustive)

7. Statsmodels, Vector Autoregression (VAR).  
8. J. Stock and M. Watson, “Dynamic Factor Models,” *Handbook of Macroeconomics*, vol. 2A.  
9. PyTorch, **`Conv1d`** documentation.
