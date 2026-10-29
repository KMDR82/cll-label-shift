# Label shift under complementary-label supervision

Code, results and figure data for the manuscript

> A. Akkaya, *Correcting label shift without observed labels: identifiability, cost and an
> applicability test for complementary-label learning* (submitted to Knowledge-Based Systems).

A predictor trained from complementary labels (labels that name a class an instance does
**not** belong to) has no ordinary labels with which to adapt to a change in class
frequencies. The repository contains everything needed to reproduce the study of how the
importance weights of label shift can be estimated in that setting:

* plain CL-BBSE, the direct inversion `w = Q C̄⁻¹ q_te`;
* the naive estimator `q_te / p_tr` and its one-parameter correction CL-SC, whose accuracy
  parameter is identified from complementary labels as `a = 1 − (K − 1) tr(C̄)`;
* the label-free misfit diagnostic `D̂` and the rule that uses CL-SC when `D̂ < τ`;
* MLLS (EM) with a temperature fitted from complementary labels, and BBSE with clean labels
  as a reference.

## Repository layout

| Path | Content |
|---|---|
| `notebooks/cll_label_shift_full.ipynb` | The complete study (Part A: training, Part B: evaluation, statistics, tables, figures). |
| `notebooks/manuscript_figures.ipynb` | One cell that re-draws every figure of the paper from `results/`. |
| `src/cll_label_shift_full.py` | The same notebook in percent format (`# %%` cells); `to_ipynb.py` converts it. |
| `src/fig_cell.py`, `src/regen_figures.py` | Figure code of the notebook and a command-line runner that uses only the CSV files. |
| `src/redraw_s7.py` | Re-draws Figure S7 from the curves stored in `figures/supplementary/Figure_S7_iw_risk.csv`. |
| `src/make_figure1_drawio.py` | Generates the draw.io source of Figure 1. |
| `results/` | All result files of the run reported in the paper (see below). |
| `figures/main/`, `figures/supplementary/` | Figures as PDF, each with a CSV of the plotted values; `Figure_1.drawio` is editable. |

## Results files

| File | Content |
|---|---|
| `manifest.csv` | The 142 trained predictors: group, dataset, seed, accuracy, rank check, time. |
| `E1_cells.csv` | Main comparison: one row per predictor, fold and shift, with every estimator and ablation. |
| `E2_cells.csv` | Sample-size sweep. |
| `E3_cells.csv` | Input-noise (accuracy) sweep. |
| `E4a_cells.csv`, `E4bc_cells.csv` | Transition-matrix experiments (training under biased `Q`; redrawn labels; misspecified `Q̂`). |
| `E6_montecarlo.csv` | Simulation of Theorem 3 and Corollary 2. |
| `E7_cells.csv` | Official test pool as target. |
| `E8_prop1_*.csv` | Two-class counterexample of the appendix. |
| `E9_collapse.csv` | EMNIST-Balanced and end-to-end ResNet-18 runs. |
| `E10_*.csv` | Applicability rule: leave-one-dataset-out, tests, out-of-sample evaluation. |
| `table*.csv`, `cd_ranks.csv`, `friedman_nemenyi.json`, `calibration.csv`, `ablation_iw.csv` | Tables and statistics. |
| `summary.json` | Every number quoted in the text. |

The table and figure numbers in the file names of `results/` follow the notebook; the
figures in `figures/` carry the numbers used in the paper.

## Reproducing the study

The notebook was run on Kaggle with one NVIDIA Tesla T4 GPU (Python 3.13, PyTorch 2.11,
NumPy 2.1). Set `PROFILE` in cell 1:

* `fake`: synthetic data, no download, a pipeline check on CPU;
* `smoke`: every loader and experiment once with 2 epochs, prints a time projection;
* `full`: the study reported in the paper (142 predictors, 109 minutes of training, about
  0.5 hours for Part B).

Part A writes `output/cache/*.npz` and `partA_cache.zip`; Part B reads only the files listed in
`output/results/manifest.csv` and writes `results_partB.zip`. A run that times out can be
resumed: attach the previous output as an input and finished predictors are not retrained.
Datasets are downloaded by `torchvision` (MNIST, Fashion-MNIST, KMNIST, EMNIST, CIFAR-10,
SVHN; SVHN falls back to the Hugging Face copy `ufldl-stanford/svhn`) and by
`sklearn.datasets.fetch_covtype` (Covertype).

### Figures only

```bash
pip install -r requirements.txt
python src/regen_figures.py results figures_out
```

or run `notebooks/manuscript_figures.ipynb` next to a `results/` (or Kaggle `output/results`)
folder. The plotted values are written to a CSV next to each figure.

## Notes on the reported run

* EMNIST-Letters (K = 26): no predictor passes the rank check of the complementary confusion
  matrix; the dataset is reported but not evaluated.
* 111 of the 142 predictors pass the check; failed runs are listed in `manifest.csv` and are
  never replaced.
* Source and target are drawn from the held-out half of the training pool (cross-fitted), so
  that they share `p(x | y)`; the official test pools are used only in `E7`.

## Citation

Please cite the article and this repository; `CITATION.cff` holds the software record.

## License

MIT, see `LICENSE`. The datasets keep their own licences.
