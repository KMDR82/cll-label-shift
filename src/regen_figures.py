"""Re-draws the manuscript figures from the CSV files of a finished Part B run, with exactly
the figure cell of the notebook (fig_cell.py). Usage: python regen_figures.py RESULTS_DIR OUT_DIR"""
import sys, os, csv, json, zlib, math
import numpy as np
RES, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
DIRS = {'fig': OUT}
def crc(*parts): return zlib.crc32('|'.join(map(str, parts)).encode())
def rng_for(*parts): return np.random.default_rng(crc(*parts))
def write_csv(path, rows):
    rows = list(rows)
    if not rows: open(path, 'w').close(); return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)
def conv(v):
    if v == '': return np.nan
    try: return float(v)
    except ValueError: return {'True': True, 'False': False}.get(v, v)
def load(name): return [{k: conv(v) for k, v in r.items()} for r in csv.DictReader(open(os.path.join(RES, name)))]
N_BOOT = 10000
def boot_ci(x, B=None):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 2: return (np.nan, np.nan)
    r = rng_for('boot', len(x)); m = x[r.integers(0, len(x), (B or N_BOOT, len(x)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
MAIN_DATASETS = ['mnist', 'fashion', 'kmnist', 'emnist_letters', 'cifar10_feat', 'svhn_feat', 'covertype']
DATASET_SHORT = {'mnist': 'MNIST', 'fashion': 'F-MNIST', 'kmnist': 'KMNIST', 'emnist_letters': 'EMNIST-L',
                 'emnist_balanced': 'EMNIST-B', 'cifar10_feat': 'CIFAR-10', 'svhn_feat': 'SVHN',
                 'cifar10_raw': 'CIFAR-10 raw', 'covertype': 'Covertype'}
TWEAK_RHOS, DIR_ALPHAS = [3.0, 6.0, 10.0], [0.3, 1.0, 3.0]
NSWEEP_SRC, NSWEEP_TE = [1000, 2000, 5000, 10000, None], [500, 1000, 2000, 5000]
QTRAIN_KINDS, QSENS_EPS = ['mild', 'strong'], [0.0, 0.1, 0.2, 0.3]
RANK_METHODS = ['naive', 'clsc', 'plain', 'mlls']
DS_METHODS = ['none', 'naive', 'clsc', 'plain', 'mlls', 'oracle']
E1, E2, E3 = load('E1_cells.csv'), load('E2_cells.csv'), load('E3_cells.csv')
E4a, E4bc, E7 = load('E4a_cells.csv'), load('E4bc_cells.csv'), load('E7_cells.csv')
CAL, LODO, E10T = load('calibration.csv'), load('E10_lodo_runs.csv'), load('E10_lodo_table.csv')
P1R, P1C = load('E8_prop1_excess.csv'), load('E8_prop1_curves.csv')
tgrid = np.array([r['t'] for r in P1C])
t1 = list({r['dataset']: r for r in load('table1_datasets.csv')}.values())
fn = json.load(open(os.path.join(RES, 'friedman_nemenyi.json')))
CD = fn['CD']; CDROWS = fn['avg_ranks']
avg_rank = np.array([{r['method']: r['avg_rank'] for r in CDROWS}[m] for m in RANK_METHODS])
TAU_STAR = json.load(open(os.path.join(RES, 'summary.json')))['rule']['tau_star']
main_cells = [c for c in E1 if np.isfinite(c.get('D_hat', np.nan))]
IW = []          # Figure_S7 needs the cached risk curves; it is taken from the notebook run as it is
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig_cell.py')).read().replace(
    "import matplotlib; matplotlib.use('Agg')", "import matplotlib; matplotlib.use('Agg')"))
