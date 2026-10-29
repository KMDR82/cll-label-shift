# %% [markdown]
# # Label shift under complementary-label supervision: full experimental notebook
#
# This notebook produces every number, table and figure of the manuscript from scratch.
#
# * **Part A (GPU)** trains the complementary-label predictors and caches their outputs
#   (logits, true labels, complementary labels). It is resumable: a finished run whose
#   configuration signature matches is never retrained, and a run with a different signature
#   is never reused.
# * **Part B (CPU only)** computes all estimators, statistics, tables and figures from the
#   cache. It can be re-run without a GPU, and it is what the public repository reproduces.
#
# **Profiles** (cell 1): `fake` runs the whole notebook on small synthetic data with no
# download (a pipeline check, about 10 minutes on CPU); `smoke` runs every dataset loader and
# every experiment once on real data with 2 epochs and prints a measured time projection for
# `full` (about 10 minutes on a T4); `full` is the study. Measured on Kaggle with one Tesla T4:
# 109 minutes of training for the 142 predictors in Part A (plus dataset downloads and the
# one-off ResNet-18 feature extraction), and about 0.5 hours for Part B.
#
# **Run order.** Cells 1 to 6 define everything. Cell 7 runs Part A, cell 8 packages it.
# Cells 9 onwards are Part B. After a timeout, re-run cells 1 to 6 and then cell 7; finished
# runs are skipped.
#
# Numbers printed under `fake` or `smoke` are pipeline checks and are not results.

# %% [markdown]
# ## 1. Configuration
#
# Every constant of the study is set here and nowhere else. `CFG_SIG` is stamped into every
# cached file; changing an item that affects training changes the signature and forces a
# retrain of the affected runs.

# %%
PROFILE = 'full'                      # 'fake' | 'smoke' | 'full'

# ---------------- Part A: which predictors are trained ----------------
MAIN_DATASETS   = ['mnist', 'fashion', 'kmnist', 'emnist_letters',
                   'cifar10_feat', 'svhn_feat', 'covertype']
SEEDS_MAIN      = list(range(10))
NOISE_DATASETS  = ['fashion', 'kmnist', 'mnist']       # E3: accuracy sweep by input noise
NOISE_LEVELS    = [0.1, 0.2, 0.3, 0.4, 0.5]           # level 0 is taken from the main runs
                                                      # (levels 1.0-2.5 of the first full run left almost no healthy predictor)
SEEDS_NOISE     = [0, 1, 2]
QTRAIN_DATASETS = ['fashion', 'kmnist']                # E4a: predictors trained under biased Q
QTRAIN_KINDS    = ['mild', 'strong']
SEEDS_QTRAIN    = [0, 1, 2, 3, 4]
IW_DATASET, SEEDS_IW, IW_RHO, IW_CLASS = 'fashion', [0, 1, 2], 6.0, 0   # ablation (f)
COLLAPSE_RUNS   = [('emnist_balanced', s) for s in (0, 1, 2)]
RUN_CIFAR_RAW   = True                                 # E9: ResNet-18 on raw CIFAR-10, 1 seed
if RUN_CIFAR_RAW: COLLAPSE_RUNS.append(('cifar10_raw', 0))

EPOCHS = {'default': 60, 'emnist_letters': 100, 'emnist_balanced': 200, 'cifar10_raw': 60}
LR     = {'default': 1e-3, 'emnist_balanced': 1e-4, 'cifar10_raw': 1e-3}
BATCH  = 256
VAL_FRAC = 0.5                        # held-out complementary source split
COVTYPE_PER_CLASS = 15000            # covertype: at most this many rows per class (rarest class: 2747)
TRAIN_SUBSET_CACHE = 10000            # in-sample outputs kept for ablation (d)
Q_STRENGTH = {'mild': 0.5, 'strong': 1.2}
TIME_BUDGET_H = 11.0

# ---------------- Part B: how the cache is analysed ----------------
N_REP        = 40                     # resamples per (run, shift, half)
TWEAK_RHOS   = [3.0, 6.0, 10.0]       # tweak-one: boosted class rotates over all K classes
DIR_ALPHAS   = [0.3, 1.0, 3.0]        # Dirichlet target priors
N_TE_MAIN    = 5000                   # target sample size (capped at half the target pool)
NSWEEP_SRC   = [1000, 2000, 5000, 10000, None]   # None = the whole source half
NSWEEP_TE    = [500, 1000, 2000, 5000]
NSWEEP_SEEDS = [0, 1, 2, 3, 4]
QSENS_SEEDS  = [0, 1, 2, 3, 4]
QSENS_EPS    = [0.0, 0.1, 0.2, 0.3]
MC_K         = [3, 5, 10, 26, 47, 100]
MC_A         = [0.6, 0.8, 0.95]
ECE_BINS, N_BOOT, ALPHA = 15, 10000, 0.05
DIAG_RHO     = 6.0                    # reference tweak-one severity of the applicability diagnostic
N_JOBS = None                         # None = all CPU cores

if PROFILE in ('fake', 'smoke'):
    SEEDS_MAIN, SEEDS_NOISE, SEEDS_QTRAIN, SEEDS_IW = [0, 1], [0], [0], [0]
    NOISE_LEVELS, NSWEEP_SEEDS, QSENS_SEEDS = [1.5], [0], [0]
    EPOCHS = {k: 2 for k in EPOCHS}
    N_REP, N_BOOT, MC_K = 6, 500, [3, 10]
    NSWEEP_SRC, NSWEEP_TE = [500, None], [500, 1000]
    COLLAPSE_RUNS = [('emnist_balanced', 0)] + ([('cifar10_raw', 0)] if RUN_CIFAR_RAW else [])
if PROFILE == 'fake':
    N_TE_MAIN = 400

CFG_SIG = (f"v1|vf{VAL_FRAC}|bs{BATCH}|ep{sorted(EPOCHS.items())}|lr{sorted(LR.items())}"
           f"|cov{COVTYPE_PER_CLASS}|q{sorted(Q_STRENGTH.items())}|{PROFILE}")
print('PROFILE =', PROFILE)
if PROFILE != 'full':
    print('NOTE: numbers produced under this profile are pipeline checks, not results.')

# %% [markdown]
# ## 2. Environment, folders and small utilities
#
# Folders follow the project layout: `dataset/` (downloads and extracted features),
# `model/` (weights), `output/` (cached predictor outputs and Part B results), `fig/`
# (figures with their CSV files). `crc()` gives seeds that are identical in every process,
# which Python's `hash()` does not.

# %%
import os, sys, json, time, glob, zlib, zipfile, math, warnings, platform
import numpy as np
warnings.filterwarnings('ignore')

ROOT = '/kaggle/working' if os.path.isdir('/kaggle/working') else os.path.abspath('kbs_run')
DIRS = {k: os.path.join(ROOT, k) for k in ('dataset', 'model', 'output', 'fig', 'tables')}
for d in DIRS.values(): os.makedirs(d, exist_ok=True)
CACHE_DIR = os.path.join(DIRS['output'], 'cache')
RES_DIR = os.path.join(DIRS['output'], 'results')
for d in (CACHE_DIR, RES_DIR): os.makedirs(d, exist_ok=True)

T_START = time.time()
def elapsed_h(): return (time.time() - T_START) / 3600.0
def crc(*parts): return zlib.crc32('|'.join(map(str, parts)).encode())
def rng_for(*parts): return np.random.default_rng(crc(*parts))

def save_json(obj, path):
    def conv(o):
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return str(o)
    with open(path, 'w') as f: json.dump(obj, f, indent=1, default=conv)

def write_csv(path, rows):
    import csv
    if not rows: return
    keys = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in keys: keys.append(k)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

def resume_from_inputs():
    """Copy extracted features and finished runs from earlier sessions attached as inputs
    (Kaggle: 'Add Input' -> the output of a previous version of this notebook). Files that
    already exist here are left alone; cached runs are still checked against CFG_SIG."""
    n = 0
    for f in glob.glob('/kaggle/input/**/*.npz', recursive=True):
        b = os.path.basename(f)
        if b in ('cifar10_feat.npz', 'svhn_feat.npz'): dst = os.path.join(DIRS['dataset'], b)
        elif os.sep + 'cache' + os.sep in f: dst = os.path.join(CACHE_DIR, b)
        else: continue
        if not os.path.exists(dst):
            import shutil; shutil.copy(f, dst); n += 1
    if n: print(f'resumed {n} files from /kaggle/input')
resume_from_inputs()
print('root:', ROOT)
print('python', platform.python_version(), '| numpy', np.__version__)

# %% [markdown]
# ## 3. Data
#
# All loaders return a **training pool** and a **test pool** as float32 tensors plus the
# number of classes. The test pool is used only for the robustness experiment E7 and the
# validation-to-test diagnostic; the main protocol draws source and target from the held-out
# part of the training pool, so that they share $p(x\mid y)$ by construction.
#
# * MNIST, Fashion-MNIST, KMNIST, EMNIST-Letters (K=26), EMNIST-Balanced (K=47):
#   flattened 28x28 pixels in [0,1].
# * CIFAR-10 and SVHN (`*_feat`): 512-dimensional features of an ImageNet-pretrained
#   ResNet-18 (torchvision `IMAGENET1K_V1`), frozen, extracted once and cached. The extractor
#   was trained with ordinary ImageNet labels; no label of the target task enters it.
# * CIFAR-10 raw (`cifar10_raw`): 3x32x32 images for the end-to-end ResNet-18 collapse run.
# * Covertype (K=7, tabular, imbalanced): 54 features; at most `COVTYPE_PER_CLASS` rows per
#   class (the rarest class has 2747), which keeps the source imbalanced (about 5.5 to 1) without
#   a class too rare to be predicted; split 80/20 into training and test pools, standardised with
#   training-pool statistics.
# * SVHN is taken from torchvision, else from `.mat` files attached under `/kaggle/input`, else
#   from the Hugging Face dataset `ufldl-stanford/svhn` (config `cropped_digits`).
# * Extracted features and finished runs of an earlier session, attached as a Kaggle input, are
#   copied in at start-up (cell 2), so slow downloads happen once.
#
# Under `PROFILE = 'fake'` every loader returns synthetic data of the same shape family, with
# the same K, so the whole notebook can be checked without a download.

# %%
import torch, torch.nn as nn, torch.nn.functional as F
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('torch', torch.__version__, '| device', DEVICE,
      torch.cuda.get_device_name(0) if DEVICE == 'cuda' else '')

DATASET_K = {'mnist': 10, 'fashion': 10, 'kmnist': 10, 'emnist_letters': 26,
             'emnist_balanced': 47, 'cifar10_feat': 10, 'svhn_feat': 10,
             'cifar10_raw': 10, 'covertype': 7}
DATASET_LABEL = {'mnist': 'MNIST', 'fashion': 'Fashion-MNIST', 'kmnist': 'KMNIST',
                 'emnist_letters': 'EMNIST-Letters', 'emnist_balanced': 'EMNIST-Balanced',
                 'cifar10_feat': 'CIFAR-10 (feat.)', 'svhn_feat': 'SVHN (feat.)',
                 'cifar10_raw': 'CIFAR-10 (raw)', 'covertype': 'Covertype'}
DATASET_SHORT = {'mnist': 'MNIST', 'fashion': 'F-MNIST', 'kmnist': 'KMNIST', 'emnist_letters': 'EMNIST-L',
                 'emnist_balanced': 'EMNIST-B', 'cifar10_feat': 'CIFAR-10', 'svhn_feat': 'SVHN',
                 'cifar10_raw': 'CIFAR-10 raw', 'covertype': 'Covertype'}
_DATA = {}

def _fake(name):
    K = DATASET_K[name]; r = rng_for('fake', name)
    if name == 'cifar10_raw':
        n_tr, n_te = 1200, 400
        y_tr, y_te = r.integers(0, K, n_tr), r.integers(0, K, n_te)
        mu = r.normal(0, .6, (K, 3, 1, 1))
        mk = lambda y: (mu[y] + r.normal(0, 1, (len(y), 3, 32, 32))).astype(np.float32)
        return torch.tensor(mk(y_tr)), torch.tensor(y_tr), torch.tensor(mk(y_te)), torch.tensor(y_te), K
    D = 54 if name == 'covertype' else (512 if name.endswith('_feat') else 784)
    n_tr, n_te = 6000, 2000
    prior = np.ones(K) / K
    if name in ('covertype', 'svhn_feat'):
        prior = r.dirichlet(np.ones(K) * 2.0)
    y_tr, y_te = r.choice(K, n_tr, p=prior), r.choice(K, n_te, p=prior)
    sep = {'emnist_balanced': 0.9}.get(name, 1.6)
    mu = r.normal(0, sep / math.sqrt(D) * 4, (K, D))
    mk = lambda y: (mu[y] + r.normal(0, 1, (len(y), D))).astype(np.float32)
    return torch.tensor(mk(y_tr)), torch.tensor(y_tr), torch.tensor(mk(y_te)), torch.tensor(y_te), K

def _stack_tv(ds):
    dl = torch.utils.data.DataLoader(ds, batch_size=4096, num_workers=2)
    xs, ys = zip(*dl)
    return torch.cat(xs), torch.cat(ys)

def _resnet_features(X_uint8_nchw):
    from torchvision.models import resnet18, ResNet18_Weights
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1); net.fc = nn.Identity()
    net = net.to(DEVICE).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)
    out = []
    with torch.no_grad(), torch.autocast(device_type='cuda', enabled=(DEVICE == 'cuda')):
        for i in range(0, len(X_uint8_nchw), 512):
            x = X_uint8_nchw[i:i + 512].to(DEVICE).float() / 255.0
            x = F.interpolate(x, size=224, mode='bilinear', align_corners=False)
            out.append(net((x - mean) / std).float().cpu())
    return torch.cat(out)

def _svhn_arrays(root):
    """SVHN cropped digits as uint8 NCHW tensors and labels 0..9, from the first source that
    works: (1) torchvision (ufldl.stanford.edu), (2) train_32x32.mat / test_32x32.mat attached
    under /kaggle/input, (3) the Hugging Face dataset ufldl-stanford/svhn, config cropped_digits."""
    from torchvision import datasets
    try:
        tr = datasets.SVHN(root, split='train', download=True); te = datasets.SVHN(root, split='test', download=True)
        print('  SVHN source: torchvision')
        return [(torch.tensor(d.data), torch.tensor(d.labels)) for d in (tr, te)]
    except Exception as e:
        print(f'  SVHN via torchvision failed ({type(e).__name__}); trying other sources')
    mats = {os.path.basename(p): p for p in glob.glob('/kaggle/input/**/*_32x32.mat', recursive=True)}
    if 'train_32x32.mat' in mats and 'test_32x32.mat' in mats:
        from scipy.io import loadmat
        out = []
        for k in ('train_32x32.mat', 'test_32x32.mat'):
            m = loadmat(mats[k]); y = m['y'].ravel().astype(np.int64); y[y == 10] = 0
            out.append((torch.tensor(np.transpose(m['X'], (3, 2, 0, 1)).copy()), torch.tensor(y)))
        print('  SVHN source: .mat files under /kaggle/input'); return out
    from huggingface_hub import list_repo_files, hf_hub_download
    import pandas as pd, io
    from PIL import Image
    repo = 'ufldl-stanford/svhn'
    files = list_repo_files(repo, repo_type='dataset')
    out = []
    for split in ('train', 'test'):
        fs = sorted(f for f in files if f.startswith(f'cropped_digits/{split}') and f.endswith('.parquet'))
        assert fs, f'no {split} parquet in {repo}: {files[:20]}'
        df = pd.concat([pd.read_parquet(hf_hub_download(repo, f, repo_type='dataset', cache_dir=root)) for f in fs])
        imgs = np.stack([np.array(Image.open(io.BytesIO(v['bytes'])).convert('RGB')) for v in df['image']])
        out.append((torch.tensor(imgs).permute(0, 3, 1, 2).contiguous(), torch.tensor(df['label'].to_numpy().astype(np.int64))))
    print(f'  SVHN source: Hugging Face {repo} ({len(out[0][1])} train, {len(out[1][1])} test)')
    return out

def load_dataset(name):
    if name in _DATA: return _DATA[name]
    if PROFILE == 'fake':
        _DATA[name] = _fake(name); return _DATA[name]
    from torchvision import datasets, transforms
    root, tf = DIRS['dataset'], transforms.ToTensor()
    if name in ('mnist', 'fashion', 'kmnist'):
        cls = {'mnist': datasets.MNIST, 'fashion': datasets.FashionMNIST, 'kmnist': datasets.KMNIST}[name]
        Xtr, Ytr = _stack_tv(cls(root, train=True, download=True, transform=tf))
        Xte, Yte = _stack_tv(cls(root, train=False, download=True, transform=tf))
        out = (Xtr.view(len(Xtr), -1), Ytr, Xte.view(len(Xte), -1), Yte, 10)
    elif name in ('emnist_letters', 'emnist_balanced'):
        split = name.split('_')[1]
        Xtr, Ytr = _stack_tv(datasets.EMNIST(root, split=split, train=True, download=True, transform=tf))
        Xte, Yte = _stack_tv(datasets.EMNIST(root, split=split, train=False, download=True, transform=tf))
        if split == 'letters': Ytr, Yte = Ytr - 1, Yte - 1          # labels 1..26 -> 0..25
        out = (Xtr.view(len(Xtr), -1), Ytr, Xte.view(len(Xte), -1), Yte, DATASET_K[name])
    elif name in ('cifar10_feat', 'svhn_feat'):
        path = os.path.join(root, f'{name}.npz')
        if not os.path.exists(path):
            t0 = time.time()
            if name == 'cifar10_feat':
                tr = datasets.CIFAR10(root, train=True, download=True)
                te = datasets.CIFAR10(root, train=False, download=True)
                Xtr = torch.tensor(tr.data).permute(0, 3, 1, 2); Ytr = torch.tensor(tr.targets)
                Xte = torch.tensor(te.data).permute(0, 3, 1, 2); Yte = torch.tensor(te.targets)
            else:
                (Xtr, Ytr), (Xte, Yte) = _svhn_arrays(root)
            np.savez(path, Xtr=_resnet_features(Xtr).numpy(), Ytr=Ytr.numpy(),
                     Xte=_resnet_features(Xte).numpy(), Yte=Yte.numpy())
            print(f'  features for {name} extracted in {time.time() - t0:.0f}s')
        z = np.load(path)
        out = (torch.tensor(z['Xtr']), torch.tensor(z['Ytr']).long(),
               torch.tensor(z['Xte']), torch.tensor(z['Yte']).long(), 10)
    elif name == 'cifar10_raw':
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
        Xtr, Ytr = _stack_tv(datasets.CIFAR10(root, train=True, download=True, transform=tf))
        Xte, Yte = _stack_tv(datasets.CIFAR10(root, train=False, download=True, transform=tf))
        out = ((Xtr - mean) / std, Ytr, (Xte - mean) / std, Yte, 10)
    elif name == 'covertype':
        from sklearn.datasets import fetch_covtype
        cv = fetch_covtype(data_home=root)
        X, y = cv.data.astype(np.float32), cv.target.astype(np.int64) - 1
        r = rng_for('covtype-subset')
        idx = np.concatenate([r.permutation(np.flatnonzero(y == k))[:COVTYPE_PER_CLASS] for k in range(7)])
        idx = r.permutation(idx); X, y = X[idx], y[idx]
        n_tr = int(0.8 * len(y))
        mu, sd = X[:n_tr].mean(0), X[:n_tr].std(0) + 1e-6
        X = (X - mu) / sd
        out = (torch.tensor(X[:n_tr]), torch.tensor(y[:n_tr]),
               torch.tensor(X[n_tr:]), torch.tensor(y[n_tr:]), 7)
    else:
        raise ValueError(name)
    _DATA[name] = out
    return out

def corrupt(X, level, seed):
    """E3: additive Gaussian input noise; the same distribution for every split."""
    if level <= 0: return X
    g = torch.Generator().manual_seed(seed)
    return X + torch.randn(X.shape, generator=g) * float(level)

print('loaders ready')

# %% [markdown]
# ## 4. Transition matrices, complementary labels, models and the training objective
#
# $Q[y,\bar y]=p(\bar y\mid y)$ has a zero diagonal and unit row sums (eq. Q). The uniform
# case is $Q=(\mathbf 1\mathbf 1^\top-I)/(K-1)$; the biased cases draw
# $M_{y\bar y}=\exp(s\,z_{y\bar y})$ with $z\sim N(0,1)$, zero the diagonal and normalise the
# rows, with $s=0.5$ (mild) or $1.2$ (strong).
#
# The predictor minimises the non-negative corrected complementary risk of Ishida et al.
# (2019): per complementary class $k$, the partial risk
# $\hat R_k=\frac1{n_k}\sum_{i:\bar y_i=k}\sum_j [Q^{-1}]_{kj}\,\ell(f(x_i),j)$, and the
# objective $\sum_k\max(\hat R_k,0)-\sum_k\min(\hat R_k,0)$. Ablation (f) multiplies each
# sample's corrected loss vector by the oracle weight of the class, which is the
# importance-weighted variant.

# %%
def Q_uniform(K):
    Q = np.full((K, K), 1.0 / (K - 1)); np.fill_diagonal(Q, 0.0); return Q

def Q_biased(K, strength, seed=1):
    r = np.random.default_rng(seed)
    M = np.exp(strength * r.standard_normal((K, K))); np.fill_diagonal(M, 0.0)
    return M / M.sum(1, keepdims=True)

def make_Q(K, kind):
    return Q_uniform(K) if kind == 'uniform' else Q_biased(K, Q_STRENGTH[kind], seed=1 if kind == 'mild' else 2)

def draw_complementary(y, Q, seed):
    y = np.asarray(y)
    u = np.random.default_rng(seed).random(len(y))
    return (Q[y].cumsum(1) < u[:, None]).sum(1)

def nncll_risk(logits, ybar, Qinv_t, K, wclass=None):
    L = -F.log_softmax(logits, 1)                       # ordinary losses, one per class
    if wclass is not None: L = L * wclass[None, :]      # ablation (f): importance weights
    v = (Qinv_t[ybar] * L).sum(1)                       # backward-corrected loss per sample
    s = torch.zeros(K, dtype=v.dtype, device=v.device).index_add_(0, ybar, v)
    c = torch.zeros(K, dtype=v.dtype, device=v.device).index_add_(0, ybar, torch.ones_like(v))
    pr = s[c > 0] / c[c > 0]
    return torch.clamp(pr, min=0).sum() - torch.clamp(pr, max=0).sum(), float(pr.sum())

def make_model(name, d_in, K):
    if name == 'cifar10_raw':
        from torchvision.models import resnet18
        m = resnet18(num_classes=K)
        m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False); m.maxpool = nn.Identity()
        return m
    return nn.Sequential(nn.Linear(d_in, 500), nn.ReLU(), nn.Linear(500, 500), nn.ReLU(),
                         nn.Linear(500, K))

def logits_of(model, X):
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), 4096):
            out.append(model(X[i:i + 4096].to(DEVICE)).float().cpu())
    return torch.cat(out).numpy().astype(np.float32)

print('model code ready')

# %% [markdown]
# ## 5. The run list
#
# Each run is one trained predictor. Groups: `main` (all datasets, uniform Q, 10 seeds),
# `noise` (E3), `qtrain` (E4a), `iw` (ablation f) and `collapse` (E9). The tag fully
# identifies a run; the cached file also carries `CFG_SIG`.

# %%
def run_specs():
    S = []
    for s in SEEDS_MAIN:
        for d in MAIN_DATASETS:
            S.append(dict(group='main', dataset=d, qkind='uniform', noise=0.0, seed=s, iw=False))
    for s in SEEDS_NOISE:
        for d in NOISE_DATASETS:
            for lv in NOISE_LEVELS:
                S.append(dict(group='noise', dataset=d, qkind='uniform', noise=lv, seed=s, iw=False))
    for s in SEEDS_QTRAIN:
        for d in QTRAIN_DATASETS:
            for qk in QTRAIN_KINDS:
                S.append(dict(group='qtrain', dataset=d, qkind=qk, noise=0.0, seed=s, iw=False))
    for s in SEEDS_IW:
        S.append(dict(group='iw', dataset=IW_DATASET, qkind='uniform', noise=0.0, seed=s, iw=True))
    for d, s in COLLAPSE_RUNS:
        S.append(dict(group='collapse', dataset=d, qkind='uniform', noise=0.0, seed=s, iw=False))
    for sp in S:
        sp['tag'] = (f"{sp['group']}_{sp['dataset']}_{sp['qkind']}_n{sp['noise']}_s{sp['seed']}"
                     + ('_iw' if sp['iw'] else ''))
    return S

SPECS = run_specs()
print(len(SPECS), 'runs:', {g: sum(s['group'] == g for s in SPECS)
                            for g in ('main', 'noise', 'qtrain', 'iw', 'collapse')})

# %% [markdown]
# ## 6. Training one predictor
#
# For a run: split the training pool into a training part and a held-out complementary
# source split (`VAL_FRAC`), draw complementary labels from Q for both, train with the
# non-negative corrected risk, and cache
# the held-out logits / true labels / complementary labels, the test-pool logits and labels,
# and the outputs on a fixed subset of the training part (ablation d). The true labels of the
# held-out split are cached for evaluation only; no estimator reads them except the clean-label
# reference BBSE, which is reported as an upper reference.
#
# The health gate is `rank(C_bar) == K` on the full held-out split. Unhealthy runs stay in the
# cache and are counted, never silently dropped. For the `iw` group the oracle weight vector of
# the tweak-one target (class `IW_CLASS` boosted by `IW_RHO`) multiplies the loss.

# %%
def cache_path(tag): return os.path.join(CACHE_DIR, f'{tag}.npz')

def train_one(sp, verbose=True):
    path = cache_path(sp['tag'])
    if os.path.exists(path):
        try:
            if str(np.load(path, allow_pickle=True)['cfg_sig']) == CFG_SIG:
                if verbose: print(f"  [cached] {sp['tag']}")
                return path
            if verbose: print(f"  [stale ] {sp['tag']} -> retraining")
        except Exception as e:
            print(f"  [broken] {sp['tag']}: {e} -> retraining")
    t0 = time.time()
    name, seed = sp['dataset'], sp['seed']
    torch.manual_seed(crc('torch', sp['tag']))
    Xtr, Ytr, Xte, Yte, K = load_dataset(name)
    if sp['noise'] > 0:
        Xtr = corrupt(Xtr, sp['noise'], crc('noise-tr', sp['tag']))
        Xte = corrupt(Xte, sp['noise'], crc('noise-te', sp['tag']))
    n = len(Ytr)
    perm = rng_for('split', name, seed).permutation(n)
    nv = int(VAL_FRAC * n); vi, ti = perm[:nv], perm[nv:]
    Q = make_Q(K, sp['qkind'])
    Qinv_t = torch.tensor(np.linalg.inv(Q), dtype=torch.float32, device=DEVICE)
    yb_t = draw_complementary(Ytr[ti].numpy(), Q, crc('ybar-tr', sp['tag']))
    yb_v = draw_complementary(Ytr[vi].numpy(), Q, crc('ybar-val', sp['tag']))
    wclass = None
    if sp['iw']:
        p_tr = np.bincount(Ytr[ti].numpy(), minlength=K) / len(ti)
        p_te = np.ones(K); p_te[IW_CLASS] = IW_RHO; p_te /= p_te.sum()
        wclass = torch.tensor(p_te / p_tr, dtype=torch.float32, device=DEVICE)

    model = make_model(name, Xtr[0].numel(), K).to(DEVICE)
    lr = LR.get(name, LR['default']); epochs = EPOCHS.get(name, EPOCHS['default'])
    opt = torch.optim.Adam(model.parameters(), lr)
    use_amp = (DEVICE == 'cuda' and name == 'cifar10_raw')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    Xh = Xtr[ti].to(DEVICE); yh = torch.tensor(yb_t, device=DEVICE)
    risk_curve = []; t_train = time.time()
    for ep in range(epochs):
        model.train()
        p = torch.randperm(len(Xh), device=DEVICE); tot = []
        for i in range(0, len(Xh), BATCH):
            b = p[i:i + BATCH]
            if len(b) < 2: continue
            xb = Xh[b]
            if name == 'cifar10_raw':                        # standard light augmentation
                if torch.rand(1).item() < 0.5: xb = xb.flip(3)
            with torch.autocast(device_type='cuda', enabled=use_amp):
                loss, raw = nncll_risk(model(xb).float(), yh[b], Qinv_t, K, wclass)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot.append(raw)
        risk_curve.append(float(np.mean(tot)))
    if DEVICE == 'cuda': torch.cuda.synchronize()
    train_seconds = time.time() - t_train
    lv, lt = logits_of(model, Xtr[vi]), logits_of(model, Xte)
    sub = ti[:min(TRAIN_SUBSET_CACHE, len(ti))]
    ls = logits_of(model, Xtr[sub])
    pv = lv.argmax(1)
    Cb = np.zeros((K, K)); np.add.at(Cb, (pv, yb_v), 1.0); Cb /= len(pv)
    rank = int(np.linalg.matrix_rank(Cb)); cov = int(len(np.unique(pv)))
    val_acc = float((pv == Ytr[vi].numpy()).mean())
    test_acc = float((lt.argmax(1) == Yte.numpy()).mean())
    secs = time.time() - t0
    np.savez_compressed(path, cfg_sig=CFG_SIG, spec=json.dumps(sp), K=K, Q=Q,
                        val_logits=lv, val_true=Ytr[vi].numpy(), val_ybar=yb_v,
                        test_logits=lt, test_true=Yte.numpy(),
                        sub_logits=ls, sub_true=Ytr[sub].numpy(), sub_ybar=yb_t[:len(sub)],
                        risk_curve=np.array(risk_curve), val_acc=val_acc, test_acc=test_acc,
                        rank=rank, coverage=cov, healthy=(rank == K), seconds=secs,
                        train_seconds=train_seconds,
                        epochs=epochs, n_train=len(ti), n_val=len(vi))
    torch.save(model.state_dict(), os.path.join(DIRS['model'], f"{sp['tag']}.pt"))
    if verbose:
        print(f"  [{'ok   ' if rank == K else 'DEGEN'}] {sp['tag']}: val_acc={val_acc:.4f} "
              f"test_acc={test_acc:.4f} cov={cov}/{K} rank={rank} ({secs:.0f}s, {epochs} ep)")
    return path

print('train_one ready')

# %% [markdown]
# ## 7. Part A: train every predictor
#
# Runs are ordered seed-major within each group, so if the time budget ends early every
# dataset still has its first seeds. Re-running this cell resumes. Under `smoke`, the cell also
# prints a time projection for `full` from the measured seconds per epoch of each dataset.

# %%
manifest = []
for sp in sorted(SPECS, key=lambda s: (['main', 'noise', 'qtrain', 'iw', 'collapse'].index(s['group']),
                                         s['seed'], s['dataset'], s['noise'])):
    if elapsed_h() > TIME_BUDGET_H:
        print(f'time budget reached ({elapsed_h():.2f} h); re-run this cell to resume'); break
    p = train_one(sp)
    z = np.load(p, allow_pickle=True)
    manifest.append(dict(sp, path=p, healthy=bool(z['healthy']), val_acc=float(z['val_acc']),
                         test_acc=float(z['test_acc']), rank=int(z['rank']),
                         coverage=int(z['coverage']), seconds=float(z['seconds']),
                         train_seconds=float(z['train_seconds']),
                         epochs=int(z['epochs']), K=int(z['K'])))
write_csv(os.path.join(RES_DIR, 'manifest.csv'), manifest)
print(f"\nPart A: {len(manifest)}/{len(SPECS)} runs, {sum(m['healthy'] for m in manifest)} healthy, "
      f"{elapsed_h():.2f} h this session")

if PROFILE == 'smoke':
    sec_per_epoch, overhead = {}, {}
    for m in manifest:            # training-loop time only; downloads and feature extraction excluded
        sec_per_epoch.setdefault(m['dataset'], []).append(m['train_seconds'] / max(m['epochs'], 1))
        overhead.setdefault(m['dataset'], []).append(m['seconds'] - m['train_seconds'])
    full_epochs = {'default': 60, 'emnist_letters': 100, 'emnist_balanced': 200, 'cifar10_raw': 60}
    plan = {'main': (MAIN_DATASETS, 10), 'noise': (NOISE_DATASETS, 3 * 5),
            'qtrain': (QTRAIN_DATASETS, 5 * 2), 'iw': ([IW_DATASET], 3),
            'collapse_b': (['emnist_balanced'], 3), 'collapse_c': (['cifar10_raw'], 1 if RUN_CIFAR_RAW else 0)}
    tot = 0.0
    for g, (dss, reps) in plan.items():
        for d in dss:
            spe = np.median(sec_per_epoch.get(d, [np.nan]))
            ovh = np.min(overhead.get(d, [0.0]))          # per-run overhead without the first download
            tot += reps * (spe * full_epochs.get(d, 60) + ovh)
    print(f"PROJECTION for PROFILE='full', Part A: {tot / 3600:.2f} h "
          f"(+ dataset downloads and feature extraction, measured once above)")

# %% [markdown]
# ## 8. Package Part A
#
# Writes `partA_cache.zip` with every cached run and the manifest. Kaggle keeps
# `/kaggle/working` when the version is saved; download the zip as well, so that Part B can be
# re-run anywhere.

# %%
zp = os.path.join(ROOT, 'partA_cache.zip')
with zipfile.ZipFile(zp, 'w', zipfile.ZIP_STORED) as zf:
    for f in sorted(glob.glob(os.path.join(CACHE_DIR, '*.npz'))) + [os.path.join(RES_DIR, 'manifest.csv')]:
        zf.write(f, os.path.relpath(f, ROOT))
print(f'{zp}: {os.path.getsize(zp) / 1e6:.1f} MB')

# %% [markdown]
# # Part B (CPU only)
#
# ## 9. Estimators and metrics
#
# Notation follows the manuscript. On a source half: $\hat{\bar C}$ (eq. Cbar), the predictor
# accuracy estimated from complementary labels $\hat a=1-(K-1)\operatorname{tr}\hat{\bar C}$
# (eq. aident; valid for any Q and any source prior), $\hat b=(1-\hat a)/(K-1)$, and the
# temperature $\hat T$ that minimises the backward-corrected log loss on complementary labels.
# On a target sample: $\hat q$ (eq. qte).
#
# * `naive`: $\Pi(\hat q/p_{tr})$ (eq. naive).
# * `clsc`: $\Pi\big((\hat q-\hat b\mathbf 1)/((\hat a-\hat b)\,p_{tr})\big)$. With a balanced
#   source this equals $\Pi((\hat q/p_{tr}-\hat\delta\mathbf 1)/(1-\hat\delta))$ (eq. clsc_hat);
#   the form here also holds for an imbalanced source, because
#   $q_{te}=(a-b)p_{te}+b\mathbf 1$ for any $p_{tr}$ under the uniform off-diagonal confusion.
# * `plain`: $\Pi(Q\hat{\bar C}^{-1}\hat q)$ (eq. clbbse).
# * `mlls`: EM of Saerens et al. (2002) on the target posteriors at temperature $\hat T$.
# * `bbse_clean`: $\Pi(\hat C^{-1}\hat q)$ with the clean confusion. Uses clean source labels;
#   an upper reference, not a competitor.
#
# Failure never becomes a default: a singular solve, a projection with no mass or
# $\hat a\le\hat b$ returns NaN and is counted. Metrics after prior correction
# $p_{te}(k\mid x)\propto p(k\mid x)\,\hat w_k$: accuracy, NLL, ECE (15 bins), Brier score.

# %%
from scipy.stats import wilcoxon, friedmanchisquare, spearmanr, studentized_range

def softmax(lg, T=1.0):
    z = lg / T; z = z - z.max(1, keepdims=True); e = np.exp(z)
    return e / e.sum(1, keepdims=True)

def confusion(rows, cols, K):
    M = np.zeros((K, K)); np.add.at(M, (rows, cols), 1.0); return M / max(len(rows), 1)

def project(w, p_tr):
    if not np.all(np.isfinite(w)): return np.full(len(w), np.nan)
    w = np.clip(w, 0.0, None); s = float(p_tr @ w)
    return w / s if s > 1e-12 else np.full(len(w), np.nan)

def a_hat_qfree(Cb): return 1.0 - (len(Cb) - 1) * float(np.trace(Cb))
def a_hat_general(Cb, Qe):
    try: return float(np.trace(Cb @ np.linalg.inv(Qe)))
    except np.linalg.LinAlgError: return np.nan

def est_naive(q, p_tr): return project(q / p_tr, p_tr)

def est_clsc(q, p_tr, a, proj=True):
    K = len(q); b = (1.0 - a) / (K - 1)
    if not np.isfinite(a) or a - b <= 1e-9: return np.full(K, np.nan)
    w = (q - b) / ((a - b) * p_tr)
    return project(w, p_tr) if proj else w

def est_plain(q, p_tr, Cb, Qe, proj=True):
    try: w = Qe @ np.linalg.solve(Cb, q)
    except np.linalg.LinAlgError: return np.full(len(q), np.nan)
    return project(w, p_tr) if proj else w

def est_bbse_clean(q, p_tr, C):
    """Lipton et al. (2018): q_te = C w with C[i,y] = p_tr(h=i, y), so w = C^{-1} q_te."""
    try: w = np.linalg.solve(C, q)
    except np.linalg.LinAlgError: return np.full(len(q), np.nan)
    return project(w, p_tr)

def est_mlls(P, p_tr, n_iter=300, tol=1e-6):
    pi = p_tr.copy(); ratio = 1.0 / p_tr
    for _ in range(n_iter):
        r = P * (pi * ratio)[None, :]; r /= r.sum(1, keepdims=True)
        new = r.mean(0)
        if np.abs(new - pi).max() < tol: pi = new; break
        pi = new
    return project(pi / p_tr, p_tr)

def fit_T_comp(lg, ybar, Qinv, grid=np.exp(np.linspace(np.log(0.2), np.log(8.0), 60))):
    best, bT = np.inf, 1.0
    for t in grid:
        z = lg / t; z = z - z.max(1, keepdims=True)
        L = -(z - np.log(np.exp(z).sum(1, keepdims=True)))
        v = float((Qinv[ybar] * L).sum(1).mean())
        if v < best: best, bT = v, float(t)
    return bT

def fit_T_clean(lg, y, grid=np.exp(np.linspace(np.log(0.2), np.log(8.0), 60))):
    best, bT = np.inf, 1.0
    for t in grid:
        P = softmax(lg, t); v = float(-np.log(np.maximum(P[np.arange(len(y)), y], 1e-12)).mean())
        if v < best: best, bT = v, float(t)
    return bT

def ece_of(P, y, bins=ECE_BINS):
    pred = P.argmax(1); conf = P.max(1); corr = (pred == y).astype(float)
    b = np.clip((conf * bins).astype(int), 0, bins - 1)
    cnt = np.bincount(b, minlength=bins)
    s_conf = np.bincount(b, weights=conf, minlength=bins); s_corr = np.bincount(b, weights=corr, minlength=bins)
    m = cnt > 0
    return float(np.sum(np.abs(s_corr[m] - s_conf[m])) / len(y))

def downstream(P, y, w):
    if not np.all(np.isfinite(w)): return (np.nan,) * 4
    Pw = P * w[None, :]; Pw /= np.maximum(Pw.sum(1, keepdims=True), 1e-300)
    acc = float((Pw.argmax(1) == y).mean())
    nll = float(-np.log(np.maximum(Pw[np.arange(len(y)), y], 1e-12)).mean())
    onehot = np.zeros_like(Pw); onehot[np.arange(len(y)), y] = 1.0
    brier = float(((Pw - onehot) ** 2).sum(1).mean())
    return acc, nll, ece_of(Pw, y), brier

def perturb_Q(Q, eps, seed):
    K = len(Q); r = np.random.default_rng(seed)
    E = r.standard_normal((K, K)); np.fill_diagonal(E, 0.0)
    E *= eps * np.linalg.norm(Q) / max(np.linalg.norm(E), 1e-12)
    Qh = np.clip(Q + E, 1e-6, None); np.fill_diagonal(Qh, 0.0)
    return Qh / Qh.sum(1, keepdims=True)

def sigma_min_lemma(K, a): return abs(K * a - 1) / (K * (K - 1) ** 2)

def _deviation(pv, yb, p_tr, Qinv, K):
    """E = P_hat - (a-b) I - b 1 1^T with P_hat(h=i|y=k) = [C_bar Q^{-1}]_{ik} / p_tr(k) (Lemma 1)."""
    Cb = confusion(pv, yb, K); a = a_hat_qfree(Cb); b = (1 - a) / (K - 1)
    return (Cb @ Qinv) / p_tr[None, :] - (a - b) * np.eye(K) - b

def diag_misfit(pv, yb, p_tr, Qe, K, rng, rho=None):
    """Label-free estimate of the misfit of the shrinkage identity (applicability diagnostic).
    The source split is halved at random into A and B; with E_A, E_B the deviations of the two
    halves, <d_A, d_B> is an unbiased estimate of ||d||^2 because the halves are independent,
    whereas ||d||^2 from one sample is inflated by the noise Q^{-1} amplifies. d = (E p_te) / p_tr for
    tweak-one targets of severity rho, averaged over the boosted class and normalised by ||w||^2.
    The quantity estimated is the deviation of the naive limit from the shrinkage identity,
    (E p_te) / p_tr (elementwise division), relative to w = p_te / p_tr.
    Uses predictions, complementary labels, Q and p_tr only."""
    rho = rho or DIAG_RHO; Qinv = np.linalg.inv(Qe)
    s = rng.permutation(len(pv)); A, B = s[: len(s) // 2], s[len(s) // 2:]
    EA, EB = _deviation(pv[A], yb[A], p_tr, Qinv, K), _deviation(pv[B], yb[B], p_tr, Qinv, K)
    m = []
    for c in range(K):
        p_te = np.ones(K); p_te[c] = rho; p_te /= p_te.sum(); w = p_te / p_tr
        m.append(float(((EA @ p_te) / p_tr) @ ((EB @ p_te) / p_tr)) / float(w @ w))
    return float(np.sqrt(max(np.mean(m), 0.0)))
print('estimators ready')

# %% [markdown]
# ## 10. The cross-fitted evaluator
#
# `evaluate_run` implements the main protocol. The held-out split is halved; half A is the
# source (it gives $\hat{\bar C}$, $\hat a$, $\hat T$, $p_{tr}$), half B is the pool targets are
# drawn from, then the halves swap. A target sample of size `n_te` is drawn by class with
# replacement from the pool under the target prior. Shifts:
#
# * tweak-one: class $c$ up-weighted by $\rho$, with $c$ = resample index mod K, so every class
#   is boosted equally often;
# * Dirichlet: $p_{te}\sim\mathrm{Dir}(\alpha\mathbf 1)$, a fresh draw per resample.
#
# For each resample the evaluator records every estimator's $\|\hat w-w\|_2$, the ablation
# variants, and (if requested) downstream metrics at $T=1$ and $T=\hat T$. For tweak-one it
# also records population diagnostics of the pool: the naive limit, its bias against the
# prediction of the general shrinkage identity, the misfit of that identity, and the
# Theorem 3 prediction of the plain CL-BBSE error at the sizes used.
#
# Options: `n_src` subsamples the source half (E2); `Q_draw` re-draws complementary labels of
# the held-out split from another Q with the predictor fixed (E4b); `Q_est` is the matrix the
# estimators are told (E4c); `protocol='testpool'` uses the whole held-out split as source and
# the official test pool as target pool (E7).

# %%
METHODS_MAIN = ['naive', 'clsc', 'plain', 'mlls', 'bbse_clean']
ABLATIONS = ['clsc_oracle_a', 'clsc_noproj', 'clsc_general', 'mlls_T1', 'mlls_Toracle',
             'clsc_insample', 'plain_insample', 'plain_unproj', 'select']
DS_METHODS = ['none', 'naive', 'clsc', 'plain', 'mlls', 'oracle']

def shift_list(tweak=True, dirichlet=True, rhos=None):
    L = []
    if tweak: L += [('tweak', r) for r in (rhos or TWEAK_RHOS)]
    if dirichlet: L += [('dirichlet', a) for a in DIR_ALPHAS]
    return L

def prior_for(kind, par, K, rep, r):
    if kind == 'tweak':
        p = np.ones(K); p[rep % K] = par; return p / p.sum()
    p = r.dirichlet(np.ones(K) * par); p = np.maximum(p, 1e-6); return p / p.sum()

def rule_prefers_clsc(q, p_tr, a, n_te):
    K = len(q); b = (1 - a) / (K - 1); delta = 1 - (a - b)
    wc = est_clsc(q, p_tr, a, proj=False)
    if not np.all(np.isfinite(wc)): return False
    s2 = float(np.sum(q * (1 - q) / (n_te * p_tr ** 2)))
    return delta ** 2 * float(np.sum((wc - 1) ** 2)) > s2 * ((1 - delta) ** -2 - 1)

def evaluate_run(path, shifts, n_te=None, n_src=None, Q_draw=None, Q_est=None,
                 protocol='crossfit', downstream_on=True, ablations_on=True, mlls_on=True,
                 tag_extra=''):
    d = np.load(path, allow_pickle=True)
    sp = json.loads(str(d['spec'])); K = int(d['K']); Q0 = d['Q']
    vl, vt, vy = d['val_logits'], d['val_true'], d['val_ybar']
    if Q_draw is not None:
        vy = draw_complementary(vt, Q_draw, crc('redraw', sp['tag'], tag_extra))
    Qtrue = Q0 if Q_draw is None else Q_draw
    Qe = Qtrue if Q_est is None else Q_est
    Qinv_e = np.linalg.inv(Qe)
    if protocol == 'crossfit':
        perm = rng_for('half', sp['tag']).permutation(len(vt)); h = len(vt) // 2
        folds = [(perm[:h], perm[h:2 * h]), (perm[h:2 * h], perm[:h])]
        pool_lg, pool_y = vl, vt
    else:
        folds = [(np.arange(len(vt)), np.arange(len(d['test_true'])))]
        pool_lg, pool_y = d['test_logits'], d['test_true']
    sub_lg, sub_ybar = d['sub_logits'], d['sub_ybar']
    rows = []
    for fi, (src, tgt) in enumerate(folds):
        if n_src is not None and n_src < len(src):
            src = src[rng_for('nsrc', sp['tag'], n_src, fi).permutation(len(src))[:n_src]]
        pv = vl[src].argmax(1)
        p_tr = np.bincount(vt[src], minlength=K) / len(src)
        if np.any(p_tr == 0): continue
        Cb = confusion(pv, vy[src], K)
        C = confusion(pv, vt[src], K)
        a_hat = a_hat_qfree(Cb); a_gen = a_hat_general(Cb, Qe)
        a_clean = float(np.mean(pv == vt[src]))
        T_hat = fit_T_comp(vl[src], vy[src], Qinv_e) if (mlls_on or downstream_on) else 1.0
        T_or = fit_T_clean(vl[src], vt[src]) if (mlls_on and ablations_on) else 1.0
        Cb_in = confusion(sub_lg.argmax(1), sub_ybar, K)
        a_in = a_hat_qfree(Cb_in)
        tl, tt = pool_lg[tgt], pool_y[tgt]; tp = tl.argmax(1)
        P1, PT = softmax(tl, 1.0), softmax(tl, T_hat)
        PO = softmax(tl, T_or) if (mlls_on and ablations_on) else None
        by = [np.flatnonzero(tt == k) for k in range(K)]
        if any(len(b) == 0 for b in by): continue
        Ph = confusion(tp, tt, K) / np.maximum(np.bincount(tt, minlength=K) / len(tt), 1e-12)[None, :]
        rank_ok = np.linalg.matrix_rank(Cb) == K
        if rank_ok:
            Ci = np.linalg.inv(Cb); M = Qe @ Ci
        nte = n_te or N_TE_MAIN
        nte = int(min(nte, len(tgt) // 2))
        smin = float(np.linalg.svd(Cb, compute_uv=False)[-1])
        D_hat = diag_misfit(pv, vy[src], p_tr, Qe, K, rng_for('diag', sp['tag'], tag_extra, protocol, n_src, fi))
        for kind, par in shifts:
            acc = {}
            for rep in range(N_REP):
                r = rng_for('tgt', sp['tag'], tag_extra, protocol, n_src, nte, fi, kind, par, rep)
                p_te = prior_for(kind, par, K, rep, r); w = p_te / p_tr
                cnt = r.multinomial(nte, p_te)
                ix = np.concatenate([r.choice(by[k], size=c, replace=True) for k, c in enumerate(cnt) if c > 0])
                y = tt[ix]; q = np.bincount(tp[ix], minlength=K) / len(ix)
                W = {'naive': est_naive(q, p_tr), 'clsc': est_clsc(q, p_tr, a_hat),
                     'plain': est_plain(q, p_tr, Cb, Qe),
                     'bbse_clean': est_bbse_clean(q, p_tr, C)}
                if mlls_on: W['mlls'] = est_mlls(PT[ix], p_tr)
                if ablations_on:
                    W['clsc_oracle_a'] = est_clsc(q, p_tr, a_clean)
                    W['clsc_noproj'] = est_clsc(q, p_tr, a_hat, proj=False)
                    W['clsc_general'] = est_clsc(q, p_tr, a_gen)
                    W['clsc_insample'] = est_clsc(q, p_tr, a_in)
                    W['plain_insample'] = est_plain(q, p_tr, Cb_in, Qe)
                    W['plain_unproj'] = est_plain(q, p_tr, Cb, Qe, proj=False)
                    W['select'] = W['clsc'] if rule_prefers_clsc(q, p_tr, a_hat, len(ix)) else W['naive']
                    if mlls_on:
                        W['mlls_T1'] = est_mlls(P1[ix], p_tr)
                        W['mlls_Toracle'] = est_mlls(PO[ix], p_tr)
                for m, v in W.items():
                    acc.setdefault('err_' + m, []).append(float(np.linalg.norm(v - w)) if np.all(np.isfinite(v)) else np.nan)
                if downstream_on:
                    Wd = dict(none=np.ones(K), naive=W['naive'], clsc=W['clsc'], plain=W['plain'],
                              mlls=W.get('mlls', np.full(K, np.nan)), oracle=w)
                    for tname, P in (('T1', P1), ('That', PT)):
                        for m, v in Wd.items():
                            a_, n_, e_, b_ = downstream(P[ix], y, v)
                            for mk, mv in (('acc', a_), ('nll', n_), ('ece', e_), ('brier', b_)):
                                acc.setdefault(f'{mk}_{tname}_{m}', []).append(mv)
            row = dict(tag=sp['tag'], group=sp['group'], dataset=sp['dataset'], seed=sp['seed'],
                       qkind=sp['qkind'], noise=sp['noise'], K=K, fold=fi, protocol=protocol,
                       shift=kind, par=par, n_src=len(src), n_te=nte, a_hat=a_hat, a_gen=a_gen,
                       a_clean=a_clean, T_hat=T_hat, T_oracle=T_or, smin=smin,
                       smin_lemma=sigma_min_lemma(K, a_hat), D_hat=D_hat, src_imbalance=float(p_tr.max() / p_tr.min()),
                       tag_extra=tag_extra)
            for k, v in acc.items():
                v = np.array(v, float)
                row[k] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
                if k.startswith('err_'): row['fail_' + k[4:]] = int(np.sum(~np.isfinite(v)))
            if kind == 'tweak':                       # population diagnostics, all K boosted classes
                b_hat = (1 - a_hat) / (K - 1)
                bias_m, bias_t, mis, pred_rmse, V = [], [], [], [], []
                for c in range(K):
                    p_te = np.ones(K); p_te[c] = par; p_te /= p_te.sum(); w = p_te / p_tr
                    qpop = Ph @ p_te
                    lim = qpop / p_tr
                    lim_thm = ((a_hat - b_hat) * p_te + b_hat) / p_tr
                    bias_m.append(float(np.linalg.norm(lim - w))); bias_t.append(float(np.linalg.norm(lim_thm - w)))
                    mis.append(float(np.linalg.norm(lim - lim_thm) / np.linalg.norm(w)))
                    if rank_ok:
                        u = Ci @ qpop
                        S_te = np.diag(qpop) - np.outer(qpop, qpop)
                        S_tr = np.diag(Cb @ (u * u)) - np.outer(qpop, qpop)
                        pred_rmse.append(float(np.sqrt(max(np.trace(M @ (S_te / nte + S_tr / len(src)) @ M.T), 0))))
                        V.append(float(np.trace(M @ (S_te + S_tr) @ M.T)))
                row.update(bias_naive=float(np.mean(bias_m)), bias_thm=float(np.mean(bias_t)),
                           misfit_w=float(np.mean(mis)),
                           pred_rmse_plain=float(np.mean(pred_rmse)) if pred_rmse else np.nan,
                           n_star=float(np.mean(V) / np.mean(np.square(bias_m))) if V else np.nan)
            rows.append(row)
    return rows

def parallel_map(fn, items):
    from joblib import Parallel, delayed
    return Parallel(n_jobs=N_JOBS or os.cpu_count(), verbose=0)(delayed(fn)(*it) for it in items)

print('evaluator ready')

# %% [markdown]
# ## 11. Load the manifest and apply the health gate
#
# Part B reads only files listed in the manifest whose `CFG_SIG` matches; it never lists the
# cache directory. The table printed here is Table 1 of the manuscript in raw form.

# %%
import csv
MAN = [m for m in csv.DictReader(open(os.path.join(RES_DIR, 'manifest.csv')))]
for m in MAN:
    m['healthy'] = m['healthy'] == 'True'; m['seed'] = int(m['seed']); m['noise'] = float(m['noise'])
    m['K'] = int(m['K']); m['val_acc'] = float(m['val_acc']); m['test_acc'] = float(m['test_acc'])
    assert str(np.load(m['path'], allow_pickle=True)['cfg_sig']) == CFG_SIG, m['tag']
def runs(group, healthy=True, **kw):
    return [m for m in MAN if m['group'] == group and (m['healthy'] or not healthy)
            and all(m[k] == v for k, v in kw.items())]
t1 = []
for dname in dict.fromkeys(MAIN_DATASETS + [d for d, _ in COLLAPSE_RUNS]):   # each dataset once
    ms = [m for m in MAN if m['dataset'] == dname and m['group'] in ('main', 'collapse')]
    if not ms: continue
    z = np.load(ms[0]['path'], allow_pickle=True)
    p = np.bincount(z['val_true'], minlength=int(z['K'])) / len(z['val_true'])
    t1.append(dict(dataset=dname, label=DATASET_LABEL[dname], K=ms[0]['K'], n_train=int(z['n_train']),
                   n_heldout=int(z['n_val']), n_test=len(z['test_true']),
                   prior_ratio=float(p.max() / p.min()), runs=len(ms),
                   healthy=sum(m['healthy'] for m in ms),
                   val_acc=float(np.mean([m['val_acc'] for m in ms])),
                   test_acc=float(np.mean([m['test_acc'] for m in ms]))))
write_csv(os.path.join(RES_DIR, 'table1_datasets.csv'), t1)
for r in t1:
    print(f"{r['label']:<18} K={r['K']:<3} heldout={r['n_heldout']:<7} healthy {r['healthy']}/{r['runs']}  "
          f"val_acc {r['val_acc']:.3f}  test_acc {r['test_acc']:.3f}  prior max/min {r['prior_ratio']:.1f}")

# %% [markdown]
# ## 12. E1 main comparison and E5 downstream (cross-fitted protocol)
#
# All healthy `main` runs, tweak-one and Dirichlet shifts, all estimators, ablations and
# downstream metrics. Output: `E1_cells.csv`, one row per (run, fold, shift).

# %%
t0 = time.time()
E1 = sum(parallel_map(lambda p: evaluate_run(p, shift_list()), [(m['path'],) for m in runs('main')]), [])
write_csv(os.path.join(RES_DIR, 'E1_cells.csv'), E1)
print(f'E1: {len(E1)} cells in {time.time() - t0:.0f}s')

# %% [markdown]
# ## 13. E2 sample-size sweep
#
# K=10 and K=26 main runs of `NSWEEP_SEEDS`; tweak-one at $\rho=6$. First the source half is
# subsampled (target size fixed at `N_TE_MAIN`), then the target size is varied (whole source
# half). Theorem 3's prediction of the plain CL-BBSE error is evaluated on the same quantities.

# %%
t0 = time.time()
sel = [m for m in runs('main') if m['seed'] in NSWEEP_SEEDS and m['K'] in (10, 26)]
items = [(m['path'], ns, None) for m in sel for ns in NSWEEP_SRC] + \
        [(m['path'], None, nt) for m in sel for nt in NSWEEP_TE]
def _e2(p, ns, nt):
    rr = evaluate_run(p, [('tweak', 6.0)], n_te=nt, n_src=ns, downstream_on=False,
                      ablations_on=True, mlls_on=False, tag_extra=f'e2_{ns}_{nt}')
    for r in rr: r['sweep'] = 'n_src' if nt is None else 'n_te'
    return rr
E2 = sum(parallel_map(_e2, items), [])
write_csv(os.path.join(RES_DIR, 'E2_cells.csv'), E2)
print(f'E2: {len(E2)} cells in {time.time() - t0:.0f}s')

# %% [markdown]
# ## 14. E3 accuracy sweep by input noise
#
# `noise` runs plus the level-0 main runs of the same datasets and seeds; tweak-one only.

# %%
t0 = time.time()
sel = runs('noise') + [m for m in runs('main') if m['dataset'] in NOISE_DATASETS and m['seed'] in SEEDS_NOISE]
E3 = sum(parallel_map(lambda p: evaluate_run(p, shift_list(dirichlet=False), downstream_on=False,
                                             ablations_on=False, tag_extra='e3'),
                      [(m['path'],) for m in sel]), [])
write_csv(os.path.join(RES_DIR, 'E3_cells.csv'), E3)
print(f'E3: {len(E3)} cells in {time.time() - t0:.0f}s')

# %% [markdown]
# ## 15. E4 sensitivity to the transition matrix
#
# (a) predictors trained under a biased Q, evaluated with that Q; (b) complementary labels of
# the held-out split re-drawn from a biased Q with the predictor fixed; (c) the estimators are
# given a misspecified $\hat Q$ (relative Frobenius perturbation $\varepsilon$). The Q-free
# CL-SC never reads $\hat Q$; the general form $\hat a=\operatorname{tr}(\hat{\bar C}\hat Q^{-1})$,
# plain CL-BBSE and the complementary-label temperature do.

# %%
t0 = time.time()
def _e4a(p):
    return evaluate_run(p, shift_list(dirichlet=False), downstream_on=False, tag_extra='e4a')
E4a = sum(parallel_map(_e4a, [(m['path'],) for m in runs('qtrain')]), [])
E4a += sum(parallel_map(_e4a, [(m['path'],) for m in runs('main') if m['dataset'] in QTRAIN_DATASETS
                                and m['seed'] in SEEDS_QTRAIN]), [])
def _e4bc(p, K, qk, eps):
    Qd = make_Q(K, qk)
    Qe = Qd if eps == 0 else perturb_Q(Qd, eps, crc('eps', qk, eps))
    rr = evaluate_run(p, shift_list(dirichlet=False), Q_draw=Qd, Q_est=Qe, downstream_on=False,
                      tag_extra=f'e4_{qk}_{eps}')
    for r in rr: r.update(q_draw=qk, eps=eps)
    return rr
sel = [m for m in runs('main') if m['dataset'] in NOISE_DATASETS and m['seed'] in QSENS_SEEDS]
items = [(m['path'], m['K'], qk, eps) for m in sel for qk in ('uniform', 'mild', 'strong') for eps in QSENS_EPS]
E4bc = sum(parallel_map(_e4bc, items), [])
write_csv(os.path.join(RES_DIR, 'E4a_cells.csv'), E4a)
write_csv(os.path.join(RES_DIR, 'E4bc_cells.csv'), E4bc)
print(f'E4: {len(E4a)} + {len(E4bc)} cells in {time.time() - t0:.0f}s')

# %% [markdown]
# ## 16. E7 robustness: the official test pool as target
#
# Source = the whole held-out split, target pool = the official test pool. Where the test pool
# differs from the training pool in $p(x\mid y)$ (measured by the validation-to-test accuracy
# gap), this is label shift plus a class-conditional change, and no estimator in this family
# is consistent; the experiment measures how each degrades.

# %%
t0 = time.time()
E7 = sum(parallel_map(lambda p: evaluate_run(p, shift_list(dirichlet=False), protocol='testpool',
                                             downstream_on=False, ablations_on=False, tag_extra='e7'),
                      [(m['path'],) for m in runs('main')]), [])
write_csv(os.path.join(RES_DIR, 'E7_cells.csv'), E7)
print(f'E7: {len(E7)} cells in {time.time() - t0:.0f}s')

# %% [markdown]
# ## 17. Calibration decomposition, ablation (f) and E9 collapse
#
# Calibration: per predicted class, the gap between mean confidence and accuracy on the clean
# held-out split; its weighted mean (global part) and weighted spread across classes
# (class-dependent part), uncalibrated, with the single complementary-label temperature, and
# with per-class temperatures fitted the same way.
#
# Ablation (f): importance-weighted nnCLL trained with oracle weights for the tweak-one target
# (class `IW_CLASS`, $\rho=$ `IW_RHO`), against the unweighted predictor of the same seed with
# prior correction at decision time. Target samples come from the held-out split, which neither
# predictor trained on.
#
# E9: health of the collapse runs.

# %%
def calib_decomp(P, y, K):
    pred = P.argmax(1); conf = P.max(1); corr = (pred == y).astype(float)
    gaps, wts = [], []
    for k in range(K):
        m = pred == k
        if m.sum() < 20: continue
        gaps.append(conf[m].mean() - corr[m].mean()); wts.append(m.mean())
    gaps, wts = np.array(gaps), np.array(wts) / np.sum(wts)
    mu = float((gaps * wts).sum())
    return abs(mu), float(np.sqrt(((gaps - mu) ** 2 * wts).sum())), ece_of(P, y)

def _calib(p):
    d = np.load(p, allow_pickle=True); sp = json.loads(str(d['spec'])); K = int(d['K'])
    vl, vt, vy = d['val_logits'], d['val_true'], d['val_ybar']; Qinv = np.linalg.inv(d['Q'])
    h = len(vt) // 2; perm = rng_for('half', sp['tag']).permutation(len(vt)); A, B = perm[:h], perm[h:]
    T = fit_T_comp(vl[A], vy[A], Qinv)
    pred = vl[A].argmax(1); Tk = np.ones(K)
    for k in range(K):
        m = pred == k
        if m.sum() >= 50: Tk[k] = fit_T_comp(vl[A][m], vy[A][m], Qinv)
    lB = vl[B]
    out = dict(tag=sp['tag'], dataset=sp['dataset'], seed=sp['seed'], K=K, T_hat=T,
               T_oracle=fit_T_clean(vl[A], vt[A]), Tk_min=float(Tk.min()), Tk_max=float(Tk.max()))
    for nm, P in (('uncal', softmax(lB)), ('single', softmax(lB, T)),
                  ('perclass', softmax(lB / Tk[lB.argmax(1)][:, None]))):
        g, s, e = calib_decomp(P, vt[B], K)
        out.update({f'{nm}_global': g, f'{nm}_spread': s, f'{nm}_ece': e})
    return out
CAL = parallel_map(_calib, [(m['path'],) for m in runs('main')])
write_csv(os.path.join(RES_DIR, 'calibration.csv'), CAL)

IW = []
for m in runs('iw', healthy=False):
    base = [b for b in runs('main') if b['dataset'] == IW_DATASET and b['seed'] == m['seed']]
    if not base: continue
    for label, p in (('iw_nncll', m['path']), ('unweighted', base[0]['path'])):
        d = np.load(p, allow_pickle=True); K = int(d['K'])
        vl, vt, vy = d['val_logits'], d['val_true'], d['val_ybar']
        p_tr = np.bincount(vt, minlength=K) / len(vt)
        p_te = np.ones(K); p_te[IW_CLASS] = IW_RHO; p_te /= p_te.sum(); w = p_te / p_tr
        T = fit_T_comp(vl, vy, np.linalg.inv(d['Q']))
        by = [np.flatnonzero(vt == k) for k in range(K)]
        res = {}
        for rep in range(N_REP):
            r = rng_for('iw', m['seed'], rep); cnt = r.multinomial(min(N_TE_MAIN, len(vt) // 2), p_te)
            ix = np.concatenate([r.choice(by[k], size=c, replace=True) for k, c in enumerate(cnt) if c > 0])
            for nm, P, ww in (('no_correction', softmax(vl[ix]), np.ones(K)),
                              ('oracle_correction_That', softmax(vl[ix], T), w)):
                res.setdefault(nm, []).append(downstream(P, vt[ix], ww)[0])
        IW.append(dict(seed=m['seed'], model=label, min_train_risk=float(np.min(d['risk_curve'])),
                       final_train_risk=float(d['risk_curve'][-1]),
                       **{f'acc_{k}': float(np.mean(v)) for k, v in res.items()}))
write_csv(os.path.join(RES_DIR, 'ablation_iw.csv'), IW)
COL = [dict(dataset=m['dataset'], seed=m['seed'], K=m['K'], val_acc=m['val_acc'], rank=m['rank'],
            coverage=m['coverage'], healthy=m['healthy']) for m in runs('collapse', healthy=False)]
write_csv(os.path.join(RES_DIR, 'E9_collapse.csv'), COL)
print(f'calibration {len(CAL)} runs, IW {len(IW)} rows, collapse {len(COL)} runs')

# %% [markdown]
# ## 18. E6 theory checks by simulation and E8 the Proposition 1 counterexample
#
# Simulation (canonical setting: uniform Q, balanced source, uniform off-diagonal confusion of
# accuracy a): Monte Carlo error of unprojected plain CL-BBSE against $\sqrt{V/n}$ of Theorem 3,
# and the crossover $n^*=V/\text{bias}^2$, over `MC_K` x `MC_A`. Theorem 3 is a first-order
# (large-n) statement, so n is set to $400/\sigma_{\min}(\bar C)^2$ (at least 20 000): the sampling
# noise of $\hat{\bar C}$ is then small against its smallest singular value. The joint counts of
# (h, y_bar) and of the target predictions are drawn as multinomials, which makes large n cheap.
# A second row per (K, a) uses n = 20 000 to show where the first-order regime has not been reached.
#
# Proposition 1: two Gaussian classes at $\pm\mu$ with variance $\sigma^2$, threshold rules, 0-1
# loss. The risk-minimising threshold under prior $p$ is $t^*(p)=\sigma^2\log(p_0/p_1)/(2\mu)$.
# A balanced source and the target $w=(0.2,1.8)$ give the excess target risk of the
# source-optimal rule; the same is computed over a range of $\|w-\mathbf 1\|$.

# %%
from scipy.stats import norm
def canonical(K, a, rho=3.0):
    Q = Q_uniform(K); b = (1 - a) / (K - 1)
    conf = np.full((K, K), b); np.fill_diagonal(conf, a)
    p_tr = np.ones(K) / K; p_te = np.ones(K); p_te[0] = rho; p_te /= p_te.sum()
    return Q, conf, p_tr, p_te, (conf * p_tr[None, :]) @ Q
def V_theorem3(Q, conf, p_tr, p_te, Cb):
    q = conf @ p_te; Ci = np.linalg.inv(Cb); M = Q @ Ci; u = Ci @ q
    S = np.diag(q) + np.diag(Cb @ (u * u)) - 2 * np.outer(q, q)
    return float(np.trace(M @ S @ M.T))
MC = []
for K in MC_K:
    for a in MC_A:
        Q, conf, p_tr, p_te, Cb = canonical(K, a); w = p_te / p_tr
        V = V_theorem3(Q, conf, p_tr, p_te, Cb)
        bias = (K * (1 - a) / (K - 1)) * np.linalg.norm(w - 1)
        q = conf @ p_te
        n_first = int(min(max(400.0 / sigma_min_lemma(K, a) ** 2, 20000), 1e13))
        for regime, n in (('first_order', n_first), ('n_20000', 20000)):
            errs = []
            for s in range(N_REP):
                r = rng_for('mc', K, a, s, regime)
                Ch = r.multinomial(n, Cb.ravel() / Cb.sum()).reshape(K, K) / n
                qh = r.multinomial(n, q / q.sum()) / n
                try: errs.append(float(np.linalg.norm(Q @ np.linalg.solve(Ch, qh) - w)))
                except np.linalg.LinAlgError: errs.append(np.nan)
            MC.append(dict(K=K, a=a, regime=regime, n=n, V=V, pred_rmse=math.sqrt(V / n),
                           mc_mean_err=float(np.nanmean(errs)), mc_median_err=float(np.nanmedian(errs)),
                           mc_rmse=float(np.sqrt(np.nanmean(np.square(errs)))), bias_naive=bias,
                           n_star=V / bias ** 2, fails=int(np.sum(~np.isfinite(errs)))))
write_csv(os.path.join(RES_DIR, 'E6_montecarlo.csv'), MC)

mu, sig = 1.0, 1.0
def risk(t, p0):                     # class 0 at -mu, class 1 at +mu; predict 1 if x > t
    return p0 * (1 - norm.cdf((t + mu) / sig)) + (1 - p0) * norm.cdf((t - mu) / sig)
P1R = []
for w0 in [1.0, 0.8, 0.6, 0.4, 0.2, 0.1]:
    p_te0 = 0.5 * w0; t_src = 0.0
    t_opt = sig ** 2 * math.log(p_te0 / (1 - p_te0)) / (2 * mu)
    P1R.append(dict(w0=w0, w1=2 - w0, norm_w_minus_1=math.sqrt(2) * abs(1 - w0), t_source=t_src,
                    t_target=t_opt, risk_source_rule=risk(t_src, p_te0), risk_target_rule=risk(t_opt, p_te0),
                    excess=risk(t_src, p_te0) - risk(t_opt, p_te0)))
tgrid = np.linspace(-3, 3, 601)
P1C = [dict(t=float(t), risk_target=risk(t, 0.1), risk_source=risk(t, 0.5)) for t in tgrid]
write_csv(os.path.join(RES_DIR, 'E8_prop1_excess.csv'), P1R)
write_csv(os.path.join(RES_DIR, 'E8_prop1_curves.csv'), P1C)
print('E6 Monte Carlo rmse/pred (first-order n):', [(r['K'], r['a'], round(r['mc_rmse'] / r['pred_rmse'], 3)) for r in MC if r['regime'] == 'first_order'])
print('E8: excess at w=(0.2,1.8) =', round([r for r in P1R if r['w0'] == 0.2][0]['excess'], 4))

# %% [markdown]
# ## 19. Statistics
#
# The unit is the run: cells are averaged within a run before any test, because resamples
# and folds of one predictor are not independent. For each dataset and pooled over datasets of
# the same K group:
#
# * paired Wilcoxon signed-rank test of CL-SC against each competitor, Holm correction over the
#   competitors within each dataset;
# * bootstrap 95% interval (`N_BOOT` resamples of runs) of the paired difference;
# * Cliff's delta and win / tie / loss counts (tie: |difference| < 1e-3);
# * Friedman test over all runs (blocks) and the Nemenyi critical difference for the average
#   ranks, $CD=q_\alpha\sqrt{k(k+1)/(6N)}$ with $q_\alpha$ from the studentised range.
#
# Separately for tweak-one, Dirichlet and both.

# %%
def per_run(cells, key, filt=None):
    g = {}
    for c in cells:
        if filt and not filt(c): continue
        g.setdefault(c['tag'], []).append(c.get(key, np.nan))
    return {t: float(np.nanmean(v)) for t, v in g.items()}

def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); adj = np.empty_like(ps); run_max = 0.0
    for i, j in enumerate(o):
        run_max = max(run_max, (len(ps) - i) * ps[j]); adj[j] = min(run_max, 1.0)
    return adj

def boot_ci(x, B=None):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 2: return (np.nan, np.nan)
    r = rng_for('boot', len(x)); m = x[r.integers(0, len(x), (B or N_BOOT, len(x)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

def cliffs_delta(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return float(np.mean(np.sign(x[:, None] - y[None, :])))

COMP = ['naive', 'plain', 'mlls']
def compare(cells, label, filt=None):
    out = []
    ref = per_run(cells, 'err_clsc', filt)
    ps = []
    for m in COMP:
        oth = per_run(cells, f'err_{m}', filt)
        tags = [t for t in ref if t in oth and np.isfinite(ref[t]) and np.isfinite(oth[t])]
        diff = np.array([oth[t] - ref[t] for t in tags])
        p = float(wilcoxon(diff).pvalue) if len(diff) >= 6 and np.any(diff != 0) else np.nan
        ps.append(p)
        out.append(dict(subset=label, competitor=m, runs=len(tags),
                        mean_clsc=float(np.mean([ref[t] for t in tags])) if tags else np.nan,
                        mean_competitor=float(np.mean([oth[t] for t in tags])) if tags else np.nan,
                        mean_diff=float(diff.mean()) if len(diff) else np.nan,
                        ci_low=boot_ci(diff)[0], ci_high=boot_ci(diff)[1], p_wilcoxon=p,
                        cliffs_delta=cliffs_delta([oth[t] for t in tags], [ref[t] for t in tags]) if tags else np.nan,
                        wins=int((diff > 1e-3).sum()), ties=int((np.abs(diff) <= 1e-3).sum()),
                        losses=int((diff < -1e-3).sum())))
    adj = holm([p if np.isfinite(p) else 1.0 for p in ps])
    for o, a in zip(out, adj): o['p_holm'] = float(a)
    return out

STATS = []
families = {'tweak': lambda c: c['shift'] == 'tweak', 'dirichlet': lambda c: c['shift'] == 'dirichlet', 'all': None}
for fam, f in families.items():
    for dname in MAIN_DATASETS:
        sub = [c for c in E1 if c['dataset'] == dname]
        if sub: STATS += [dict(r, family=fam, dataset=dname) for r in compare(sub, dname, f)]
    for kg, cond in (('K10', lambda c: c['K'] == 10), ('allK', lambda c: True)):
        sub = [c for c in E1 if cond(c)]
        STATS += [dict(r, family=fam, dataset=kg) for r in compare(sub, kg, f)]
write_csv(os.path.join(RES_DIR, 'table3_tests.csv'), STATS)

RANK_METHODS = ['naive', 'clsc', 'plain', 'mlls']
mat = []
for t in sorted({c['tag'] for c in E1}):
    vals = [per_run([c for c in E1 if c['tag'] == t], f'err_{m}')[t] for m in RANK_METHODS]
    if np.all(np.isfinite(vals)): mat.append(vals)
mat = np.array(mat)
from scipy.stats import rankdata
ranks = np.array([rankdata(r) for r in mat]); avg_rank = ranks.mean(0)
fr = friedmanchisquare(*mat.T) if len(mat) > 2 else None
kM = len(RANK_METHODS); N = len(mat)
q_alpha = float(studentized_range.ppf(1 - ALPHA, kM, np.inf) / math.sqrt(2))
CD = q_alpha * math.sqrt(kM * (kM + 1) / (6 * N)) if N else np.nan
CDROWS = [dict(method=m, avg_rank=float(r)) for m, r in zip(RANK_METHODS, avg_rank)]
save_json(dict(friedman_stat=float(fr.statistic) if fr else np.nan, friedman_p=float(fr.pvalue) if fr else np.nan,
               N_runs=N, k=kM, q_alpha=q_alpha, CD=CD, avg_ranks=CDROWS), os.path.join(RES_DIR, 'friedman_nemenyi.json'))
write_csv(os.path.join(RES_DIR, 'cd_ranks.csv'), CDROWS)
print('Friedman p =', None if fr is None else f'{fr.pvalue:.2e}', '| CD =', round(CD, 3), '|',
      {m: round(float(r), 2) for m, r in zip(RANK_METHODS, avg_rank)})
for r in [s for s in STATS if s['family'] == 'all' and s['dataset'] in ('K10', 'allK')]:
    print(f"{r['dataset']:>5} CL-SC vs {r['competitor']:<6} diff {r['mean_diff']:+.3f} "
          f"[{r['ci_low']:+.3f},{r['ci_high']:+.3f}] p_holm={r['p_holm']:.2g} W/T/L {r['wins']}/{r['ties']}/{r['losses']}")

# %% [markdown]
# ## 19b. E10: a label-free applicability rule
#
# The shrinkage identity behind CL-SC holds exactly only when the confusion is uniform off the
# diagonal; its misfit on the data decides whether CL-SC helps. `D_hat` (section 9) estimates that
# misfit from complementary labels alone. The rule is: use CL-SC when `D_hat < tau`, otherwise the
# naive estimator. The decision is taken per source fold, from that fold's `D_hat`.
#
# * tau is chosen on the `main` runs by leave-one-dataset-out: for each dataset, tau minimises the
#   mean relative error, rule / min(naive, CL-SC), over the runs of the other datasets, and the rule
#   is then scored on the held-out dataset;
# * tau* fitted on all `main` runs is applied unchanged to runs never used to choose it: the noise
#   sweep (E3) and the predictors trained under a biased Q (E4a, group `qtrain`).

# %%
def rule_cells(cells, tau):
    for c in cells:
        c['err_rule'] = c['err_clsc'] if c['D_hat'] < tau else c['err_naive']
        c['rule_chose_clsc'] = float(c['D_hat'] < tau)
    return cells

def fit_tau(cells):
    cand = np.unique(np.r_[0.0, [c['D_hat'] for c in cells], 1e9])
    def score(t):
        rule_cells(cells, t)
        r = per_run(cells, 'err_rule'); n = per_run(cells, 'err_naive'); k = per_run(cells, 'err_clsc')
        return float(np.mean([r[x] / min(n[x], k[x]) for x in r]))
    sc = np.array([score(t) for t in cand])
    return float(np.median(cand[sc <= sc.min() + 1e-12]))

main_cells = [c for c in E1 if np.isfinite(c.get('D_hat', np.nan))]
LODO = []
for dname in sorted({c['dataset'] for c in main_cells}):
    tr = [dict(c) for c in main_cells if c['dataset'] != dname]
    te = [dict(c) for c in main_cells if c['dataset'] == dname]
    tau = fit_tau(tr); rule_cells(te, tau)
    for t, v in per_run(te, 'err_rule').items():
        LODO.append(dict(tag=t, dataset=dname, tau=tau, err_rule=v,
                         err_naive=per_run(te, 'err_naive')[t], err_clsc=per_run(te, 'err_clsc')[t],
                         err_mlls=per_run(te, 'err_mlls')[t], D_hat=per_run(te, 'D_hat')[t],
                         misfit_w=per_run([c for c in te if c['shift'] == 'tweak'], 'misfit_w')[t],
                         chose_clsc=per_run(te, 'rule_chose_clsc')[t]))
write_csv(os.path.join(RES_DIR, 'E10_lodo_runs.csv'), LODO)

def paired(a, b, label):
    d = np.asarray(b) - np.asarray(a); nz = d[np.abs(d) > 0]
    return dict(comparison=label, runs=len(d), mean_diff=float(d.mean()), ci_low=boot_ci(d)[0], ci_high=boot_ci(d)[1],
                p_wilcoxon=float(wilcoxon(nz).pvalue) if len(nz) >= 6 else np.nan,
                wins=int((d > 1e-3).sum()), ties=int((np.abs(d) <= 1e-3).sum()), losses=int((d < -1e-3).sum()))
E10T = []
for dname in sorted({r['dataset'] for r in LODO}) + ['all']:
    rr = [r for r in LODO if dname == 'all' or r['dataset'] == dname]
    E10T.append(dict(dataset=dname, tau=float(np.median([r['tau'] for r in rr])),
                     frac_clsc=float(np.mean([r['chose_clsc'] for r in rr])),
                     rule=float(np.mean([r['err_rule'] for r in rr])), naive=float(np.mean([r['err_naive'] for r in rr])),
                     clsc=float(np.mean([r['err_clsc'] for r in rr])), mlls=float(np.mean([r['err_mlls'] for r in rr])),
                     best_of_two=float(np.mean([min(r['err_naive'], r['err_clsc']) for r in rr]))))
E10S = [paired([r['err_rule'] for r in LODO], [r['err_naive'] for r in LODO], 'rule vs naive (LODO)'),
        paired([r['err_rule'] for r in LODO], [r['err_clsc'] for r in LODO], 'rule vs CL-SC (LODO)')]
TAU_STAR = fit_tau([dict(c) for c in main_cells])
OOS = []
for label, cells in (('E3 noise sweep', [c for c in E3 if c['group'] == 'noise']),
                     ('E4a biased-Q training', [c for c in E4a if c['group'] == 'qtrain'])):
    cells = rule_cells([dict(c) for c in cells if np.isfinite(c.get('D_hat', np.nan))], TAU_STAR)
    if not cells: continue
    r_, n_, k_ = per_run(cells, 'err_rule'), per_run(cells, 'err_naive'), per_run(cells, 'err_clsc')
    tags = sorted(r_)
    OOS.append(dict(subset=label, runs=len(tags), tau=TAU_STAR, frac_clsc=float(np.mean(list(per_run(cells, 'rule_chose_clsc').values()))),
                    rule=float(np.mean([r_[t] for t in tags])), naive=float(np.mean([n_[t] for t in tags])),
                    clsc=float(np.mean([k_[t] for t in tags])),
                    best_of_two=float(np.mean([min(n_[t], k_[t]) for t in tags]))))
    E10S += [dict(paired([r_[t] for t in tags], [n_[t] for t in tags], f'rule vs naive ({label})')),
             dict(paired([r_[t] for t in tags], [k_[t] for t in tags], f'rule vs CL-SC ({label})'))]
write_csv(os.path.join(RES_DIR, 'E10_lodo_table.csv'), E10T)
write_csv(os.path.join(RES_DIR, 'E10_tests.csv'), E10S)
write_csv(os.path.join(RES_DIR, 'E10_out_of_sample.csv'), OOS)
diag_corr = spearmanr([c['D_hat'] for c in main_cells if c['shift'] == 'tweak'],
                      [c['misfit_w'] for c in main_cells if c['shift'] == 'tweak']).correlation
diag_corr_run = spearmanr([r['D_hat'] for r in LODO], [r['misfit_w'] for r in LODO]).correlation
gain_corr_run = spearmanr([r['D_hat'] for r in LODO], [r['err_naive'] - r['err_clsc'] for r in LODO]).correlation
print(f'D_hat vs clean misfit, Spearman: {diag_corr:.3f} (tweak cells), {diag_corr_run:.3f} (runs); '
      f'D_hat vs CL-SC gain (runs): {gain_corr_run:.3f};  tau* = {TAU_STAR:.3f}')
for r in E10T: print(f"  {r['dataset']:<14} tau {r['tau']:.3f}  CL-SC chosen {100 * r['frac_clsc']:.0f}%  rule {r['rule']:.3f}  naive {r['naive']:.3f}  CL-SC {r['clsc']:.3f}  best {r['best_of_two']:.3f}")
for r in E10S: print(f"  {r['comparison']:<38} diff {r['mean_diff']:+.3f} W/T/L {r['wins']}/{r['ties']}/{r['losses']} p={r['p_wilcoxon']:.2g}")
for r in OOS: print('  out of sample:', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})

# %% [markdown]
# ## 20. Figures
#
# Style: serif text, no figure titles (the caption carries the title), panel labels (a), (b),
# (c) centred under each panel below the x-axis label, every figure saved as PDF and PNG at
# 300 dpi together with a CSV holding exactly the plotted values. Files are numbered in the
# order the figures appear in the manuscript (Figure_1 is the draw.io architecture diagram, made
# outside this notebook); Figure_A1 is in the appendix and Figure_S* in the supplementary material.
# Datasets without a healthy run (EMNIST-Letters) are left out of the per-dataset plots.

# %%
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
_fams = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams.update({'font.family': 'serif',
                     'font.serif': [f for f in ('Liberation Serif', 'Times New Roman', 'DejaVu Serif') if f in _fams] or ['serif'],
                     'mathtext.fontset': 'stix', 'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 7,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'axes.grid': True, 'grid.alpha': 0.3, 'grid.linewidth': 0.4})
MCOL = {'naive': '#4d4d4d', 'clsc': '#1f5fa8', 'plain': '#b2312b', 'mlls': '#2c8c4a', 'bbse_clean': '#8c6d1f'}
MLAB = {'naive': r'naive $\hat q/p_{tr}$', 'clsc': 'CL-SC', 'plain': 'plain CL-BBSE', 'mlls': 'MLLS (EM)',
        'bbse_clean': 'BBSE, clean labels'}
MMARK = {'naive': 'o', 'clsc': 's', 'plain': '^', 'mlls': 'D', 'bbse_clean': 'v'}

def panel_labels(fig, axes):
    fig.canvas.draw(); rend = fig.canvas.get_renderer()
    for i, ax in enumerate(axes):
        bb = ax.get_tightbbox(rend).transformed(fig.transFigure.inverted())
        x = (ax.get_position().x0 + ax.get_position().x1) / 2
        fig.text(x, bb.y0 - 0.012, f'({chr(97 + i)})', ha='center', va='top', fontsize=8)

def save_fig(fig, axes, name, rows):
    from matplotlib.ticker import NullFormatter
    from matplotlib.ticker import FixedLocator, FuncFormatter
    nice = [m * 10.0 ** e for e in range(-4, 6) for m in (1, 2, 3, 5)]
    finer = sorted(round(m * 10.0 ** e, 10) for e in range(-4, 6) for m in (1, 2, 3, 4, 5, 6, 8))
    fig.canvas.draw()                         # settle autoscaled limits before choosing ticks
    for a_ in axes:                           # no labels on minor ticks of logarithmic axes
        for axis, sc, lim in ((a_.yaxis, a_.get_yscale(), a_.get_ylim()), (a_.xaxis, a_.get_xscale(), a_.get_xlim())):
            if sc != 'log': continue
            axis.set_minor_formatter(NullFormatter())
            lo, hi = min(lim), max(lim)
            decades = sum(lo <= 10.0 ** e <= hi for e in range(-6, 8))
            if decades < 2 and isinstance(axis.get_major_locator(), matplotlib.ticker.LogLocator):
                tk = [t for t in nice if lo <= t <= hi]      # at most one labelled decade: label plain values
                if len(tk) < 3: tk = [t for t in finer if lo <= t <= hi]
                if len(tk) >= 2:
                    axis.set_major_locator(FixedLocator(tk)); axis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    fig.tight_layout()
    if len(axes) > 1:                         # single-panel figures carry no panel label
        fig.subplots_adjust(bottom=max(fig.subplotpars.bottom, 0.24)); panel_labels(fig, axes)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(DIRS['fig'], f'{name}.{ext}'), dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig); write_csv(os.path.join(DIRS['fig'], f'{name}.csv'), rows)
    print('  figure', name, f'({len(rows)} rows)')

def run_means(cells, key, by):
    g = {}
    for c in cells:
        g.setdefault((by(c), c['tag']), []).append(c.get(key, np.nan))
    out = {}
    for (k, t), v in g.items(): out.setdefault(k, []).append(np.nanmean(v))
    return out

def mean_ci(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) == 0: return np.nan, np.nan, np.nan
    lo, hi = boot_ci(v, 2000) if len(v) > 1 else (v[0], v[0])
    return float(v.mean()), lo, hi

W = 7.16
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
EVAL = [d for d in MAIN_DATASETS if any(c['dataset'] == d for c in E1)]   # datasets with healthy runs
def plain_ticks(axis, ticks, fmt='{:g}'):
    axis.set_major_locator(FixedLocator(ticks)); axis.set_minor_locator(NullLocator())
    axis.set_major_formatter(FuncFormatter(lambda v, _: fmt.format(v)))
# ---- Fig. 2: theory against measurement ----
rows = []
for c in [c for c in E1 if c['shift'] == 'tweak']:
    rows.append(dict(dataset=c['dataset'], seed=c['seed'], K=c['K'], fold=c['fold'], rho=c['par'],
                     smin=c['smin'], smin_lemma=c['smin_lemma'], src_imbalance=c['src_imbalance'],
                     bias_naive=c['bias_naive'], bias_thm=c['bias_thm'],
                     err_plain_unproj=c.get('err_plain_unproj', np.nan), pred_rmse_plain=c['pred_rmse_plain']))
for r in rows:   # Lemma 2 assumes a balanced source and a full-rank estimate
    r['in_panel_a'] = bool(r['src_imbalance'] < 2.0 and r['smin'] > 1e-12)
fig, ax = plt.subplots(1, 3, figsize=(W, 2.3))
for dname in EVAL:
    rr = [r for r in rows if r['dataset'] == dname]
    if not rr: continue
    ra = [r for r in rr if r['in_panel_a']]
    ax[0].loglog([r['smin_lemma'] for r in ra], [r['smin'] for r in ra], '.', ms=3, label=DATASET_SHORT[dname])
    ax[1].plot([r['bias_thm'] for r in rr], [r['bias_naive'] for r in rr], '.', ms=3)
    ax[2].loglog([r['pred_rmse_plain'] for r in rr], [r['err_plain_unproj'] for r in rr], '.', ms=3)
for a_, lab in zip(ax, (('closed form (Lemma 2)', r'measured $\sigma_{\min}(\hat{\bar C})$'),
                        ('prediction (Theorem 2)', 'measured naive bias'),
                        ('prediction (Theorem 3)', 'measured plain CL-BBSE error'))):
    lim = [min(a_.get_xlim()[0], a_.get_ylim()[0]), max(a_.get_xlim()[1], a_.get_ylim()[1])]
    a_.plot(lim, lim, 'k--', lw=0.7); a_.set_xlabel(lab[0]); a_.set_ylabel(lab[1])
ax[0].legend(fontsize=5.5, markerscale=2, loc='upper left')
for t_ in (ax[0].xaxis, ax[0].yaxis): plain_ticks(t_, [0.002, 0.004, 0.006, 0.01], '{:.3f}')
save_fig(fig, ax, 'Figure_2_theory', rows)

# ---- Fig. 3: main comparison per dataset ----
rows = []
fig, ax = plt.subplots(1, 2, figsize=(W, 2.5))
for j, (fam, f) in enumerate((('tweak', lambda c: c['shift'] == 'tweak'), ('dirichlet', lambda c: c['shift'] == 'dirichlet'))):
    for i, m in enumerate(['naive', 'clsc', 'plain', 'mlls', 'bbse_clean']):
        xs, ys, lo, hi = [], [], [], []
        for di, dname in enumerate(EVAL):
            v = run_means([c for c in E1 if c['dataset'] == dname and f(c)], f'err_{m}', lambda c: 0).get(0, [])
            mu_, l_, h_ = mean_ci(v)
            rows.append(dict(family=fam, dataset=dname, method=m, mean=mu_, ci_low=l_, ci_high=h_, runs=len(v)))
            xs.append(di + (i - 2) * 0.14); ys.append(mu_); lo.append(mu_ - l_); hi.append(h_ - mu_)
        ax[j].errorbar(xs, ys, yerr=[lo, hi], fmt=MMARK[m], color=MCOL[m], ms=3, lw=0.8, capsize=1.5,
                       label=MLAB[m] if j == 0 else None)
    ax[j].set_yscale('log'); ax[j].set_xticks(range(len(EVAL)))
    ax[j].set_xticklabels([DATASET_SHORT[d] for d in EVAL], fontsize=6)
    ax[j].set_ylabel(r'$\|\hat w-w\|_2$'); ax[j].set_xlabel('tweak-one shifts' if fam == 'tweak' else 'Dirichlet shifts')
fig.legend(loc='upper center', ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.06))
save_fig(fig, ax, 'Figure_S1_main', rows)

# ---- Fig. 4: error against shift severity ----
rows = []
fig, ax = plt.subplots(1, 2, figsize=(W, 2.3))
k10 = [c for c in E1 if c['K'] == 10]
for j, (fam, pars) in enumerate((('tweak', TWEAK_RHOS), ('dirichlet', DIR_ALPHAS))):
    for m in ['naive', 'clsc', 'plain', 'mlls']:
        ys, lo, hi = [], [], []
        for par in pars:
            v = run_means([c for c in k10 if c['shift'] == fam and c['par'] == par], f'err_{m}', lambda c: 0).get(0, [])
            mu_, l_, h_ = mean_ci(v); ys.append(mu_); lo.append(mu_ - l_); hi.append(h_ - mu_)
            rows.append(dict(family=fam, par=par, method=m, mean=mu_, ci_low=l_, ci_high=h_, runs=len(v)))
        ax[j].errorbar(pars, ys, yerr=[lo, hi], marker=MMARK[m], color=MCOL[m], ms=3, lw=0.9, capsize=1.5, label=MLAB[m])
    ax[j].set_xlabel(r'severity $\rho$ (tweak-one)' if fam == 'tweak' else r'concentration $\alpha$ (Dirichlet)')
    ax[j].set_ylabel(r'$\|\hat w-w\|_2$')
ax[1].set_xscale('log'); plain_ticks(ax[1].xaxis, DIR_ALPHAS); plain_ticks(ax[0].xaxis, TWEAK_RHOS); ax[0].legend()
save_fig(fig, ax, 'Figure_S2_severity', rows)

# ---- Fig. 5: sample-size sweep ----
rows = []
fig, ax = plt.subplots(1, 2, figsize=(W, 2.3))
e2k = [c for c in E2 if c['K'] == 10]
for j, sw in enumerate(('n_src', 'n_te')):
    cc = [c for c in e2k if c['sweep'] == sw]
    key = (lambda c: c['n_src']) if sw == 'n_src' else (lambda c: c['n_te'])
    grid = [x for x in (NSWEEP_SRC if sw == 'n_src' else NSWEEP_TE) if x is not None]
    xs = [x for x in grid if any(key(c) == x for c in cc)]      # sizes common to every dataset
    for m, k_ in (('naive', 'err_naive'), ('clsc', 'err_clsc'), ('plain', 'err_plain_unproj')):
        ys = [np.median(run_means([c for c in cc if key(c) == x], k_, lambda c: 0).get(0, [np.nan])) for x in xs]
        ax[j].plot(xs, ys, marker=MMARK[m], color=MCOL[m], ms=3, lw=0.9, label=MLAB[m] + (' (unprojected)' if m == 'plain' else ''))
        rows += [dict(sweep=sw, n=x, series=m, value=y) for x, y in zip(xs, ys)]
    ys = [np.median(run_means([c for c in cc if key(c) == x], 'pred_rmse_plain', lambda c: 0).get(0, [np.nan])) for x in xs]
    ax[j].plot(xs, ys, 'k--', lw=0.8, label='Theorem 3 prediction')
    rows += [dict(sweep=sw, n=x, series='theorem3', value=y) for x, y in zip(xs, ys)]
    ax[j].set_xscale('log'); ax[j].set_yscale('log'); ax[j].set_ylabel(r'$\|\hat w-w\|_2$')
    plain_ticks(ax[j].xaxis, xs, '{:,.0f}')
    ax[j].set_xlabel('source size $n_{src}$' if sw == 'n_src' else 'target size $n_{te}$')
ax[0].legend()
save_fig(fig, ax, 'Figure_3_samplesize', rows)

# ---- Fig. 6: accuracy sweep ----
rows = []
fig, ax = plt.subplots(1, 2, figsize=(W, 2.3))
for c in E3:
    b_ = (1 - c['a_hat']) / (c['K'] - 1); c['delta_hat'] = 1 - (c['a_hat'] - b_)
g6 = {}
for c in E3: g6.setdefault(c['tag'], []).append(c)
for t, cs in g6.items():
    rows.append(dict(tag=t, dataset=cs[0]['dataset'], noise=cs[0]['noise'], seed=cs[0]['seed'],
                     delta_hat=float(np.mean([c['delta_hat'] for c in cs])),
                     bias_naive=float(np.mean([c['bias_naive'] for c in cs])),
                     **{f'err_{m}': float(np.nanmean([c.get(f'err_{m}', np.nan) for c in cs]))
                        for m in ('naive', 'clsc', 'plain', 'mlls', 'bbse_clean')}))
for m in ('naive', 'clsc', 'mlls', 'plain'):
    ax[0].plot([r['delta_hat'] for r in rows], [r[f'err_{m}'] for r in rows], MMARK[m], color=MCOL[m], ms=3, label=MLAB[m])
ax[0].set_yscale('log'); ax[0].set_xlabel(r'shrinkage coefficient $\hat\delta$'); ax[0].set_ylabel(r'$\|\hat w-w\|_2$')
ax[0].legend()
ax[1].plot([r['bias_naive'] for r in rows], [r['err_naive'] - r['err_clsc'] for r in rows], 'o', color=MCOL['clsc'], ms=3)
ax[1].axhline(0, color='k', lw=0.7); ax[1].set_xlabel('naive bias (population)'); ax[1].set_ylabel('error(naive) $-$ error(CL-SC)')
save_fig(fig, ax, 'Figure_5_accuracy_sweep', rows)

# ---- Fig. 7: transition-matrix sensitivity ----
rows = []
fig, ax = plt.subplots(1, 3, figsize=(W, 2.3))
qk = ['uniform'] + QTRAIN_KINDS
for m in ('naive', 'clsc', 'clsc_general', 'plain'):
    ys = []
    for k_ in qk:
        v = run_means([c for c in E4a if c['qkind'] == k_], f'err_{m}', lambda c: 0).get(0, [np.nan])
        ys.append(np.mean(v)); rows.append(dict(panel='a_trained', q=k_, eps=0.0, method=m, value=ys[-1]))
    ax[0].plot(range(len(qk)), ys, marker=MMARK.get(m, 'x'), color=MCOL.get(m, '#7f3fbf'), ms=3, lw=0.9,
               label=MLAB.get(m, r'CL-SC, $\hat a=\mathrm{tr}(\hat{\bar C}\hat Q^{-1})$'))
    ys = []
    for k_ in ('uniform', 'mild', 'strong'):
        v = run_means([c for c in E4bc if c['q_draw'] == k_ and c['eps'] == 0.0], f'err_{m}', lambda c: 0).get(0, [np.nan])
        ys.append(np.mean(v)); rows.append(dict(panel='b_redrawn', q=k_, eps=0.0, method=m, value=ys[-1]))
    ax[1].plot(range(3), ys, marker=MMARK.get(m, 'x'), color=MCOL.get(m, '#7f3fbf'), ms=3, lw=0.9)
    ys = []
    for e in QSENS_EPS:
        v = run_means([c for c in E4bc if c['q_draw'] == 'mild' and c['eps'] == e], f'err_{m}', lambda c: 0).get(0, [np.nan])
        ys.append(np.mean(v)); rows.append(dict(panel='c_misspecified', q='mild', eps=e, method=m, value=ys[-1]))
    ax[2].plot(QSENS_EPS, ys, marker=MMARK.get(m, 'x'), color=MCOL.get(m, '#7f3fbf'), ms=3, lw=0.9)
ax[0].set_xticks(range(len(qk))); ax[0].set_xticklabels(qk); ax[0].set_xlabel('Q used in training')
ax[1].set_xticks(range(3)); ax[1].set_xticklabels(['uniform', 'mild', 'strong']); ax[1].set_xlabel('Q of re-drawn labels')
ax[2].set_xlabel(r'misspecification $\varepsilon$ of $\hat Q$')
for a_ in ax: a_.set_ylabel(r'$\|\hat w-w\|_2$')
ax[0].legend(fontsize=5.5)
save_fig(fig, ax, 'Figure_7_Q', rows)

# ---- Fig. 8: critical-difference diagram ----
fig, ax = plt.subplots(1, 1, figsize=(3.5, 1.3))
order = np.argsort(avg_rank)
ax.set_xlim(len(RANK_METHODS) + 0.3, 0.7); ax.set_ylim(0, 1); ax.set_yticks([]); ax.grid(False)
for s in ('left', 'right', 'top'): ax.spines[s].set_visible(False)
for i, j in enumerate(order):
    ax.plot([avg_rank[j]] * 2, [0, 0.35 + 0.12 * i], color=MCOL[RANK_METHODS[j]], lw=0.8)
    ax.text(avg_rank[j], 0.38 + 0.12 * i, f"{MLAB[RANK_METHODS[j]]} ({avg_rank[j]:.2f})", ha='center', fontsize=6)
if np.isfinite(CD):
    ax.plot([1, 1 + CD], [0.92, 0.92], 'k-', lw=1.2); ax.text(1 + CD / 2, 0.95, f'CD = {CD:.2f}', ha='center', fontsize=6)
    srt = np.sort(avg_rank); cliques = []
    for i in range(len(srt)):                  # maximal runs of methods within CD of each other
        j = i
        while j + 1 < len(srt) and srt[j + 1] - srt[i] <= CD: j += 1
        if j > i and not any(c[0] <= i and j <= c[1] for c in cliques): cliques.append((i, j))
    for n_, (i, j) in enumerate(cliques):
        ax.plot([srt[i] - 0.03, srt[j] + 0.03], [0.12 + 0.06 * n_] * 2, 'k-', lw=2.0)
ax.set_xlabel('average rank over runs (1 = best)')
save_fig(fig, [ax], 'Figure_4_cd', CDROWS + [dict(method='CD', avg_rank=CD)])

# ---- Fig. 9: downstream calibration and NLL ----
rows = []
fig, ax = plt.subplots(1, 2, figsize=(W, 2.4))
for j, metric in enumerate(('ece', 'nll')):
    for i, m in enumerate(DS_METHODS):
        xs, ys = [], []
        for di, dname in enumerate(EVAL):
            v = run_means([c for c in E1 if c['dataset'] == dname and c['shift'] == 'tweak'],
                          f'{metric}_That_{m}', lambda c: 0).get(0, [np.nan])
            mu_ = float(np.nanmean(v)); xs.append(di + (i - 2.5) * 0.12); ys.append(mu_)
            rows.append(dict(metric=metric, temperature='That', dataset=dname, method=m, value=mu_))
        ax[j].plot(xs, ys, MMARK.get(m, 'x'), color=MCOL.get(m, '#999999' if m == 'none' else '#000000'), ms=3,
                   label=(MLAB.get(m, m)) if j == 0 else None)
    ax[j].set_xticks(range(len(EVAL)))
    ax[j].set_xticklabels([DATASET_SHORT[d] for d in EVAL], fontsize=6)
    ax[j].set_ylabel('ECE' if metric == 'ece' else 'NLL'); ax[j].set_xlabel('dataset (tweak-one shifts)')
fig.legend(loc='upper center', ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.06))
save_fig(fig, ax, 'Figure_S3_downstream', rows)

# ---- Fig. 10: Proposition 1 ----
fig, ax = plt.subplots(1, 2, figsize=(W, 2.2))
ax[0].plot(tgrid, [r['risk_target'] for r in P1C], color=MCOL['clsc'], lw=1, label='target risk')
ax[0].plot(tgrid, [r['risk_source'] for r in P1C], color=MCOL['naive'], lw=1, ls='--', label='source risk')
ax[0].axvline(0.0, color=MCOL['naive'], lw=0.6); ax[0].axvline([r for r in P1R if r['w0'] == 0.2][0]['t_target'], color=MCOL['clsc'], lw=0.6)
ax[0].set_xlabel('threshold $t$'); ax[0].set_ylabel('0-1 risk'); ax[0].legend()
ax[1].plot([r['norm_w_minus_1'] for r in P1R], [r['excess'] for r in P1R], 'o-', color=MCOL['plain'], ms=3, lw=0.9)
ax[1].set_xlabel(r'$\|w-\mathbf{1}\|_2$'); ax[1].set_ylabel('excess target risk')
save_fig(fig, ax, 'Figure_A1_prop', P1R)

# ---- Fig. 11: label-free applicability rule ----
rows = []
fig, ax = plt.subplots(1, 3, figsize=(W, 2.3))
tw_main = [c for c in main_cells if c['shift'] == 'tweak']
for dname in EVAL:
    cc = [c for c in tw_main if c['dataset'] == dname]
    if not cc: continue
    ax[0].plot([c['misfit_w'] for c in cc], [c['D_hat'] for c in cc], '.', ms=3, label=DATASET_SHORT[dname])
    rows += [dict(panel='a', dataset=dname, tag=c['tag'], fold=c['fold'], rho=c['par'],
                  misfit_clean=c['misfit_w'], D_hat=c['D_hat']) for c in cc]
lim = [0, max(ax[0].get_xlim()[1], ax[0].get_ylim()[1])]
ax[0].plot(lim, lim, 'k--', lw=0.7); ax[0].set_xlabel('misfit with clean labels'); ax[0].set_ylabel(r'label-free estimate $\hat D$')
ax[0].legend(fontsize=5.5, markerscale=2)
for dname in EVAL:
    rr = [r for r in LODO if r['dataset'] == dname]
    if not rr: continue
    ax[1].plot([r['D_hat'] for r in rr], [r['err_naive'] - r['err_clsc'] for r in rr], '.', ms=4)
    rows += [dict(panel='b', dataset=dname, tag=r['tag'], D_hat=r['D_hat'], gain=r['err_naive'] - r['err_clsc']) for r in rr]
ax[1].axhline(0, color='k', lw=0.6); ax[1].axvline(TAU_STAR, color=MCOL['plain'], lw=0.8, ls='--')
ax[1].set_xlabel(r'$\hat D$ (per run)'); ax[1].set_ylabel('error(naive) $-$ error(CL-SC)')
dd = [r for r in E10T if r['dataset'] != 'all']
for i, (m, col) in enumerate((('naive', MCOL['naive']), ('clsc', MCOL['clsc']), ('rule', '#7f3fbf'))):
    ax[2].bar(np.arange(len(dd)) + (i - 1) * 0.27, [r[m] for r in dd], 0.27, color=col,
              label={'naive': 'naive', 'clsc': 'CL-SC', 'rule': 'rule (held-out dataset)'}[m])
    rows += [dict(panel='c', dataset=r['dataset'], method=m, value=r[m]) for r in dd]
ax[2].set_xticks(range(len(dd))); ax[2].set_xticklabels([DATASET_SHORT[r['dataset']] for r in dd], fontsize=6,
                                                        rotation=30, ha='right')
ax[2].set_yscale('log'); ax[2].set_ylabel(r'$\|\hat w-w\|_2$'); ax[2].set_xlabel('dataset'); ax[2].legend(fontsize=5.5)
save_fig(fig, ax, 'Figure_6_rule', rows)

# ---- Supplementary figures ----
rows = []
fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.3))
abl = ['clsc', 'clsc_oracle_a', 'clsc_noproj', 'clsc_general', 'clsc_insample', 'select', 'naive',
       'plain', 'plain_insample', 'mlls', 'mlls_T1', 'mlls_Toracle']
vals = [float(np.mean(run_means(E1, f'err_{m}', lambda c: 0).get(0, [np.nan]))) for m in abl]
rows = [dict(variant=m, mean_err=v) for m, v in zip(abl, vals)]
ax.barh(range(len(abl)), vals, color=['#1f5fa8' if a.startswith('clsc') or a == 'select' else '#999999' for a in abl])
ax.set_yticks(range(len(abl))); ax.set_yticklabels(abl, fontsize=6); ax.invert_yaxis(); ax.set_xlabel(r'mean $\|\hat w-w\|_2$, all runs')
save_fig(fig, [ax], 'Figure_S4_ablations', rows)

fig, ax = plt.subplots(1, 2, figsize=(W, 2.2))
rows = []
for j, part in enumerate(('global', 'spread')):
    for i, nm in enumerate(('uncal', 'single', 'perclass')):
        ys = [float(np.mean([c[f'{nm}_{part}'] for c in CAL if c['dataset'] == d] or [np.nan])) for d in EVAL]
        ax[j].bar(np.arange(len(EVAL)) + (i - 1) * 0.27, ys, 0.27, label=nm if j == 0 else None)
        rows += [dict(part=part, calibration=nm, dataset=d, value=y) for d, y in zip(EVAL, ys)]
    ax[j].set_xticks(range(len(EVAL))); ax[j].set_xticklabels([DATASET_SHORT[d] for d in EVAL], fontsize=6)
    ax[j].set_ylabel('global gap' if part == 'global' else 'class-dependent spread'); ax[j].set_xlabel('dataset')
ax[0].legend()
save_fig(fig, ax, 'Figure_S5_calibration', rows)

fig, ax = plt.subplots(1, 2, figsize=(W, 2.3))
rows = []
for m in ('naive', 'clsc', 'plain', 'mlls'):
    a_ = [float(np.mean(run_means([c for c in E1 if c['dataset'] == d and c['shift'] == 'tweak'], f'err_{m}', lambda c: 0).get(0, [np.nan]))) for d in EVAL]
    b_ = [float(np.mean(run_means([c for c in E7 if c['dataset'] == d], f'err_{m}', lambda c: 0).get(0, [np.nan]))) for d in EVAL]
    ax[0].plot(a_, b_, MMARK[m], color=MCOL[m], ms=3.5, label=MLAB[m])
    rows += [dict(method=m, dataset=d, crossfit=x, testpool=y) for d, x, y in zip(EVAL, a_, b_)]
lim = [min(ax[0].get_xlim()[0], ax[0].get_ylim()[0]), max(ax[0].get_xlim()[1], ax[0].get_ylim()[1])]
ax[0].plot(lim, lim, 'k--', lw=0.7); ax[0].set_xlabel('error, cross-fitted protocol'); ax[0].set_ylabel('error, test-pool protocol'); ax[0].legend()
gaps = [(DATASET_SHORT[r['dataset']], r['val_acc'] - r['test_acc']) for r in t1 if r['dataset'] in EVAL]
ax[1].bar(range(len(gaps)), [g[1] for g in gaps], color='#777777')
ax[1].set_xticks(range(len(gaps))); ax[1].set_xticklabels([g[0] for g in gaps], fontsize=6)
ax[1].set_ylabel('accuracy: held-out minus test pool'); ax[1].set_xlabel('dataset')
rows += [dict(method='acc_gap', dataset=g[0], crossfit=g[1], testpool=np.nan) for g in gaps]
save_fig(fig, ax, 'Figure_S6_testpool', rows)

if IW:
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.2)); rows = []
    for model, ms, sty in (('iw_nncll', runs('iw', healthy=False), dict(color=MCOL['plain'])),
                           ('unweighted', [b for b in runs('main') if b['dataset'] == IW_DATASET and b['seed'] in SEEDS_IW],
                            dict(color=MCOL['naive'], ls='--'))):
        for m in ms:
            rc = np.load(m['path'], allow_pickle=True)['risk_curve']; ax.plot(np.arange(1, len(rc) + 1), rc, lw=0.8, **sty)
            rows += [dict(model=model, seed=m['seed'], epoch=e + 1, risk=float(v)) for e, v in enumerate(rc)]
    ax.set_xlabel('epoch'); ax.set_ylabel('corrected training risk')
    save_fig(fig, [ax], 'Figure_S7_iw_risk', rows)

# %% [markdown]
# ## 21. Tables (CSV and LaTeX)
#
# LaTeX tables use `booktabs`, no vertical rules, and columns sized to their content.

# %%
def to_latex(rows, cols, heads, path, fmt=None):
    fmt = fmt or {}
    spec = 'l' + 'r' * (len(cols) - 1)
    L = ['\\begin{tabular}{' + spec + '}', '\\toprule', ' & '.join(heads) + ' \\\\', '\\midrule']
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, '')
            cells.append(fmt[c](v) if c in fmt else (f'{v:.3f}' if isinstance(v, float) else str(v)))
        L.append(' & '.join(cells) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}']
    open(path, 'w').write('\n'.join(L) + '\n')

to_latex(t1, ['label', 'K', 'n_train', 'n_heldout', 'prior_ratio', 'healthy', 'val_acc', 'test_acc'],
         ['Dataset', '$K$', 'train', 'held-out', 'prior max/min', 'healthy', 'acc. (held-out)', 'acc. (test)'],
         os.path.join(DIRS['tables'], 'table1_datasets.tex'),
         {'healthy': lambda v: str(v), 'prior_ratio': lambda v: f'{v:.1f}', 'n_train': str, 'n_heldout': str})
T2 = []
for dname in MAIN_DATASETS:
    r = dict(dataset=DATASET_LABEL[dname])
    rr = [x for x in LODO if x['dataset'] == dname]
    r['rule'] = (f"{np.mean([x['err_rule'] for x in rr]):.3f} $\\pm$ {np.std([x['err_rule'] for x in rr], ddof=1):.3f}"
                 if len(rr) > 1 else '--')
    for m in ['naive', 'clsc', 'plain', 'mlls', 'bbse_clean']:
        v = list(per_run([c for c in E1 if c['dataset'] == dname], f'err_{m}').values())
        r[m] = f'{np.nanmean(v):.3f} $\\pm$ {np.nanstd(v, ddof=1) if len(v) > 1 else 0:.3f}'
        r[m + '_mean'] = float(np.nanmean(v)) if v else np.nan
    T2.append(r)
write_csv(os.path.join(RES_DIR, 'table2_weight_error.csv'), T2)
to_latex([t for t in T2 if t['naive'] != 'nan $\\pm$ nan'], ['dataset', 'naive', 'clsc', 'rule', 'plain', 'mlls', 'bbse_clean'],
         ['Dataset', 'naive', 'CL-SC', 'rule', 'plain CL-BBSE', 'MLLS', 'BBSE (clean)'], os.path.join(DIRS['tables'], 'table2_weight_error.tex'))
T3 = [s for s in STATS if s['family'] == 'all']
to_latex(T3, ['dataset', 'competitor', 'runs', 'mean_diff', 'ci_low', 'ci_high', 'p_holm', 'cliffs_delta', 'wins', 'losses'],
         ['Subset', 'vs', 'runs', '$\\Delta$', 'CI low', 'CI high', '$p_{\\mathrm{Holm}}$', 'Cliff', 'W', 'L'],
         os.path.join(DIRS['tables'], 'table3_tests.tex'), {'p_holm': lambda v: f'{v:.1e}', 'runs': str, 'wins': str, 'losses': str})
T4 = []
for dname in MAIN_DATASETS:
    for m in DS_METHODS:
        cc = [c for c in E1 if c['dataset'] == dname and c['shift'] == 'tweak']
        T4.append(dict(dataset=DATASET_LABEL[dname], method=m,
                       **{f'{k}': float(np.nanmean(list(per_run(cc, f'{k}_That_{m}').values()))) for k in ('acc', 'nll', 'ece', 'brier')}))
write_csv(os.path.join(RES_DIR, 'table4_downstream.csv'), T4)
to_latex(T4, ['dataset', 'method', 'acc', 'nll', 'ece', 'brier'], ['Dataset', 'weights', 'acc.', 'NLL', 'ECE', 'Brier'],
         os.path.join(DIRS['tables'], 'table4_downstream.tex'), {'ece': lambda v: f'{v:.4f}'})
T5 = []
ds_ok = [d for d in MAIN_DATASETS if any(c['dataset'] == d for c in E1)]
for m in abl:
    row = dict(variant=m)
    for d in ds_ok:
        row[d] = float(np.mean(list(per_run([c for c in E1 if c['dataset'] == d], f'err_{m}').values())))
    T5.append(row)
write_csv(os.path.join(RES_DIR, 'table5_ablations.csv'), T5)
to_latex(T5, ['variant'] + ds_ok, ['Variant'] + [DATASET_SHORT[d] for d in ds_ok],
         os.path.join(DIRS['tables'], 'table5_ablations.tex'))
to_latex(E10T, ['dataset', 'tau', 'frac_clsc', 'naive', 'clsc', 'rule', 'best_of_two'],
         ['Held-out dataset', '$\\tau$', 'CL-SC chosen', 'naive', 'CL-SC', 'rule', 'best of two'],
         os.path.join(DIRS['tables'], 'table7_rule.tex'), {'dataset': lambda v: DATASET_SHORT.get(v, v),
                                                          'frac_clsc': lambda v: f'{100 * v:.0f}\\%'})
T6 = []
for g in ('main', 'noise', 'qtrain', 'iw', 'collapse'):
    for dname in sorted({m['dataset'] for m in MAN if m['group'] == g}):
        ms = [m for m in MAN if m['group'] == g and m['dataset'] == dname]
        T6.append(dict(group=g, dataset=dname, runs=len(ms), minutes_per_run=float(np.mean([float(m['seconds']) for m in ms]) / 60)))
write_csv(os.path.join(RES_DIR, 'table6_runtime.csv'), T6)
to_latex(T6, ['group', 'dataset', 'runs', 'minutes_per_run'], ['Group', 'Dataset', 'runs', 'min/run'],
         os.path.join(DIRS['tables'], 'table6_runtime.tex'), {'runs': str, 'minutes_per_run': lambda v: f'{v:.1f}'})
print('tables written:', sorted(os.listdir(DIRS['tables'])))

# %% [markdown]
# ## 22. Summary of quoted numbers and packaging
#
# `summary.json` collects every number quoted in the text so that the manuscript can be checked
# against a single file. `results_partB.zip` holds `output/results`, `fig` and `tables`.

# %%
def gm(cells, key, filt=lambda c: True):
    v = list(per_run([c for c in cells if filt(c)], key).values()); return float(np.nanmean(v)) if v else np.nan
k10 = lambda c: c['K'] == 10
tw = [c for c in E1 if c['shift'] == 'tweak']
ratio = np.array([c['err_plain_unproj'] / c['pred_rmse_plain'] for c in tw
                  if np.isfinite(c.get('err_plain_unproj', np.nan)) and np.isfinite(c['pred_rmse_plain']) and c['pred_rmse_plain'] > 0])
SUMMARY = dict(
    profile=PROFILE, cfg_sig=CFG_SIG, runs_total=len(MAN), runs_healthy=sum(m['healthy'] for m in MAN),
    mean_err_K10={m: gm(E1, f'err_{m}', k10) for m in METHODS_MAIN},
    mean_err_all={m: gm(E1, f'err_{m}') for m in METHODS_MAIN},
    failures={m: int(sum(c.get(f'fail_{m}', 0) for c in E1)) for m in METHODS_MAIN + ABLATIONS},
    theorem3_median_measured_over_pred=float(np.median(ratio)) if len(ratio) else np.nan,
    theorem3_spearman=float(spearmanr([c['pred_rmse_plain'] for c in tw], [c.get('err_plain_unproj', np.nan) for c in tw],
                                      nan_policy='omit').correlation) if tw else np.nan,
    theorem3_winner_agreement=int(sum(((c['pred_rmse_plain'] > c['bias_naive']) == (c['err_plain'] > c['err_naive'])) for c in tw)),
    theorem3_cells=len(tw),
    smin_lemma_over_measured_median=float(np.median([c['smin_lemma'] / c['smin'] for c in tw if c['smin'] > 0])),
    montecarlo_rmse_over_pred=[(r['K'], r['a'], r['regime'], r['n'], r['mc_rmse'] / r['pred_rmse']) for r in MC],
    friedman=dict(p=float(fr.pvalue) if fr else np.nan, CD=CD, avg_ranks={m: float(r) for m, r in zip(RANK_METHODS, avg_rank)}),
    calibration={k: float(np.mean([c[k] for c in CAL])) for k in CAL[0] if k.split('_')[0] in ('uncal', 'single', 'perclass')} if CAL else {},
    prop1_excess_w02=[r for r in P1R if r['w0'] == 0.2][0]['excess'],
    rule=dict(tau_star=TAU_STAR, D_hat_vs_misfit_spearman=float(diag_corr), D_hat_vs_misfit_spearman_runs=float(diag_corr_run),
              D_hat_vs_gain_spearman_runs=float(gain_corr_run), lodo=E10T, tests=E10S, out_of_sample=OOS),
    elapsed_h_this_session=elapsed_h())
save_json(SUMMARY, os.path.join(RES_DIR, 'summary.json'))
zp = os.path.join(ROOT, 'results_partB.zip')
with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as zf:
    for d in (RES_DIR, DIRS['fig'], DIRS['tables']):
        for f in sorted(glob.glob(os.path.join(d, '*'))):
            if os.path.isfile(f): zf.write(f, os.path.relpath(f, ROOT))
print(json.dumps({k: SUMMARY[k] for k in ('runs_total', 'runs_healthy', 'mean_err_K10', 'theorem3_median_measured_over_pred',
                                          'theorem3_winner_agreement', 'theorem3_cells')}, indent=1, default=str))
print(f'{zp}: {os.path.getsize(zp) / 1e6:.1f} MB')
