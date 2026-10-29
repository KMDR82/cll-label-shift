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
    ra = [r for r in rr if r['in_panel_a'] and r['rho'] == min(TWEAK_RHOS)]
    if ra:
        xi = EVAL.index(dname); jit = rng_for('jit', dname).uniform(-0.18, 0.18, len(ra))
        ratio = [r['smin'] / r['smin_lemma'] for r in ra]
        ax[0].plot(xi + jit, ratio, '.', ms=3, color=f'C{xi}')
        ax[0].plot([xi - 0.3, xi + 0.3], [np.median(ratio)] * 2, 'k-', lw=1.0)
    ax[1].plot([], [], '.', ms=3, color=f'C{EVAL.index(dname)}', label=DATASET_SHORT[dname])
    col = f'C{EVAL.index(dname)}'
    ax[1].plot([r['bias_thm'] for r in rr], [r['bias_naive'] for r in rr], '.', ms=3, color=col)
    ax[2].loglog([r['pred_rmse_plain'] for r in rr], [r['err_plain_unproj'] for r in rr], '.', ms=3, color=col)
ina = [d for d in EVAL if any(r['dataset'] == d and r['in_panel_a'] for r in rows)]
ax[0].axhline(1.0, color='k', ls='--', lw=0.7); ax[0].set_ylim(0, 1.1)
ax[0].set_xticks([EVAL.index(d) for d in ina]); ax[0].set_xticklabels([DATASET_SHORT[d] for d in ina], fontsize=6)
ax[0].set_xlim(-0.6, max(EVAL.index(d) for d in ina) + 0.6)
ax[0].set_xlabel('dataset'); ax[0].set_ylabel(r'measured / closed-form $\sigma_{\min}(\bar C)$')
for a_, lab in zip(ax[1:], (('prediction (Theorem 2)', 'measured naive bias'),
                            ('prediction (Theorem 3)', 'measured plain CL-BBSE error'))):
    lim = [min(a_.get_xlim()[0], a_.get_ylim()[0]), max(a_.get_xlim()[1], a_.get_ylim()[1])]
    a_.plot(lim, lim, 'k--', lw=0.7); a_.set_xlabel(lab[0]); a_.set_ylabel(lab[1])
ax[1].legend(fontsize=5.5, markerscale=2, loc='upper left')
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

