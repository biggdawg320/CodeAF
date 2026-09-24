#!/usr/bin/env python3
"""Draws every figure and generated table of pareto-crewing.tex from data/*.

    python3 figures.py        # writes fig-*.pdf and tab-*.tex beside this file

Reads only files under data/ (rebuilt from the research workspace by data/extract.py). Needs numpy, pandas,
matplotlib. Style, sizes and palette come from paperstyle.py; every figure is saved at its final printed size.
"""
import json, os, re

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FixedLocator, FuncFormatter, MultipleLocator

import paperstyle as ps
from paperstyle import PAL, TEXT_W, COL_W, log_axis, title, refline

ps.apply()
HERE = os.path.dirname(os.path.abspath(__file__))
D = lambda f: os.path.join(HERE, "data", f)
OUT = lambda f: os.path.join(HERE, f)
C = PAL


def frontier(pts):
    pts = np.asarray(pts, float)
    pts = pts[np.isfinite(pts).all(1)]
    o = np.lexsort((-pts[:, 1], pts[:, 0])); pts = pts[o]
    prev = np.maximum.accumulate(np.r_[-np.inf, pts[:-1, 1]])
    par = pts[pts[:, 1] > prev + 1e-15]
    h = []
    for p in par:
        while len(h) >= 2:
            (x1, y1), (x2, y2) = h[-2], h[-1]
            if (y2 - y1) * (p[0] - x1) <= (p[1] - y1) * (x2 - x1) + 1e-15:
                h.pop()
            else:
                break
        h.append(p)
    return np.array(h)


def cost_to_reach(V, t):
    if len(V) == 0 or V[-1, 1] < t:
        return np.nan
    if V[0, 1] >= t:
        return V[0, 0]
    i = int(np.searchsorted(V[:, 1], t))
    x0, x1, y0, y1 = V[i - 1, 0], V[i, 0], V[i - 1, 1], V[i, 1]
    return x0 + (t - y0) * (x1 - x0) / (y1 - y0)


def f_at(V, x):
    return np.interp(x, V[:, 0], V[:, 1])


def save(fig, name):
    fig.savefig(OUT(name))
    plt.close(fig)
    print("wrote", name)


MK = dict(marker="o", mec="white", mew=0.5)

# ------------------------------------------------------------------ data
aiq = pd.read_csv(D("seq_aiq.csv"))
crv = pd.read_csv(D("seq_curves.csv"))
orc = pd.read_csv(D("seq_oracle.csv"))
th = pd.read_csv(D("theory_twostage.csv"))
lb = pd.read_csv(D("leaderboard_summary.csv"))
ss = pd.read_csv(D("singleshot_pairs.csv"))
pol = pd.read_csv(D("trial_policies.csv"))
ca = pd.read_csv(D("trial_class_arm.csv"))
con = pd.read_csv(D("trial_contrasts.csv"))
cand = pd.read_csv(D("trial_candidates.csv"))

ARM = {  # configuration -> (short label, colour, is diagonal)
    "B": ("uniform low-cost", C["diag_low"], True),
    "D": ("uniform frontier", C["diag_front"], True),
    "C": ("low-cost + strong checker", C["verify"], False),
    "E": ("low-cost + low-cost checker", C["verify_light"], False),
    "F": ("alt. low-cost worker", C["alt"], False),
}


def num(s):
    try:
        return float(str(s).split("±")[0].replace("+", "").strip())
    except ValueError:
        return np.nan


def g_pop(r):
    return np.interp(r, th.r.values, th.gain.values)


def r_at(target):
    x, y = th.r.values, th.gain.values
    i = int(np.argmax(y >= target))
    return x[i - 1] + (target - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1])


# ------------------------------------------------------------------ figures
def fig_crew_frontier():
    """Crew frontier vs diagonal (uniform-crew) frontier on the real-issue trial."""
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_W, 2.75), gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
    specs = [("bug-fix", "defect fixes ($n=14$)", [0.02, 0.05, 0.1, 0.2, 0.4], (0.017, 0.5)),
             ("open-ended", "open-ended tasks ($n=8$)", [0.04, 0.06, 0.1], (0.035, 0.14))]
    for k, (ax, (cl, ttl, ticks, xl)) in enumerate(zip(axes[:2], specs)):
        s = ca[ca.cls == cl].set_index("arm")
        V = frontier(np.c_[s.cost.values, s.score.values])
        ax.plot(V[:, 0], V[:, 1], color=C["ours"], lw=3.2, alpha=0.45, zorder=2, solid_capstyle="round")
        dg = s[[ARM[a][2] for a in s.index]]
        if len(dg) > 1:
            Vd = frontier(np.c_[dg.cost.values, dg.score.values])
            ax.plot(Vd[:, 0], Vd[:, 1], color=C["diag_low"], lw=1.0, ls=(0, (4, 2)), zorder=2.5)
        for a, r in s.iterrows():
            lab, col, diag = ARM[a]
            ax.errorbar(r.cost, r.score, yerr=[[r.score - r.lo], [r.hi - r.score]], fmt="s" if diag else "o", color=col,
                        ms=4.4, mec="white", mew=0.5, elinewidth=0.8, capsize=0, zorder=3)
        log_axis(ax, "x", ticks); ax.set_xlim(*xl)
        ax.set_ylim(2.5, 8.8); ax.yaxis.set_major_locator(MultipleLocator(2))
        ax.set_xlabel("cost per issue (log)")
        title(ax, "ab"[k], ttl)
    axes[0].set_ylabel("review score (0–10)")
    ax = axes[2]
    P = pol.set_index("policy")
    # label anchors in data coordinates, placed clear of every error bar
    items = [("uniform low-cost crew", "uniform low-cost", C["diag_low"], "s", ("pt", 0, -0.12), "center", "top"),
             ("low-cost crew + low-cost checker", "low-cost checker", C["verify_light"], "o", ("abs", 0.047, 5.72), "left", "center"),
             ("strong-checker crew", "strong checker", C["verify"], "o", ("pt", 0, 0.15), "center", "bottom"),
             ("routed (class-aware)", "class-aware crew", C["ours"], "D", ("abs", 0.051, 7.35), "right", "center")]
    for key, lab, col, mk, (mode, tx, ty), ha, va in items:
        r = P.loc[key]
        ax.errorbar(r.cost, r.score, xerr=[[r.cost - r.cost_lo], [r.cost_hi - r.cost]], yerr=[[r.score - r.lo], [r.hi - r.score]],
                    fmt=mk, color=col, ms=4.8 if mk == "D" else 4.4, mec="white", mew=0.5, elinewidth=0.8, capsize=0, zorder=3)
        if mode == "pt":
            x, y = r.cost, (r.lo + ty if ty < 0 else r.hi + ty)
        else:
            x, y = tx, ty
        ax.text(x, y, f"{lab}\n{int(r.mergeable)}/22 mergeable", fontsize=6.3, color=col, ha=ha, va=va, linespacing=1.05)
    log_axis(ax, "x", [0.02, 0.05, 0.1]); ax.set_xlim(0.016, 0.14)
    ax.set_ylim(2.5, 8.8); ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.set_xlabel("cost per issue (log)")
    title(ax, "c", "all issues ($n=22$), policies")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=C["ours"], lw=3.2, alpha=0.45, label="crew frontier"),
         Line2D([], [], color=C["diag_low"], lw=1.0, ls=(0, (4, 2)), label="diagonal frontier"),
         Line2D([], [], color=C["diag_low"], marker="s", ls="none", mec="white", label="uniform low-cost (diagonal)"),
         Line2D([], [], color=C["diag_front"], marker="s", ls="none", mec="white", label="uniform frontier (diagonal)"),
         Line2D([], [], color=C["verify_light"], marker="o", ls="none", mec="white", label="low-cost worker + low-cost checker"),
         Line2D([], [], color=C["verify"], marker="o", ls="none", mec="white", label="low-cost worker + strong checker"),
         Line2D([], [], color=C["alt"], marker="o", ls="none", mec="white", label="alt. low-cost worker"),
         Line2D([], [], color=C["ours"], marker="D", ls="none", mec="white", label="class-aware crew (router)")]
    fig.legend(handles=h, loc="outside lower center", ncol=4, fontsize=6.5, columnspacing=1.2, handletextpad=0.4)
    save(fig, "fig-crew-frontier.pdf")


def fig_headline():
    h = pd.read_csv(D("headline.csv"))
    c = pd.read_csv(D("deepswe_configs.csv")); c["release"] = pd.to_datetime(c["release"])
    a = cost_to_reach(frontier(c[c.release <= "2026-04-01"][["cost", "q"]].values), 0.5)
    b = cost_to_reach(frontier(c[c.release <= "2026-05-01"][["cost", "q"]].values), 0.5)
    drift = a / b
    rows = [
        (2, "sequential search vs best configuration\n(DeepSWE, verifier $r\\geq0.8$, equal success)", C["ours"], "range over $r$"),
        (1, "uniform low-cost vs uniform frontier crew\n(defect fixes, score difference n.s.)", C["ours_light"], "95% CI"),
        (3, "class-conditional single-shot router vs best configuration\n(DeepSWE, 95% of best success)", C["ours_light"], "range 90–98%"),
        (0, "class-aware crew vs best fixed crew\n(GitHub issues, score difference 0.00)", C["ours"], "95% CI"),
    ]
    fig, ax = plt.subplots(figsize=(TEXT_W, 2.15))
    y = np.arange(len(rows) + 1)[::-1].astype(float)
    for yi, (k, lab, col, note) in zip(y[1:], rows):
        r = h.iloc[k]
        ax.errorbar(r.factor, yi, xerr=[[r.factor - r.lo], [r.hi - r.factor]], fmt="o", color=col, ms=5, mec="white", mew=0.6,
                    elinewidth=1.2, capsize=2.5)
        ax.text(r.hi * 1.15, yi, f"{r.factor:.1f}$\\times$  [{r.lo:.1f}, {r.hi:.1f}]" if "CI" in note else f"{r.factor:.1f}$\\times$  ({r.lo:.1f}–{r.hi:.1f})",
                va="center", fontsize=7)
    ax.plot([drift], [y[0]], marker="D", ms=4.5, color=C["context"], mec="white", mew=0.5, ls="none")
    ax.text(drift * 1.15, y[0], f"{drift:.0f}$\\times$  (\\${a:.2f} $\\to$ \\${b:.2f})", va="center", fontsize=7, color=C["context"])
    ax.set_yticks(y)
    ax.set_yticklabels(["frontier shift in one month, for scale\n(DeepSWE, 50% success, Apr → May 2026)"] + [r[1] for r in rows], fontsize=6.6)
    ax.get_yticklabels()[0].set_color(C["context"])
    log_axis(ax, "x", [1, 2, 5, 10, 20, 50, 100], fmt="times"); ax.set_xlim(1, 150)
    refline(ax, 1, axis="x", color=C["diag_low"])
    ax.set_xlabel("cost-reduction factor at equal quality (log scale)")
    ax.set_ylim(y.min() - 0.6, y.max() + 0.6)
    ax.tick_params(axis="y", length=0)
    save(fig, "fig-headline.pdf")
    return drift, a, b


def fig_deepswe_front():
    fig, ax = plt.subplots(figsize=(TEXT_W, 2.55))
    sp = crv[(crv.r == 1.0) & (crv.policy == "single_points")]
    ax.scatter(sp.cost, sp.q, s=6, color=C["points"], lw=0, zorder=1, label="single configuration (89)")
    V = frontier(sp[["cost", "q"]].values)
    ax.plot(V[:, 0], V[:, 1], color=C["diag_low"], lw=1.0, ls=(0, (4, 2)), zorder=3.5, label="single-configuration hull")
    su = crv[(crv.r == 1.0) & (crv.policy == "u_single_emp")]
    Vs = frontier(su[["cost", "q"]].values)
    ax.plot(Vs[:, 0], Vs[:, 1], color=C["ours_light"], lw=1.2, zorder=3, label="class-conditional single-shot router")
    for r, a in [(1.0, 1.0), (0.9, 0.7), (0.8, 0.45)]:
        s = crv[(crv.r == r) & (crv.policy == "u_pidx_emp")]
        Vq = frontier(s[["cost", "q"]].values)
        ax.plot(Vq[:, 0], Vq[:, 1], color=C["ours"], alpha=a, lw=1.5, zorder=4, label=f"sequential search, $r={r:.1f}$")
    o = orc[orc.r == 1.0]
    ax.plot(o.cost, o.q, marker="*", ms=8, color=C["ref"], ls="none", zorder=5, label="per-task oracle (optimistic)")
    log_axis(ax, "x", [0.1, 0.3, 1, 3, 10])
    ax.set_xlim(0.085, 25); ax.set_ylim(0, 1.0)
    ax.set_xlabel("mean cost per task (log scale)"); ax.set_ylabel("success rate")
    fig.legend(loc="outside right center", fontsize=7)
    save(fig, "fig-deepswe-front.pdf")


def fig_aiq_vs_r():
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.45))
    ax = axes[0]
    d_est = -aiq[(aiq.r == 1.0) & (aiq.policy == "u_single_emp")].gain.iloc[0]
    rs = r_at(d_est)
    ax.axvspan(0.6, 0.7, color="#F0F0F0", lw=0, zorder=0)
    rr = np.linspace(0.58, 1.0, 101)
    ax.plot(rr, g_pop(rr), color=C["ref"], ls=(0, (1, 1.5)), lw=1.0, label="$G_{\\rm pop}(r)$: two-stage, known rates")
    ax.plot(rr, g_pop(rr) - d_est, color=C["ref"], ls=(0, (4, 2)), lw=0.9, label="$G_{\\rm pop}(r)-\\Delta_{\\rm est}$")
    for p, col, lab, mk in [("u_pidx_emp", C["ours"], "Pandora index", "o"), ("u_dp3_emp", C["ours_light"], "exact DP, 3 attempts", "s")]:
        s = aiq[aiq.policy == p].sort_values("r")
        ax.errorbar(s.r, s.gain, yerr=s.ci, color=col, marker=mk, ms=3.8, mec="white", mew=0.5, capsize=2, lw=1.3, elinewidth=0.8, label=lab)
    refline(ax, 0, color=C["diag_low"])
    refline(ax, rs, axis="x", color=C["verify"], ls=(0, (3, 1, 1, 1)))
    ax.text(rs + 0.005, -0.035, f"$\\hat r^\\ast={rs:.2f}$", color=C["verify"], fontsize=7, ha="left")
    ax.text(1.0, 0.004, "single-configuration hull", fontsize=6.3, ha="right", va="bottom", color=C["diag_low"])
    ax.set_xlabel("verifier accuracy $r$"); ax.set_ylabel("AIQ gain over hull")
    ax.set_xlim(0.585, 1.01); ax.set_ylim(-0.04, 0.2)
    ax.legend(loc="upper left", fontsize=6.4, bbox_to_anchor=(0.0, 0.93))
    title(ax, "a", "value of sequential search")
    ax = axes[1]
    bq, bc = aiq.best_single_q.iloc[0], aiq.best_single_cost.iloc[0]
    for p, col, lab, mk in [("u_pidx_emp", C["ours"], "Pandora index", "o"), ("u_dp3_emp", C["ours_light"], "exact DP, 3 attempts", "s")]:
        s = aiq[aiq.policy == p].sort_values("r")
        ax.plot(s.r, 100 * s.cost_to_best / bc, color=col, marker=mk, ms=3.8, mec="white", mew=0.5, lw=1.3, label=lab)
    refline(ax, 100, color=C["diag_low"])
    ax.text(0.595, 106, f"best single configuration (success {bq:.3f}, \\${bc:.2f} per task)", fontsize=6.3, color=C["diag_low"], va="bottom")
    log_axis(ax, "y", [3, 10, 30, 100], fmt="pct"); ax.set_ylim(2.5, 180)
    ax.set_xlabel("verifier accuracy $r$"); ax.set_ylabel("cost to match best configuration\n(% of its cost, log)")
    ax.set_xlim(0.585, 1.01)
    ax.legend(loc="upper right", fontsize=6.4, bbox_to_anchor=(1.0, 0.86))
    title(ax, "b", "cost at equal success")
    save(fig, "fig-aiq-vs-r.pdf")
    return d_est, rs


def fig_pacing():
    p = pd.read_csv(D("seq_pacing.csv"))
    p = p[p.eta == 0.2]
    g = p.groupby("r").agg(over=("cost_over_B", "mean"), q=("q", "mean"), hq=("hindsight_q", "mean"), n=("cost_over_B", "size")).reset_index()
    g["gap"] = g.hq - g.q
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 1.9))
    x = np.arange(len(g))
    axes[0].bar(x, g.over, color=C["ours"], width=0.62)
    refline(axes[0], 1, color=C["diag_low"])
    axes[0].set_ylabel("realised cost / budget"); axes[0].set_ylim(0, 1.55)
    axes[1].bar(x, g.gap, color=C["ours_light"], width=0.62)
    refline(axes[1], 0, color=C["diag_low"])
    axes[1].set_ylabel("quality gap to best\nfixed $\\lambda$ in hindsight"); axes[1].set_ylim(-0.02, 0.02)
    for k, ax in enumerate(axes):
        ax.set_xticks(x); ax.set_xticklabels([f"{v:g}" for v in g.r]); ax.set_xlabel("verifier accuracy $r$")
        title(ax, "ab"[k])
    save(fig, "fig-pacing.pdf")
    return g


def fig_anchors():
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    col = "within gain_vs_hull"
    K = [0, 25, 50, 100]
    series = [("irt_comp_substack_rand", "random anchors", C["ours"], "o", "-"),
              ("irt_comp_substack_cur", "curriculum anchors", C["ours_light"], "s", "-"),
              ("irt_comp_substack_vopt", "D-optimal anchors", C["verify_light"], "^", "-"),
              ("subset_id", "class means, no prior", C["diag_low"], "x", (0, (4, 2)))]
    for m, lab, c, mk, ls in series:
        y = []
        for k in K:
            mm = m if k else ("subset_id" if m == "subset_id" else "irt_comp_substack")
            r = lb[(lb.split == f"S3_K{k}") & (lb.method == mm)]
            y.append(num(r[col].iloc[0]) if len(r) else np.nan)
        ax.plot(K, y, marker=mk, color=c, ls=ls, ms=3.8, mec="white" if mk != "x" else c, mew=0.5, label=lab)
        if m == "irt_comp_substack_rand":
            fitdat = y
    refline(ax, 0, color=C["diag_low"], lw=0.5)
    ax.set_xlabel("anchor tasks $K$ run by the new model"); ax.set_ylabel("AIQ gain over hull")
    ax.set_xticks(K); ax.set_xlim(-4, 104); ax.set_ylim(-0.08, 0.03)
    ax.legend(loc="lower right", fontsize=6.4)
    save(fig, "fig-anchors.pdf")
    # saturating form g(K) = g0 + (ginf - g0) K / (K + K_half), fitted by least squares on the random-anchor series
    y = np.array(fitdat); g0 = y[0]; best = None
    for kh in np.linspace(0.5, 60, 600):
        for ginf in np.linspace(y.max() - 0.02, y.max() + 0.02, 81):
            e = ((g0 + (ginf - g0) * np.array(K) / (np.array(K) + kh) - y) ** 2).sum()
            if best is None or e < best[0]:
                best = (e, kh, ginf)
    return best[1], best[2], g0


def fig_unseen():
    rows = [("S5", "unseen model family (S5)"), ("S2", "unseen benchmark family (S2)"), ("S3_K0", "unseen models, no anchors (S3)")]
    fig, ax = plt.subplots(figsize=(COL_W, 1.55))
    y = np.arange(len(rows))[::-1]
    for yi, (sp, lab) in zip(y, rows):
        r = ss[(ss.split == sp) & (ss.method == "unified") & (ss.ref == "subset_id")].iloc[0]
        ax.errorbar(r["diff"], yi, xerr=[[r["diff"] - r.diff_lo], [r.diff_hi - r["diff"]]], fmt="o", color=C["ours"], ms=4, mec="white",
                    mew=0.5, capsize=2, elinewidth=1)
        ax.text(r.diff_hi + 0.006, yi, f"+{r['diff']:.3f}", va="center", fontsize=6.8)
    ax.set_yticks(y); ax.set_yticklabels([r[1] for r in rows], fontsize=6.8)
    refline(ax, 0, axis="x", color=C["diag_low"])
    ax.set_xlim(-0.01, 0.2); ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("AIQ gain over class-mean router")
    ax.tick_params(axis="y", length=0)
    save(fig, "fig-unseen.pdf")


def fig_class_suff():
    it = pd.read_csv(D("irt_summary.csv"))
    fams = [("S1", "subset_id", "class identity (reference)"),
            ("S1", "irt_comp_substack", "per-task IRT difficulty"),
            ("S1", "gbdt_feats", "boosted text features"),
            ("S1", "ct_hash_con", "contrastive tiny encoder"),
            ("S1", "knn_embed", "embedding nearest neighbours"),
            ("S1cv", "irt_comp_substack", "per-task IRT difficulty, CV"),
            ("S1cv", "knn_embed", "embedding nearest neighbours, CV")]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.9), gridspec_kw=dict(width_ratios=[1.1, 1]))
    ax = axes[0]
    yy = np.arange(len(fams))[::-1]
    for yi, (sp, m, lab) in zip(yy, fams):
        r = it[(it.split == sp) & (it.method == m)]
        v = str(r["gain vs subset_id"].iloc[0])
        mu = num(v); ci = float(v.split("±")[1]) if "±" in v else 0.0
        ref = m == "subset_id"
        ax.errorbar(mu, yi, xerr=ci if ci else None, fmt="s" if ref else "o", ms=3.8, capsize=2, elinewidth=0.9,
                    color=C["diag_low"] if ref else C["ours"], mec="white", mew=0.5)
        ax.text(0.045, yi, f"{mu:+.3f}" + (f" ± {ci:.3f}" if ci else ""), va="center", fontsize=6.4)
    ax.set_yticks(yy); ax.set_yticklabels([f[2] for f in fams], fontsize=6.8)
    refline(ax, 0, axis="x", color=C["diag_low"])
    ax.set_xlim(-0.16, 0.1); ax.set_ylim(-0.6, len(fams) - 0.4)
    ax.set_xlabel("AIQ gain over class identity")
    ax.tick_params(axis="y", length=0)
    title(ax, "a", "instance features vs class identity")
    ax = axes[1]
    h = pd.read_csv(D("headroom.csv")).sort_values("headroom", ascending=True)
    h = h[h.matrix != "__pooled__"]
    yy = np.arange(len(h))
    ax.barh(yy, h.headroom, color=C["points"], height=0.72, label="naive per-task oracle")
    ax.barh(yy, h.headroom_denoised, color=C["ours"], height=0.72, label="noise-aware estimate")
    ax.set_yticks(yy); ax.set_yticklabels([m.replace("_", " ") for m in h.matrix], fontsize=6.0)
    ax.set_xlabel("AIQ headroom over hull"); ax.set_xlim(0, 0.36); ax.set_ylim(-0.6, len(h) - 0.4)
    ax.legend(loc="lower right", fontsize=6.4)
    ax.tick_params(axis="y", length=0)
    title(ax, "b", "per-task headroom")
    save(fig, "fig-class-suff.pdf")
    return h


def fig_frontier_time():
    c = pd.read_csv(D("deepswe_configs.csv"))
    c["release"] = pd.to_datetime(c["release"])
    cuts = ["2026-04-01", "2026-05-01", "2026-08-01", "2026-09-30"]
    cmap = plt.get_cmap("cividis")
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.45), gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax = axes[0]
    ax.scatter(c.cost, c.q, s=5, color=C["points"], lw=0, zorder=1)
    for i, d in enumerate(cuts):
        x = c[c.release <= d]
        V = frontier(x[["cost", "q"]].values)
        Vx = np.r_[V, [[30, V[-1, 1]]]]
        ax.plot(Vx[:, 0], Vx[:, 1], color=cmap(0.1 + 0.8 * i / (len(cuts) - 1)), lw=1.4, label=f"released by {pd.Timestamp(d):%-d %b %Y}")
    log_axis(ax, "x", [0.1, 0.3, 1, 3, 10]); ax.set_xlim(0.085, 25); ax.set_ylim(0, 0.8)
    ax.set_xlabel("mean cost per task (log scale)"); ax.set_ylabel("success rate")
    ax.legend(loc="lower right", fontsize=6.4)
    title(ax, "a", "hull of available configurations")
    ax = axes[1]
    dates = pd.date_range("2026-03-15", "2026-09-30", freq="3D")
    levels = [0.5, 0.6, 0.65, 0.7]
    for i, t in enumerate(levels):
        yv = [cost_to_reach(frontier(c[c.release <= d][["cost", "q"]].values), t) if (c.release <= d).any() else np.nan for d in dates]
        col = cmap(0.1 + 0.8 * i / (len(levels) - 1))
        ax.plot(dates, yv, color=col, lw=1.4, drawstyle="steps-post")
        last = [v for v in yv if np.isfinite(v)][-1]
        ax.text(dates[-1] + pd.Timedelta(days=3), last, f"{t:.0%}", fontsize=6.6, color=col, va="center")
    log_axis(ax, "y", [0.1, 0.3, 1, 3, 10])
    ax.set_ylim(0.07, 15)
    ax.set_ylabel("lowest cost per task to reach\nsuccess level (log)")
    ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_xlim(dates[0], dates[-1] + pd.Timedelta(days=14))
    ax.set_xlabel("2026")
    title(ax, "b", "cost of a fixed success level")
    save(fig, "fig-frontier-time.pdf")
    out = {}
    for d in ["2026-04-01", "2026-05-01", "2026-07-01", "2026-08-01", "2026-09-30"]:
        x = c[c.release <= d]; V = frontier(x[["cost", "q"]].values)
        out[d] = {t: cost_to_reach(V, t) for t in (0.5, 0.55, 0.6, 0.65, 0.7)}
    return out


def fig_trial_class():
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.3), sharey=True, gridspec_kw=dict(width_ratios=[5, 3]))
    short = {"B": "uniform\nlow-cost", "F": "alt. low-cost\nworker", "E": "+ low-cost\nchecker", "C": "+ strong\nchecker", "D": "uniform\nfrontier"}
    for k, (ax, cl) in enumerate(zip(axes, ["bug-fix", "open-ended"])):
        s = ca[ca.cls == cl].copy()
        order = [a for a in ["B", "D", "F", "E", "C"] if a in set(s.arm)]
        s = s.set_index("arm").loc[order].reset_index()
        x = np.arange(len(s))
        ax.bar(x, s.score, color=[ARM[a][1] for a in s.arm], width=0.64)
        ax.errorbar(x, s.score, yerr=[s.score - s.lo, s.hi - s.score], fmt="none", ecolor="black", capsize=2, elinewidth=0.8)
        for xi, r in zip(x, s.itertuples()):
            ax.text(xi, r.hi + 0.25, f"{r.mergeable}/{r.n}\n\\${r.cost:.3f}", ha="center", va="bottom", fontsize=6.2, linespacing=1.1)
        ax.set_xticks(x); ax.set_xticklabels([short[a] for a in s.arm], fontsize=6.6)
        ax.tick_params(axis="x", length=0)
        title(ax, "ab"[k], f"{'defect fixes' if cl == 'bug-fix' else 'open-ended tasks'} ($n={int(s.n.iloc[0])}$)")
    axes[0].set_ylabel("review score (0–10)"); axes[0].set_ylim(0, 10.5)
    axes[0].yaxis.set_major_locator(MultipleLocator(2))
    save(fig, "fig-trial-class.pdf")


def fig_seat():
    fig, ax = plt.subplots(figsize=(COL_W, 2.1))
    items = [("bug-fix", "E", "B", "defect: + low-cost checker", C["verify_light"]),
             ("bug-fix", "C", "B", "defect: + strong checker", C["verify"]),
             ("bug-fix", "D", "B", "defect: frontier in all roles", C["diag_front"]),
             ("open-ended", "E", "B", "open-ended: + low-cost checker", C["verify_light"]),
             ("open-ended", "C", "B", "open-ended: + strong checker", C["verify"])]
    y = np.arange(len(items))[::-1]
    for yi, (cl, a, b, lab, col) in zip(y, items):
        r = con[(con.cls == cl) & (con.a == a) & (con.b == b)].iloc[0]
        ax.errorbar(r["diff"], yi, xerr=[[r["diff"] - r.lo], [r.hi - r["diff"]]], fmt="o", color=col, capsize=2, ms=4, mec="white",
                    mew=0.5, elinewidth=1)
        ax.text(5.6, yi, f"{r.cost_a / r.cost_b:.1f}$\\times$", va="center", fontsize=6.6)
    ax.text(5.6, len(items) - 0.35, "cost", fontsize=6.3, va="center", color=C["diag_low"])
    ax.set_yticks(y); ax.set_yticklabels([i[3] for i in items], fontsize=6.6)
    refline(ax, 0, axis="x", color=C["diag_low"])
    ax.axhline(1.5, color=C["points"], lw=0.6)
    ax.set_xlim(-2, 6.6); ax.set_ylim(-0.6, len(items) - 0.1)
    ax.set_xlabel("score change vs uniform low-cost crew")
    ax.tick_params(axis="y", length=0)
    save(fig, "fig-seat.pdf")


def fig_heatmap():
    pi = pd.read_csv(D("trial_per_issue.csv"))
    arms = ["B", "D", "F", "E", "C"]
    pi = pi.sort_values(["cls", "task"])
    lab = []; nb = no = 0
    for r in pi.itertuples():
        if r.cls == "bug-fix":
            nb += 1; lab.append(f"D{nb}")
        else:
            no += 1; lab.append(f"O{no}")
    M = pi[[f"{a}_score" for a in arms]].values.T
    fig, ax = plt.subplots(figsize=(TEXT_W, 1.75))
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=10)
    names = {"B": "uniform low-cost", "D": "uniform frontier", "F": "alt. low-cost worker", "E": "+ low-cost checker", "C": "+ strong checker"}
    ax.set_yticks(range(len(arms))); ax.set_yticklabels([names[a] for a in arms], fontsize=6.6)
    ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab, fontsize=6.2)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:g}", ha="center", va="center", fontsize=5.6, color="white" if M[i, j] < 6 else "black")
    for i in range(M.shape[0]):
        if np.isnan(M[i, nb:]).all():
            ax.text((nb + M.shape[1] - 1) / 2, i, "not run on open-ended tasks", ha="center", va="center", fontsize=6.2, color=C["context"])
    ax.axvline(nb - 0.5, color="white", lw=2)
    ax.tick_params(length=0)
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01); cb.ax.tick_params(labelsize=6.2)
    cb.set_label("review score (0–10)", fontsize=6.6); cb.outline.set_linewidth(0.5)
    save(fig, "fig-heatmap.pdf")
    return dict(zip(lab, pi.task))


def fig_s1_matrix():
    s = pd.read_csv(D("s1_per_matrix.csv")).sort_values("n")
    fig, ax = plt.subplots(figsize=(TEXT_W, 2.7))
    y = np.arange(len(s))
    ax.errorbar(s["diff"], y, xerr=[s["diff"] - s.lo, s.hi - s["diff"]], fmt="o", color=C["ours"], ms=3.6, mec="white", mew=0.5,
                capsize=2, elinewidth=0.9)
    ax.set_yticks(y); ax.set_yticklabels([f"{m.replace('_', ' ')} ($n={n}$)" for m, n in zip(s.matrix, s.n)], fontsize=6.4)
    refline(ax, 0, axis="x", color=C["diag_low"])
    ax.set_xlabel("AIQ difference, router minus class means (held-out tasks)")
    ax.set_ylim(-0.6, len(s) - 0.4); ax.tick_params(axis="y", length=0)
    save(fig, "fig-s1-matrix.pdf")


def fig_theory_pairs():
    p = pd.read_csv(D("theory_pairs.csv")).dropna(subset=["rstar"])
    p = p[(p.rstar > 0) & (p.rstar < 1)].copy()
    p["gap"] = p.p2 - p.p1
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    sc = ax.scatter(p.rho, p.rstar, c=p.gap, cmap="cividis", vmin=0, vmax=0.75, s=22, edgecolor="black", linewidth=0.4, zorder=3)
    refline(ax, 0.5, color=C["diag_low"])
    ax.text(0.011, 0.51, "$r^\\ast=1/2$", fontsize=6.4, va="bottom", color=C["diag_low"])
    log_axis(ax, "x", [0.01, 0.1, 1], fmt="plain"); ax.set_xlim(0.01, 1)
    ax.set_ylim(0.45, 0.8)
    ax.set_xlabel("cost ratio $\\rho=c_1/c_2$ of the pair (log)"); ax.set_ylabel("threshold $r^\\ast$")
    cb = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02); cb.ax.tick_params(labelsize=6.2)
    cb.set_label("quality gap $p_2-p_1$", fontsize=6.6); cb.outline.set_linewidth(0.5)
    save(fig, "fig-theory-pairs.pdf")
    return p


# ------------------------------------------------------------------ tables
def tex_money(x):
    return f"\\${x:.3f}"


def tab_trial():
    rows = []
    order = ["routed (class-aware)", "strong-checker crew", "low-cost crew + low-cost checker", "uniform low-cost crew"]
    p = pol.set_index("policy")
    for k in order:
        r = p.loc[k]
        rows.append(f"{k} & {r.score:.2f} [{r.lo:.2f}, {r.hi:.2f}] & {int(r.mergeable)}/22 & {int(r.good)}/22 & {tex_money(r.cost)} \\\\")
    s = "\\begin{tabular}{lcccc}\\toprule\npolicy & score [95\\% CI] & mergeable & good ($\\geq 7$) & cost / issue\\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n"
    open(OUT("tab-trial.tex"), "w").write("% generated by figures.py; do not edit\n" + s)


def tab_seq(d_est, rs):
    a = aiq.set_index(["policy", "r"])
    rsel = [1.0, 0.9, 0.8, 0.7, 0.6]
    lines = []
    for pol_, lab in [("u_pidx_emp", "Pandora index"), ("u_dp3_emp", "exact DP ($A{=}3$)"), ("u_single_emp", "learned single-shot")]:
        cells = []
        for r in rsel:
            x = a.loc[(pol_, r)]
            cells.append(f"${x.gain:+.3f}\\pm{x.ci:.3f}$")
        lines.append(lab + " & " + " & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    for pol_, lab in [("u_pidx_emp", "Pandora index"), ("u_dp3_emp", "exact DP ($A{=}3$)")]:
        cells = []
        for r in rsel:
            x = a.loc[(pol_, r)]
            cells.append("--" if not np.isfinite(x.cost_to_best) else f"\\${x.cost_to_best:.2f} ({100 * x.cost_to_best / x.best_single_cost:.0f}\\%)")
        lines.append(lab + " & " + " & ".join(cells) + " \\\\")
    s = ("\\begin{tabular}{l" + "c" * len(rsel) + "}\\toprule\n & " + " & ".join(f"$r={r:g}$" for r in rsel) + "\\\\\\midrule\n"
         + "\\multicolumn{" + str(len(rsel) + 1) + "}{l}{\\emph{AIQ gain over the single-configuration hull (mean $\\pm$ 95\\% $t$-CI, 15 folds)}}\\\\\n"
         + "\n".join(lines[:3]) + "\n\\midrule\n\\multicolumn{" + str(len(rsel) + 1) + "}{l}{\\emph{cost per task to reach the best configuration's success (share of its cost)}}\\\\\n"
         + "\n".join(lines[4:]) + "\n\\bottomrule\\end{tabular}\n")
    open(OUT("tab-seq.tex"), "w").write("% generated by figures.py; do not edit\n" + s)


def tab_models():
    m = pd.read_csv(D("deepswe_models.csv")).sort_values("cost")
    rows = []
    for r in m.itertuples():
        pin = "--" if not np.isfinite(r.price_in) else f"{r.price_in:.2f}"
        pout = "--" if not np.isfinite(r.price_out) else f"{r.price_out:.2f}"
        rel = str(r.release)[:10] if isinstance(r.release, str) else "--"
        rows.append(f"\\texttt{{{r.model_canonical}}} & {rel} & {pin} & {pout} & {int(r.trials)} & {r.q:.3f} & {r.cost:.2f}\\\\")
    s = ("\\begin{longtable}{lcrrrrr}\\caption{Base models in the DeepSWE trials: catalog price per million input and output tokens, release date, trials, mean success and mean cost per trial (pooled over benchmark versions).}\\label{tab:allmodels}\\\\\\toprule\nmodel & released & \\$/M in & \\$/M out & trials & success & \\$/task\\\\\\midrule\\endfirsthead\n\\toprule\nmodel & released & \\$/M in & \\$/M out & trials & success & \\$/task\\\\\\midrule\\endhead\n"
         + "\n".join(rows) + "\n\\bottomrule\\end{longtable}\n")
    open(OUT("tab-models.tex"), "w").write("% generated by figures.py; do not edit\n" + s)


def tab_per_issue(labmap):
    pi = pd.read_csv(D("trial_per_issue.csv"))
    inv = {v: k for k, v in labmap.items()}
    pi["lab"] = pi.task.map(inv)
    pi["k"] = pi.lab.str[0] + pi.lab.str[1:].str.zfill(2)
    pi = pi.sort_values("k")
    arms = ["B", "F", "E", "C", "D"]
    rows = []
    for r in pi.itertuples():
        cells = []
        for a in arms:
            s = getattr(r, f"{a}_score"); c = getattr(r, f"{a}_cost"); mg = getattr(r, f"{a}_merge")
            cells.append("--" if not np.isfinite(s) else f"{s:g}{'' if mg in (True, 'True') else '$^\\times$'} / {c:.3f}")
        rows.append(f"{r.lab} & \\texttt{{{r.task.replace('_', chr(92) + '_')}}} & " + " & ".join(cells) + "\\\\")
    s = ("\\begin{longtable}{llccccc}\\caption{Per-issue results: mean review score / cost in dollars per configuration; $^\\times$ marks a candidate that at least one reviewer rejected; -- = not run.}\\label{tab:perissue}\\\\\\toprule\nid & repository--issue & uniform low-cost & alt.\\ worker & + low-cost chk & + strong chk & uniform frontier\\\\\\midrule\\endfirsthead\n\\toprule\nid & repository--issue & uniform low-cost & alt.\\ worker & + low-cost chk & + strong chk & uniform frontier\\\\\\midrule\\endhead\n"
         + "\n".join(rows) + "\n\\bottomrule\\end{longtable}\n")
    open(OUT("tab-per-issue.tex"), "w").write("% generated by figures.py; do not edit\n" + s)


def tab_leverage():
    rows = [("open-ended", "C", "B", "checker: low-cost $\\to$ frontier"), ("open-ended", "E", "B", "checker: low-cost $\\to$ second low-cost"),
            ("defect fix", "C", "B", "checker: low-cost $\\to$ frontier"), ("defect fix", "E", "B", "checker: low-cost $\\to$ second low-cost"),
            ("defect fix", "D", "B", "all roles: low-cost $\\to$ frontier")]
    out = []
    for cl, a, b, lab in rows:
        r = con[(con.cls == ("bug-fix" if cl == "defect fix" else cl)) & (con.a == a) & (con.b == b)].iloc[0]
        dc = r.cost_a - r.cost_b
        lev = "dominant ($\\Delta c_s\\le0$)" if dc <= 0 else f"{r['diff'] / dc:.1f}"
        out.append(f"{cl} & {lab} & {r['diff']:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}] & {dc:+.3f} & {lev}\\\\")
    s = ("\\begin{tabular}{lllrl}\\toprule\nclass & upgrade & $\\Delta_s$ & $\\Delta c_s$ (\\$) & $\\Lambda_s$ (points/\\$)\\\\\\midrule\n" + "\n".join(out) + "\n\\bottomrule\\end{tabular}\n")
    s = re.sub(r"(?<=[\[ ])-(\d)", r"$-$\1", s)
    open(OUT("tab-leverage.tex"), "w").write("% generated by figures.py; do not edit\n" + s)


def tab_summary(drift):
    h = pd.read_csv(D("headline.csv")); ss_ = ss
    cd = con[(con.cls == "all") & (con.b == "uniform low-cost crew")]
    P = pol.set_index("policy")
    dcost = P.loc["routed (class-aware)"].cost - P.loc["uniform low-cost crew"].cost
    s5 = ss_[(ss_.split == "S5") & (ss_.method == "unified") & (ss_.ref == "subset_id")].iloc[0]
    cf = con[(con.cls == "all") & (con.b == "strong-checker crew")].iloc[0]
    rows = [
        ("GitHub issues (22)", "class-aware crew vs best fixed crew", f"{h.iloc[0].factor:.2f}$\\times$ [{h.iloc[0].lo:.2f}, {h.iloc[0].hi:.2f}] lower cost", f"score $\\Delta={abs(cf['diff']):.2f}$ [{cf.lo:+.2f}, {cf.hi:+.2f}]"),
        ("GitHub issues (22)", "class-aware crew vs best diagonal policy", f"mergeable 12 $\\to$ 20, +\\${dcost:.3f}", f"score $\\Delta={cd.iloc[0]['diff']:+.2f}$ [{cd.iloc[0].lo:+.2f}, {cd.iloc[0].hi:+.2f}]"),
        ("Open-ended issues (8)", "off-diagonal: checker upgrade only", "mergeable 0\\% $\\to$ 100\\%", "score $\\Delta=+4.00$ [+3.19, +4.94]"),
        ("Defect fixes (14)", "diagonal: uniform low-cost vs uniform frontier", f"{h.iloc[1].factor:.1f}$\\times$ [{h.iloc[1].lo:.1f}, {h.iloc[1].hi:.1f}] lower cost", f"score $\\Delta={h.iloc[1].dq:+.2f}$ [{h.iloc[1].dq_lo:+.2f}, {h.iloc[1].dq_hi:+.2f}]"),
        ("DeepSWE (111 tasks)", "sequential search, $r\\geq0.8$", f"{h.iloc[2].lo:.0f}--{h.iloc[2].hi:.0f}$\\times$ lower cost", "equal success (0.745)"),
        ("DeepSWE (111 tasks)", "single-shot class router", f"{h.iloc[3].factor:.1f}$\\times$ lower cost", "95\\% of best success"),
        ("Unseen model families", "credibility priors vs class means", f"closes {100 * s5['diff'] / (-s5.ref_gain_hull):.0f}\\% of AIQ gap", f"$+{s5['diff']:.3f}$ AIQ [{s5.diff_lo:.3f}, {s5.diff_hi:.3f}]"),
        ("DeepSWE frontier", "Apr $\\to$ May 2026, 50\\% success", f"{drift:.0f}$\\times$ lower cost", "context"),
    ]
    s = ("\\begin{tabularx}{\\textwidth}{l>{\\raggedright\\arraybackslash}Xll}\\toprule\nsetting & comparison & effect & quality\\\\\\midrule\n"
         + "\n".join(" & ".join(r[:2] + tuple(re.sub(r"(?<=[\[ ])-(\d)", r"$-$\1", x) for x in r[2:])) + "\\\\" for r in rows)
         + "\n\\bottomrule\\end{tabularx}\n")
    open(OUT("tab-summary.tex"), "w").write("% generated by figures.py; do not edit\n" + s)


def interaction_estimates():
    """Partial interaction estimates available without a factorial design.

    (1) Real-issue trial: class x verifier interaction, the checker upgrade's paired gain on open-ended tasks minus
        its paired gain on defect fixes (independent task bootstrap within each class).
    (2) DeepSWE: correlated failures between hull configurations (phi coefficient, covariance) and the variance
        shares of the fitted latent model (shared difficulty tau, base-model x task omega_b, configuration x task omega).
    """
    pi = pd.read_csv(D("trial_per_issue.csv"))
    rng = np.random.default_rng(0)
    g = {cl: (pi[pi.cls == cl].C_score - pi[pi.cls == cl].B_score).values for cl in ("open-ended", "bug-fix")}
    est = g["open-ended"].mean() - g["bug-fix"].mean()
    bs = [rng.choice(g["open-ended"], len(g["open-ended"])).mean() - rng.choice(g["bug-fix"], len(g["bug-fix"])).mean()
          for _ in range(10000)]
    err = {cl: float(1 - (pi[pi.cls == cl].B_merge.astype(str) == "True").mean()) for cl in ("open-ended", "bug-fix")}
    tp = pd.read_csv(D("theory_pairs.csv"))
    tp = tp[tp.p1 > 0.1].copy()                               # drop pairs whose cheap member almost never succeeds
    p11 = tp.p1 - tp.p10
    tp["cov"] = p11 - tp.p1 * tp.p2
    tp["phi"] = tp["cov"] / np.sqrt(tp.p1 * (1 - tp.p1) * tp.p2 * (1 - tp.p2))
    tp["two_obs"] = tp.p1 + tp.p01
    tp["two_ind"] = 1 - (1 - tp.p1) * (1 - tp.p2)
    f = pd.read_csv(D("seq_fit.csv"))
    tau, om, omb = f.tau.mean(), f.omega.mean(), f.omega_b.mean()
    tot = tau ** 2 + om ** 2 + omb ** 2
    rows = []
    for r in tp.itertuples():
        rows.append(f"{r.p1:.3f} & {r.p2:.3f} & {r.phi:.2f} & {r.two_ind:.3f} & {r.two_obs:.3f} & {r.two_ind - r.two_obs:.3f}\\\\")
    s = ("\\begin{tabular}{rrrrrr}\\toprule\n$p_1$ & $p_2$ & $\\mathrm{Corr}(y_1,y_2)$ & two attempts, independent & two attempts, observed & loss\\\\\\midrule\n"
         + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")
    open(OUT("tab-corr.tex"), "w").write("% generated by figures.py; do not edit\n" + s)
    print("wrote tab-corr.tex")
    return dict(class_x_checker=float(est), class_x_checker_lo=float(np.percentile(bs, 2.5)), class_x_checker_hi=float(np.percentile(bs, 97.5)),
                gain_open=float(g["open-ended"].mean()), gain_bug=float(g["bug-fix"].mean()), err_open=err["open-ended"], err_bug=err["bug-fix"],
                phi_min=float(tp.phi.min()), phi_max=float(tp.phi.max()), phi_median=float(tp.phi.median()),
                loss_min=float((tp.two_ind - tp.two_obs).min()), loss_max=float((tp.two_ind - tp.two_obs).max()),
                tau=float(tau), omega=float(om), omega_b=float(omb), share_shared=float((tau ** 2) / tot),
                share_base=float((tau ** 2 + omb ** 2) / tot))


if __name__ == "__main__":
    fig_deepswe_front()
    d_est, rs = fig_aiq_vs_r()
    g = fig_pacing()
    kh, ginf, g0 = fig_anchors()
    fig_unseen()
    h = fig_class_suff()
    ft = fig_frontier_time()
    fig_crew_frontier()
    fig_trial_class()
    fig_seat()
    labmap = fig_heatmap()
    fig_s1_matrix()
    tp = fig_theory_pairs()
    tab_trial(); tab_seq(d_est, rs); tab_models(); tab_per_issue(labmap)
    drift, _, _ = fig_headline(); tab_leverage(); tab_summary(drift)
    ie = interaction_estimates()
    summary = dict(interactions=ie, delta_est=d_est, rstar_hat=rs, anchors_Khalf=kh, anchors_ginf=ginf, anchors_g0=g0,
                   pacing=g.to_dict("records"), frontier_time={k: {str(t): v for t, v in d.items()} for k, d in ft.items()},
                   theory_pairs_rstar=sorted(tp.rstar.round(3).tolist()), gpop_07=float(g_pop(0.7)), gpop_06=float(g_pop(0.6)))
    json.dump(summary, open(D("figures_summary.json"), "w"), indent=1, default=float)
    print(json.dumps(summary, indent=1, default=float)[:3000])
