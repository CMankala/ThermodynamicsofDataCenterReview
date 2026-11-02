import sys; sys.path.insert(0, "/home/claude/rser")
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from figdata import *

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5,
                     "axes.labelsize": 8, "legend.fontsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e3e3e3",
                     "grid.linewidth": 0.5})
COL = {"M": "#0072B2", "S": "#D55E00", "C": "#009E73"}
MK = {"M": "o", "S": "s", "C": "^"}
LAB = {"M": "Measured (field or laboratory)", "S": "Simulated / modelled", "C": "Compiled from secondary sources"}
OUT = "/home/claude/rser/submission/"

def save(fig, n):
    fig.savefig(f"{OUT}Figure_{n}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}Figure_{n}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

def legend(ax, loc="lower right"):
    h = [Line2D([0], [0], marker=MK[k], color=COL[k], lw=0, ms=5, label=LAB[k]) for k in "MSC"]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)

def range_plot(data, xlabel, n, log=True, n_override=None, xlim=None, order=None, transform=None, figsize=(7.2, 4.0), tick_fmt=None, legend_loc="lower right"):
    cats = order or list(dict.fromkeys(d[0] for d in data))
    fig, ax = plt.subplots(figsize=figsize)
    rng = np.random.default_rng(3)
    counts = {c: 0 for c in cats}
    ncat = {c: sum(1 for d in data if d[0] == c) for c in cats}
    for cat, lo, hi, ref, cls, lab in data:
        i = cats.index(cat)
        k = counts[cat]; counts[cat] += 1
        y = i + (k - (ncat[cat] - 1) / 2) * min(0.11, 0.8 / max(ncat[cat] - 1, 1)) if ncat[cat] > 1 else i
        f = transform or (lambda v: v)
        a, b = f(lo), f(hi)
        if a != b:
            ax.plot([a, b], [y, y], color=COL[cls], lw=2, solid_capstyle="round", alpha=0.9)
        ax.plot([a], [y], marker=MK[cls], color=COL[cls], ms=4.5, lw=0, zorder=3)
        if a != b:
            ax.plot([b], [y], marker=MK[cls], color=COL[cls], ms=4.5, lw=0, zorder=3)
        ax.annotate(f"[{ref}]", (b, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=6, color="#444444")
    ax.set_yticks(range(len(cats))); ax.set_yticklabels(cats)
    for i in range(len(cats) - 1): ax.axhline(i + 0.5, color="#dddddd", lw=0.5)
    ax.invert_yaxis()
    if log: ax.set_xscale("log")
    if xlim: ax.set_xlim(*xlim)
    if tick_fmt: tick_fmt(ax)
    ax.set_xlabel(xlabel); ax.grid(axis="y", visible=False)
    legend(ax, legend_loc)
    fig.tight_layout(); save(fig, n_override or n)

# ---- Fig. 2: corpus composition (two separate panels are allowed to be compared, but keep one figure with two subplots of the same kind of data)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1.1, 1]})
labels, vals = zip(*STUDY_TYPES)
a1.barh(labels, vals, color="#0072B2", height=0.6); a1.invert_yaxis()
for i, v in enumerate(vals): a1.text(v + 0.5, i, str(v), va="center", fontsize=7)
a1.set_xlabel("Included studies (n = 146)"); a1.grid(axis="y", visible=False); a1.set_title("(a) Evidence type", loc="left")
yrs, cnt = zip(*YEARS)
a2.bar([str(y) for y in yrs], cnt, color="#0072B2", width=0.65)
a2.set_yscale("log"); a2.set_ylim(0.7, 200); a2.set_ylabel("Studies (log scale)")
for i, v in enumerate(cnt): a2.text(i, v * 1.15, str(v), ha="center", fontsize=6.5)
a2.tick_params(axis="x", rotation=60); a2.grid(axis="x", visible=False); a2.set_title("(b) Publication year", loc="left")
fig.tight_layout(); save(fig, 2)

# ---- Fig. 3: Carnot-limited COP vs temperature lift (log y)
fig, ax = plt.subplots(figsize=(4.6, 3.4))
lift = np.linspace(2, 60, 300); TL = 293.15
carnot = TL / lift
ax.plot(lift, carnot, color="#000000", lw=1.6, label="Reversed Carnot, $T_L$ = 20 °C")
for eta, ls in [(0.7, "--"), (0.5, ":")]:
    ax.plot(lift, eta * carnot, color="#0072B2", lw=1.4, ls=ls, label=f"$\\eta_{{II}}$ = {eta}")
ax.fill_between(lift, 0.68 * carnot, 0.74 * carnot, color="#0072B2", alpha=0.18, label="$\\eta_{II}$ 0.68–0.74, series chillers [18]")
ax.set_yscale("log"); ax.set_ylim(1, 200); ax.set_xlim(0, 60)
ax.set_xlabel("Temperature lift $T_H - T_L$ (K)"); ax.set_ylabel("COP (log scale)")
ax.axvspan(0, 12, color="#009E73", alpha=0.10); ax.text(6, 1.3, "per-stage lift\nof series\nchillers [18]", ha="center", fontsize=6.5, color="#006b4f")
ax.legend(loc="upper right", frameon=True, edgecolor="#cccccc"); ax.grid(axis="x", visible=False)
fig.tight_layout(); save(fig, 3)

# ---- Fig. 4: PUE overhead (PUE - 1) on log axis
def pue_ticks(ax):
    ticks = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2]
    ax.set_xticks(ticks); ax.set_xticklabels([f"{1 + t:.2f}".rstrip("0").rstrip(".") if t < 0.1 else f"{1 + t:.1f}" for t in ticks])
    ax.minorticks_off()
range_plot(PUE, "PUE (axis is PUE − 1, logarithmic)", 6, xlim=(0.008, 2.6), transform=lambda v: v - 1, tick_fmt=pue_ticks, figsize=(7.2, 6.2), n_override=6)

# ---- Fig. 5: heat flux
range_plot(HEATFLUX, "Heat flux (W cm$^{-2}$, log scale)", 4, xlim=(1, 6000), figsize=(7.2, 4.6))

# ---- Fig. 6: thermal resistance
range_plot(RTH, "Absolute thermal resistance (K W$^{-1}$, log scale)", 5, xlim=(0.004, 40), figsize=(7.2, 3.6))

# ---- Fig. 8: COP ladder
range_plot(COP, "Coefficient of performance (log scale)", 8, xlim=(0.2, 30000), figsize=(7.2, 4.4))

# ---- Fig. 9: control savings
fig, ax = plt.subplots(figsize=(7.2, 3.0))
labs = [f"{l}\n(vs {b}) [{r}]" for l, lo, hi, r, c, b in CONTROL]
for i, (l, lo, hi, r, c, b) in enumerate(CONTROL):
    ax.barh(i, hi, color=COL[c], height=0.55, alpha=0.35 if hi != lo else 0.9)
    ax.barh(i, lo, color=COL[c], height=0.55)
    ax.text(hi + 0.4, i, f"{lo:g}%" if lo == hi else f"{lo:g}–{hi:g}%", va="center", fontsize=7)
ax.set_yticks(range(len(CONTROL))); ax.set_yticklabels(labs, fontsize=6.8); ax.invert_yaxis()
ax.set_xlabel("Reported cooling energy or cost saving against the stated baseline (%)"); ax.set_xlim(0, 36)
ax.grid(axis="y", visible=False)
h = [Line2D([0], [0], marker="s", color=COL[k], lw=0, ms=6, label=LAB[k]) for k in "SC"]
ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
fig.tight_layout(); save(fig, 9)

# ---- Fig. 7: waste-heat grade ladder (Carnot factor vs source temperature)
fig, ax = plt.subplots(figsize=(4.8, 3.3))
T = np.linspace(25, 100, 400); T0 = 293.15
ax.plot(T, 1 - T0 / (T + 273.15), color="black", lw=1.6, label="Carnot factor $1 - T_0/T$ ($T_0$ = 20 °C)")
ax.plot(T, 1 - 287.15 / (T + 273.15), color="black", lw=1.0, ls="--", label="$T_0$ = 14 °C sink [26]")
bands = [(36, 45, "#009E73", "Direct heating\n36–45 °C [84]"), (55, 65, "#D55E00", "Sorption cooling\n55–65 °C\nCOP 0.3–0.6 [90–92]"), (75, 100, "#0072B2", "Power cycles\n≥75 °C [26,27]")]
for lo, hi, c, t in bands:
    ax.axvspan(lo, hi, color=c, alpha=0.13); ax.text((lo + hi) / 2, 0.235, t, ha="center", va="top", fontsize=6.5, color=c)
ax.plot([75], [0.1103], marker="*", color="#0072B2", ms=9, lw=0, label="ORC gross efficiency 11.03% at 75 °C [26]")
ax.set_xlabel("Heat source temperature (°C)"); ax.set_ylabel("Fraction of heat convertible to work")
ax.set_ylim(0, 0.24); ax.set_xlim(25, 100); ax.grid(axis="x", visible=False)
ax.legend(loc="lower right", fontsize=6.3, edgecolor="#cccccc")
fig.tight_layout(); save(fig, 7)
print("ok")
