"""Slide 22 - Quando una potenza è in minoranza, i membri la seguono?

Per Cina, Russia e USA: votazioni in cui la potenza vota diversamente dalla maggioranza.
Per ogni Stato (presente in almeno il 75% delle votazioni):
  osservato = quota di queste votazioni in cui vota come la potenza
  atteso    = accordo per caso, dalle frequenze di sì / astenuto / no dello Stato e della potenza
Box plot (con notch) di osservato - atteso, membri contro non membri.
Test: binomiale a una coda per Stato con correzione di Benjamini-Hochberg (pallini bordati);
Wilcoxon sui membri contro lo zero.
Output: figure/slide22_follow.png, figure/s22.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import binomtest, false_discovery_control, wilcoxon

import base

df = base.carica()
pres = base.presenza(df)
tenuti = pres[pres >= 0.75].index
d = base.voti_espressi(df)
maj = base.maggioranza(d)

righe = []
for P in ["CHINA", "RUSSIAN FEDERATION", "UNITED STATES"]:
    vp, minoranza = base.in_minoranza(d, maj, P)
    sub = d[d.rcid.isin(minoranza) & (d.Countryname != P) & d.Countryname.isin(tenuti)].copy()
    sub["pv"] = vp.reindex(sub.rcid).values
    for c, g in sub.groupby("Countryname"):
        oss = (g.vote == g.pv).mean()
        fc = g.vote.value_counts(normalize=True)
        fp = g.pv.value_counts(normalize=True)
        att = sum(fc.get(v, 0) * fp.get(v, 0) for v in [1, 2, 3])
        righe.append((P, c, c in base.NAM, oss * 100, att * 100, len(g)))
    print(P, "votazioni in minoranza:", len(minoranza))

r = pd.DataFrame(righe, columns=["P", "c", "mem", "obs", "exp", "n"])
r["diff"] = r.obs - r.exp
r["p"] = [binomtest(int(round(o / 100 * n)), n, e / 100, alternative="greater").pvalue
          for o, e, n in zip(r.obs, r.exp, r.n)]
r["sig"] = False
for P, g in r.groupby("P"):
    r.loc[g.index, "sig"] = false_discovery_control(g.p.values, method="bh") < 0.05
print(r.groupby(["P", "mem"])[["obs", "exp", "diff"]].median().round(1))
membri = r[r.mem]
print("membri sotto il proprio caso:",
      membri.groupby("P").apply(lambda g: (g["diff"] < 0).mean()).round(2).to_dict())
print("Wilcoxon (membri):", membri.groupby("P")["diff"].apply(lambda x: wilcoxon(x).pvalue).to_dict())
print("membri sopra il caso dopo BH:", membri[membri.sig].groupby("P").size().to_dict())
r.to_csv(base.OUT / "s22.csv", index=False)

# figura
plt = base.stile()
BG, NAVY = base.SFONDO, base.NAVY
rng = np.random.default_rng(3)
fig, ax = base.nuova_figura(plt, 10, 6.3)
pos = {}
for i, (P, nome) in enumerate([("CHINA", "China"), ("RUSSIAN FEDERATION", "Russia"),
                               ("UNITED STATES", "US")]):
    x0 = i * 1.3
    for mem, dx, c, cb, lab in [(True, -0.22, base.CORALLO, base.CORALLO_TESTO, "members"),
                                (False, 0.22, base.GRIGIO, base.GRIGIO_TESTO, "others")]:
        g = r[(r.P == P) & (r.mem == mem)].set_index("c")
        s = g["diff"]
        sig = g["sig"].values.astype(bool)
        x = x0 + dx + rng.uniform(-.12, .12, len(s))
        for k, v, xx in zip(s.index, s.values, x):
            pos[(P, k)] = (xx, v)
        ax.scatter(x[~sig], s[~sig], s=40, color=c, alpha=.55 if mem else .5, edgecolor=BG, lw=.5, zorder=3)
        ax.scatter(x[sig], s[sig], s=50, color=c, alpha=1, edgecolor=NAVY, lw=1.3, zorder=6)
        ax.boxplot(s.values, positions=[x0 + dx], widths=0.3, showfliers=False, patch_artist=True,
                   notch=True, boxprops=dict(facecolor="none", edgecolor=cb, lw=1.8),
                   whiskerprops=dict(color=cb, lw=1.6), capprops=dict(color=cb, lw=1.6),
                   medianprops=dict(color=NAVY, lw=3.2), zorder=5, manage_ticks=False)
        ax.text(x0 + dx, -13.3, lab, ha="center", va="top", fontsize=15, color=cb,
                fontweight="bold" if mem else "normal")
    ax.text(x0, -15.6, nome, ha="center", va="top", fontsize=18, color=NAVY, fontweight="bold")

ax.axhline(0, color=NAVY, lw=1.6, zorder=2, xmin=0.205)
ax.text(-1.45, 0, "chance", ha="left", va="center", fontsize=17, color=NAVY, fontweight="bold")
ax.text(-1.45, 1.6, "▲ follows more", ha="left", va="bottom", fontsize=14, color=base.GRIGIO_TESTO)
ax.text(-1.45, -1.6, "▼ follows less", ha="left", va="top", fontsize=14, color=base.GRIGIO_TESTO)


def etichetta(P, stato, testo, dx, dy, colore, ha="left"):
    x, y = pos[(P, stato)]
    ax.annotate(testo, (x, y), xytext=(x + dx, y + dy), fontsize=15, color=colore, ha=ha, va="center",
                arrowprops=dict(arrowstyle="-", color=base.GRIGIO_LINEA, lw=1, shrinkA=2, shrinkB=4))


etichetta("CHINA", "PAKISTAN", "Pakistan", 0.3, 0, base.CORALLO_TESTO)
x, _ = pos[("CHINA", "SYRIAN ARAB REPUBLIC")]
ax.text(x + 0.15, 25, "Syria · N. Korea · Iran", ha="left", va="center", fontsize=13, color=base.CORALLO_TESTO)
etichetta("UNITED STATES", "ISRAEL", "Israel", -0.35, 0, NAVY, "right")

ax.set_xlim(-1.5, 3.15)
ax.set_ylim(-17.5, 27)
ax.set_yticks([-10, 10, 20])
ax.set_yticklabels(["−10", "+10", "+20"])
ax.tick_params(axis="y", labelsize=15, colors=base.GRIGIO_TESTO, length=0)
ax.set_xticks([])
for s in ["top", "right", "left", "bottom"]:
    ax.spines[s].set_visible(False)
for yv in [-10, 10, 20]:
    ax.plot([-0.55, 3.15], [yv, yv], color=base.GRIGIO_GRIGLIA, lw=1, zorder=1)
ax.set_ylabel("Points above or below own chance", fontsize=15, color=base.GRIGIO_TESTO, labelpad=6)
fig.subplots_adjust(left=.09, right=.99, top=.98, bottom=.02)
base.salva(fig, "slide22_follow.png")
