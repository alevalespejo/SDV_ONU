"""Slide 25 - Il 2022 ha cambiato chi segue chi?

Per USA, Cina e Russia: quota di votazioni in cui ogni Stato vota come la potenza quando
la potenza è in minoranza, prima (2014-21) e dopo (2022-25). Mediana di membri e non membri
e test di Wilcoxon appaiato sui membri (prima contro dopo).
Sulla slide i valori sono riportati come barre; il grafico a manubri qui sotto mostra gli
stessi numeri.
Output: figure/s25_follow.csv, figure/slide25_2022.png
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import base

df = base.carica()
d = base.voti_espressi(df)
maj = base.maggioranza(d)
POTENZE = {"US": "UNITED STATES", "China": "CHINA", "Russia": "RUSSIAN FEDERATION"}

tabelle = []
for p, stato in POTENZE.items():
    vp, minoranza = base.in_minoranza(d, maj, stato)
    sub = d[d.rcid.isin(minoranza) & (d.Countryname != stato)].copy()
    sub["same"] = sub.vote.values == vp.reindex(sub.rcid).values
    sub["per"] = np.where(sub.year <= 2021, "before", "after")
    print(p, "votazioni in minoranza:", len(minoranza), sub.groupby("per").rcid.nunique().to_dict())
    t = sub.groupby(["Countryname", "per"]).same.mean().unstack() * 100
    t["member"] = t.index.isin(base.NAM)
    t["power"] = p
    tabelle.append(t)
R = pd.concat(tabelle)
M = R.groupby(["power", "member"])[["before", "after"]].median()
print(M.round(1))
for p, g in R[R.member].groupby("power"):
    g = g.dropna(subset=["before", "after"])
    print(f"Wilcoxon membri {p}: p = {wilcoxon(g.after, g.before).pvalue:.3f} (n = {len(g)})")
R.to_csv(base.OUT / "s25_follow.csv")

# figura
plt = base.stile()
BG, NAVY = base.SFONDO, base.NAVY
fig, ax = base.nuova_figura(plt, 11, 5.6)
ordine = ["US", "China", "Russia"]
y0 = {p: i for i, p in enumerate(ordine[::-1])}
for p in ordine:
    for mem, dy, c, ct in [(True, 0.17, base.CORALLO, base.CORALLO_TESTO),
                           (False, -0.17, base.GRIGIO_LINEA, base.GRIGIO_TESTO)]:
        prima, dopo = M.loc[(p, mem)]
        y = y0[p] + dy
        if abs(dopo - prima) >= 1.5:
            ax.annotate("", xy=(dopo, y), xytext=(prima, y), zorder=2,
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=2.4, mutation_scale=16, shrinkA=7, shrinkB=7))
        ax.scatter([prima], [y], s=110, facecolor=BG, edgecolor=c, lw=2.2, zorder=3)
        ax.scatter([dopo], [y], s=110, color=c, zorder=3)
        ax.text(max(prima, dopo) + 1.2, y, f"{prima:.0f}% → {dopo:.0f}%", ha="left", va="center",
                fontsize=15, color=ct, fontweight="bold")
    ax.text(-1.2, y0[p], p + " in\nminority", ha="right", va="center", fontsize=17, color=NAVY, fontweight="bold")

ax.set_xlim(0, 50)
ax.set_ylim(-0.55, 2.8)
base.asse_x_pulito(ax, [0, 10, 20, 30, 40, 50], ["0", "10", "20", "30", "40", "50%"])
ax.set_xlabel("Share of votes in which the median country votes like the power, when the power is in the minority",
              fontsize=13, color=base.GRIGIO_TESTO, labelpad=8)

# legenda
ky = 2.62
ax.scatter([27.5], [ky], s=80, facecolor=BG, edgecolor=NAVY, lw=2)
ax.text(28.5, ky, "2014–21", va="center", fontsize=13, color=NAVY)
ax.annotate("", xy=(35.4, ky), xytext=(33.0, ky),
            arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.6, mutation_scale=12))
ax.scatter([36.4], [ky], s=80, color=NAVY)
ax.text(37.4, ky, "2022–25", va="center", fontsize=13, color=NAVY)
ax.scatter([27.5], [ky - 0.24], s=80, color=base.CORALLO)
ax.text(28.5, ky - 0.24, "Movement members", va="center", fontsize=13, color=base.CORALLO_TESTO)
ax.scatter([39.5], [ky - 0.24], s=80, color=base.GRIGIO_LINEA)
ax.text(40.5, ky - 0.24, "non-members", va="center", fontsize=13, color=base.GRIGIO_TESTO)
fig.subplots_adjust(left=.13, right=.98, top=.97, bottom=.14)
base.salva(fig, "slide25_2022.png")
