"""Slide 30 - Mali: anno per anno, quanto vota con la Russia quando la Russia è in minoranza.

Barre: quota osservata, con intervallo di Wilson al 95%.
Cerchi: quota attesa per caso con le abitudini del Mali in quell'anno.
Linea tratteggiata: membro mediano del Movimento (Stati presenti in almeno il 75% delle votazioni).
Stampa anche il confronto 2014-21 / 2022-25 con il test binomiale contro il caso.
Output: figure/slide30_mali.png
"""
import numpy as np
from scipy.stats import binomtest
from statsmodels.stats.proportion import proportion_confint

import base


def atteso(g):
    """Accordo per caso: frequenze di sì/astenuto/no del paese per quelle della potenza."""
    fc = g.vote.value_counts(normalize=True)
    fp = g.pv.value_counts(normalize=True)
    return sum(fc.get(v, 0) * fp.get(v, 0) for v in [1, 2, 3])


df = base.carica()
pres = base.presenza(df)
d = base.voti_espressi(df)
maj = base.maggioranza(d)
vr, minoranza = base.in_minoranza(d, maj, "RUSSIAN FEDERATION")
s = d[d.rcid.isin(minoranza) & (d.Countryname != "RUSSIAN FEDERATION")].copy()
s["pv"] = vr.reindex(s.rcid).values
s["same"] = s.vote == s.pv
mali = s[s.Countryname == "MALI"]
membri = s[s.Countryname.isin(base.NAM) & s.Countryname.isin(pres[pres >= 0.75].index)]

anni = sorted(mali.year.unique())
oss, lo, hi, caso, med, n = [], [], [], [], [], []
for y in anni:
    g = mali[mali.year == y]
    k = int(g.same.sum())
    a, b = proportion_confint(k, len(g), method="wilson")
    oss.append(100 * k / len(g)); lo.append(100 * a); hi.append(100 * b)
    caso.append(100 * atteso(g)); n.append(len(g))
    med.append(100 * membri[membri.year == y].groupby("Countryname").same.mean().median())
    print(f"{y}: {k}/{len(g)} = {oss[-1]:.0f}% [{lo[-1]:.0f}-{hi[-1]:.0f}]  caso {caso[-1]:.0f}%  membro mediano {med[-1]:.1f}%")
for nome, g in [("2014-21", mali[mali.year <= 2021]), ("2022-25", mali[mali.year >= 2022])]:
    k, e = int(g.same.sum()), atteso(g)
    p = binomtest(k, len(g), e, alternative="greater").pvalue
    print(f"{nome}: {k}/{len(g)} = {100 * k / len(g):.0f}% contro {100 * e:.0f}% per caso, p = {p:.3f}")

# figura
plt = base.stile()
fig, ax = base.nuova_figura(plt, 10.4, 5.17)
x = np.arange(len(anni))
dopo = np.array(anni) >= 2022
ax.axvspan(x[dopo][0] - .5, x[-1] + .5, color="#FBE3DC", zorder=0, lw=0)
ax.bar(x, oss, width=.62, color=np.where(dopo, base.CORALLO, "#F4B8A8"), zorder=2)
ax.errorbar(x, oss, yerr=np.clip([np.array(oss) - lo, np.array(hi) - oss], 0, None), fmt="none",
            ecolor=base.NAVY, elinewidth=1.3, capsize=5, zorder=3)
ax.scatter(x, caso, s=80, facecolor="white", edgecolor=base.NAVY, lw=2, zorder=4, label="Mali by chance (its own habits)")
ax.plot(x, med, color=base.GRIGIO_LINEA, lw=2, ls=(0, (3, 2)), zorder=3, label="median Movement member")
for xv in [x[anni.index(2020)] + .5, x[anni.index(2021)] + .5]:
    ax.axvline(xv, color=base.NAVY, lw=1, ls=(0, (1, 2)), zorder=1)
ax.text(x[anni.index(2021)], 71, "coups\n2020 · 2021", ha="center", va="center", fontsize=13, color=base.NAVY)
picco = int(np.argmax(oss))
ax.text(x[picco], hi[picco] + 2.5, f"{oss[picco]:.0f}%", ha="center", va="bottom", fontsize=16,
        fontweight="bold", color=base.NAVY)
ax.text(x[-1] + .4, 76, "2022–25", ha="right", va="center", fontsize=14, fontweight="bold", color=base.CORALLO_TESTO)

ax.set_xticks(x)
ax.set_xticklabels([f"{y}\n(n {k})" for y, k in zip(anni, n)])
ax.tick_params(axis="x", labelsize=12, colors=base.NAVY, length=0)
ax.tick_params(axis="y", labelsize=12, colors=base.NAVY, length=0)
ax.set_xlim(-.5, x[-1] + .5)
ax.set_ylim(0, 80)
ax.set_yticks(range(0, 81, 10))
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
ax.spines["left"].set_color(base.NAVY)
ax.spines["bottom"].set_color(base.NAVY)
ax.grid(axis="y", color=base.GRIGIO_GRIGLIA, lw=1)
ax.set_axisbelow(True)
ax.set_ylabel("% of Russia-in-minority votes\nin which Mali votes with Russia", fontsize=12.5, color=base.NAVY)
ax.legend(loc="upper left", frameon=False, fontsize=13, labelcolor=base.NAVY)
fig.subplots_adjust(left=.09, right=.99, top=.97, bottom=.15)
base.salva(fig, "slide30_mali.png")
